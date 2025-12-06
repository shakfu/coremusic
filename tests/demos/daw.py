#!/usr/bin/env python3
"""Demo for DAW (Digital Audio Workstation) essentials module.

This script demonstrates the coremusic.daw module capabilities:
- Multi-track timeline creation
- Clip management
- Automation
- Markers and loop regions
- Transport control
"""

from pathlib import Path

BUILD_DIR = Path.cwd() / "build"

import coremusic as cm
from coremusic.daw import (
    Timeline,
    Track,
    Clip,
    MIDIClip,
    MIDINote,
    AudioUnitPlugin,
    TimelineMarker,
    TimeRange,
    AutomationLane,
)
from coremusic.capi import (
    fourchar_to_int,
    extended_audio_file_create_with_url,
    extended_audio_file_write,
    extended_audio_file_dispose,
    get_linear_pcm_format_flag_is_signed_integer,
    get_linear_pcm_format_flag_is_packed,
)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available - audio generation will be skipped")

# Output directory for DAW demo files
OUTPUT_DIR = BUILD_DIR / "daw_output"


# ============================================================================
# Audio Generation Helpers
# ============================================================================

def generate_drum_pattern(duration, sample_rate=48000, tempo=128.0):
    """Generate a punchy electronic drum pattern"""
    if not NUMPY_AVAILABLE:
        return None

    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)

    beat_interval = int((60.0 / tempo) * sample_rate)
    sixteenth = beat_interval // 4

    # Number of bars to fill
    num_beats = int(duration * tempo / 60)

    for beat in range(num_beats):
        pos = beat * beat_interval

        # Kick on 1 and 3 (with pitch sweep for punch)
        if beat % 4 in [0, 2]:
            if pos < num_samples:
                kick_len = min(int(0.15 * sample_rate), num_samples - pos)
                t = np.arange(kick_len) / sample_rate
                # Pitch drops from 150Hz to 50Hz for punch
                freq = 150 * np.exp(-t * 30) + 50
                phase = np.cumsum(2 * np.pi * freq / sample_rate)
                kick = np.sin(phase) * np.exp(-t * 15) * 0.9
                audio[pos:pos + kick_len] += kick

        # Snare on 2 and 4 (layered noise + tone)
        if beat % 4 in [1, 3]:
            if pos < num_samples:
                snare_len = min(int(0.12 * sample_rate), num_samples - pos)
                t = np.arange(snare_len) / sample_rate
                # Body tone
                body = np.sin(2 * np.pi * 180 * t) * np.exp(-t * 25)
                # Noise snap
                noise = np.random.randn(snare_len) * np.exp(-t * 35)
                snare = (body * 0.4 + noise * 0.3) * 0.7
                audio[pos:pos + snare_len] += snare

        # Hi-hat on every eighth note
        for eighth in [0, 2]:
            hat_pos = pos + eighth * (beat_interval // 2)
            if hat_pos < num_samples:
                hat_len = min(int(0.04 * sample_rate), num_samples - hat_pos)
                t = np.arange(hat_len) / sample_rate
                # Filtered noise for hi-hat
                noise = np.random.randn(hat_len)
                hat = noise * np.exp(-t * 80) * 0.25
                audio[hat_pos:hat_pos + hat_len] += hat

    return audio


def generate_bass_line(duration, sample_rate=48000, tempo=128.0):
    """Generate a melodic bass line in A minor"""
    if not NUMPY_AVAILABLE:
        return None

    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)

    beat_duration = 60.0 / tempo

    # A minor bass pattern (A-E-F-G progression, musically pleasing)
    # MIDI notes: A1=33, E2=40, F2=41, G2=43
    pattern = [
        (33, 0.0, 0.9),   # A1 - root
        (33, 1.0, 0.4),   # A1 - octave hit
        (40, 1.5, 0.9),   # E2 - fifth
        (41, 2.5, 0.9),   # F2 - minor sixth
        (43, 3.5, 0.4),   # G2 - minor seventh
    ]

    # Repeat pattern for duration
    pattern_length = 4 * beat_duration
    num_repeats = int(duration / pattern_length) + 1

    for repeat in range(num_repeats):
        for midi_note, beat_offset, note_dur in pattern:
            note_start = repeat * pattern_length + beat_offset * beat_duration
            if note_start >= duration:
                break

            freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
            start_sample = int(note_start * sample_rate)
            dur_samples = int(note_dur * beat_duration * sample_rate)
            end_sample = min(start_sample + dur_samples, num_samples)

            if start_sample >= num_samples:
                break

            note_t = np.arange(end_sample - start_sample) / sample_rate

            # Warm bass with sub and slight harmonics
            fundamental = np.sin(2 * np.pi * freq * note_t)
            sub = np.sin(2 * np.pi * freq * 0.5 * note_t) * 0.5
            harmonic = np.sin(2 * np.pi * freq * 2 * note_t) * 0.15

            # Punchy envelope with quick attack
            envelope = (1 - np.exp(-note_t * 50)) * np.exp(-note_t * 4)
            bass_note = (fundamental + sub + harmonic) * envelope * 0.5
            audio[start_sample:end_sample] += bass_note

    return audio


def generate_synth_pad(duration, sample_rate=48000):
    """Generate a lush ambient pad with chord progression"""
    if not NUMPY_AVAILABLE:
        return None

    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)
    t = np.arange(num_samples) / sample_rate

    # A minor chord progression: Am - F - C - G (each 2 beats at ~60 BPM for ambient)
    # Using longer, overlapping chords for pad feel
    chord_duration = duration / 4

    chords = [
        [57, 60, 64],      # Am (A3, C4, E4)
        [53, 57, 60],      # F (F3, A3, C4)
        [48, 52, 55, 60],  # C (C3, E3, G3, C4)
        [55, 59, 62],      # G (G3, B3, D4)
    ]

    for chord_idx, chord_notes in enumerate(chords):
        chord_start = chord_idx * chord_duration
        chord_end = min(chord_start + chord_duration * 1.2, duration)  # Slight overlap

        start_sample = int(chord_start * sample_rate)
        end_sample = int(chord_end * sample_rate)
        if start_sample >= num_samples:
            break
        end_sample = min(end_sample, num_samples)

        chord_t = np.arange(end_sample - start_sample) / sample_rate
        chord_audio = np.zeros(len(chord_t), dtype=np.float32)

        for midi_note in chord_notes:
            freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

            # Detuned oscillators for richness
            for detune in [-0.02, 0, 0.02]:
                detuned_freq = freq * (1 + detune)
                # Slow LFO for movement
                lfo = 1 + 0.003 * np.sin(2 * np.pi * 0.3 * chord_t)
                osc = np.sin(2 * np.pi * detuned_freq * lfo * chord_t)
                chord_audio += osc * 0.08

        # Soft envelope with slow attack and release
        attack = 0.8
        release = 0.5
        env = np.ones(len(chord_t))
        attack_samples = int(attack * sample_rate)
        release_samples = int(release * sample_rate)

        if attack_samples < len(env):
            env[:attack_samples] = np.linspace(0, 1, attack_samples) ** 0.5
        if release_samples < len(env):
            env[-release_samples:] *= np.linspace(1, 0, release_samples) ** 0.5

        audio[start_sample:end_sample] += chord_audio * env

    # Overall fade in/out
    fade_samples = int(1.5 * sample_rate)
    if fade_samples < num_samples:
        audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
        audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    return audio


def generate_vocal_melody(duration, sample_rate=48000, tempo=128.0):
    """Generate a beautiful lead melody in A minor"""
    if not NUMPY_AVAILABLE:
        return None

    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)
    beat_duration = 60.0 / tempo

    # Melodic phrase in A minor - singable, memorable melody
    # (MIDI note, beat start, beat duration)
    melody = [
        # Phrase 1: Rising
        (64, 0.0, 1.0),    # E4
        (67, 1.0, 0.5),    # G4
        (69, 1.5, 1.5),    # A4
        (72, 3.0, 1.0),    # C5
        # Phrase 2: Falling resolution
        (71, 4.0, 0.5),    # B4
        (69, 4.5, 0.5),    # A4
        (67, 5.0, 1.0),    # G4
        (64, 6.0, 2.0),    # E4 (held)
        # Phrase 3: Variation
        (69, 8.0, 0.75),   # A4
        (71, 8.75, 0.25),  # B4
        (72, 9.0, 1.0),    # C5
        (74, 10.0, 0.5),   # D5
        (72, 10.5, 0.5),   # C5
        (69, 11.0, 1.0),   # A4
        # Phrase 4: Resolution
        (67, 12.0, 1.0),   # G4
        (69, 13.0, 3.0),   # A4 (final, held)
    ]

    pattern_beats = 16
    pattern_duration = pattern_beats * beat_duration
    num_repeats = int(duration / pattern_duration) + 1

    for repeat in range(num_repeats):
        for midi_note, beat_start, beat_dur in melody:
            note_start = repeat * pattern_duration + beat_start * beat_duration
            if note_start >= duration:
                break

            freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
            start_sample = int(note_start * sample_rate)
            dur_samples = int(beat_dur * beat_duration * sample_rate)
            end_sample = min(start_sample + dur_samples, num_samples)

            if start_sample >= num_samples:
                break

            note_t = np.arange(end_sample - start_sample) / sample_rate

            # Expressive vibrato that develops over time
            vibrato_depth = 0.012 * (1 - np.exp(-note_t * 3))  # Delayed vibrato
            vibrato = 1 + vibrato_depth * np.sin(2 * np.pi * 5.5 * note_t)

            # Rich tone with harmonics
            fundamental = np.sin(2 * np.pi * freq * vibrato * note_t)
            harmonic2 = 0.3 * np.sin(2 * np.pi * freq * 2 * vibrato * note_t)
            harmonic3 = 0.1 * np.sin(2 * np.pi * freq * 3 * vibrato * note_t)

            # Natural envelope with attack, sustain, release
            attack_time = 0.02
            release_time = 0.15
            note_len = len(note_t)
            attack_samples = int(attack_time * sample_rate)
            release_samples = int(release_time * sample_rate)

            env = np.ones(note_len)
            if attack_samples < note_len:
                env[:attack_samples] = np.linspace(0, 1, attack_samples) ** 0.7
            if release_samples < note_len:
                env[-release_samples:] *= np.linspace(1, 0, release_samples) ** 0.5

            note_audio = (fundamental + harmonic2 + harmonic3) * env * 0.4
            audio[start_sample:end_sample] += note_audio

    return audio


def render_midi_to_audio(midi_clip, duration, sample_rate=48000, instrument_type="piano"):
    """Render MIDI clip to audio using synthesized instrument.

    Args:
        midi_clip: MIDIClip with notes
        duration: Total duration in seconds
        sample_rate: Sample rate
        instrument_type: 'piano', 'synth', 'bass', etc.

    Returns:
        Audio data as numpy array
    """
    if not NUMPY_AVAILABLE:
        return None

    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)

    # Render each note
    for note in midi_clip.notes:
        start_sample = int(note.start_time * sample_rate)
        duration_samples = int(note.duration * sample_rate)
        end_sample = min(start_sample + duration_samples, num_samples)

        if start_sample >= num_samples:
            continue

        note_samples = end_sample - start_sample
        t = np.arange(note_samples) / sample_rate

        # Convert MIDI note to frequency (A4 = 440Hz is MIDI note 69)
        freq = 440.0 * (2.0 ** ((note.note - 69) / 12.0))

        # Velocity scaling
        vel_scale = note.velocity / 127.0

        # Generate different sounds based on instrument type
        if instrument_type == "piano":
            # Piano-like sound with multiple harmonics and decay
            fundamental = np.sin(2 * np.pi * freq * t)
            harmonic2 = 0.3 * np.sin(2 * np.pi * freq * 2 * t)
            harmonic3 = 0.15 * np.sin(2 * np.pi * freq * 3 * t)
            envelope = np.exp(-t * 3.0)  # Fast decay
            note_audio = (fundamental + harmonic2 + harmonic3) * envelope * vel_scale * 0.5

        elif instrument_type == "synth":
            # Synth pad with slower envelope
            fundamental = np.sin(2 * np.pi * freq * t)
            harmonic2 = 0.5 * np.sin(2 * np.pi * freq * 2 * t)
            envelope = np.exp(-t * 1.0)  # Slower decay
            note_audio = (fundamental + harmonic2) * envelope * vel_scale * 0.4

        elif instrument_type == "bass":
            # Bass with emphasis on low frequencies
            fundamental = np.sin(2 * np.pi * freq * t)
            subharmonic = 0.4 * np.sin(2 * np.pi * freq * 0.5 * t)
            envelope = np.exp(-t * 2.0)
            note_audio = (fundamental + subharmonic) * envelope * vel_scale * 0.6

        else:
            # Default: simple sine wave
            envelope = np.exp(-t * 2.0)
            note_audio = np.sin(2 * np.pi * freq * t) * envelope * vel_scale * 0.5

        # Add to output
        audio[start_sample:end_sample] += note_audio

    return audio


def apply_delay_effect(audio, sample_rate, delay_time=0.25, feedback=0.4, mix=0.3):
    """Apply a simple delay effect to audio.

    Args:
        audio: Input audio data
        sample_rate: Sample rate
        delay_time: Delay time in seconds
        feedback: Feedback amount (0-1)
        mix: Dry/wet mix (0=dry, 1=wet)

    Returns:
        Audio with delay effect applied
    """
    if not NUMPY_AVAILABLE or audio is None:
        return audio

    delay_samples = int(delay_time * sample_rate)
    output = audio.copy()

    # Simple delay with feedback
    for i in range(delay_samples, len(audio)):
        output[i] += feedback * output[i - delay_samples]

    # Mix dry and wet
    return (1 - mix) * audio + mix * output


def apply_reverb_effect(audio, sample_rate, room_size=0.5, damping=0.5, mix=0.3):
    """Apply a simple reverb effect using comb filters.

    Args:
        audio: Input audio data
        sample_rate: Sample rate
        room_size: Room size (0-1)
        damping: High frequency damping (0-1)
        mix: Dry/wet mix

    Returns:
        Audio with reverb applied
    """
    if not NUMPY_AVAILABLE or audio is None:
        return audio

    # Simple comb filter delays (Freeverb-inspired)
    delay_times = [0.0297, 0.0371, 0.0411, 0.0437]

    output = np.zeros_like(audio)

    for delay_time in delay_times:
        scaled_delay = delay_time * room_size
        delay_samples = int(scaled_delay * sample_rate)

        if delay_samples < len(audio):
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples]

            # Apply simple lowpass (damping)
            if damping > 0:
                for i in range(1, len(delayed)):
                    delayed[i] = delayed[i] * (1 - damping) + delayed[i-1] * damping

            output += delayed

    output = output / len(delay_times)

    # Mix dry and wet
    return (1 - mix) * audio + mix * output


def write_audio_file(audio_data, sample_rate, file_path, num_channels=2):
    """Write audio data to a WAV file"""
    if audio_data is None:
        return

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure audio_data is 1D for mono
    if audio_data.ndim == 1 and num_channels == 2:
        # Mono to stereo by duplicating
        audio_data = np.repeat(audio_data, 2)

    # Normalize to int16 range
    audio_data = np.clip(audio_data, -1.0, 1.0)
    audio_int16 = (audio_data * 32767).astype(np.int16)

    # Create Extended Audio File for writing
    asbd = {
        'sample_rate': float(sample_rate),
        'format_id': fourchar_to_int('lpcm'),
        'format_flags': (
            get_linear_pcm_format_flag_is_signed_integer() |
            get_linear_pcm_format_flag_is_packed()
        ),
        'bytes_per_packet': num_channels * 2,
        'frames_per_packet': 1,
        'bytes_per_frame': num_channels * 2,
        'channels_per_frame': num_channels,
        'bits_per_channel': 16
    }

    file_type = fourchar_to_int('WAVE')
    ext_audio_file_id = extended_audio_file_create_with_url(
        str(file_path), file_type, asbd
    )

    try:
        num_frames = len(audio_int16) // num_channels
        extended_audio_file_write(ext_audio_file_id, num_frames, audio_int16.tobytes())
    finally:
        extended_audio_file_dispose(ext_audio_file_id)


def render_timeline_to_audio(timeline, sample_rate=48000):
    """Render a timeline with automation to audio"""
    if not NUMPY_AVAILABLE:
        return None

    duration = timeline.get_duration()
    num_samples = int(duration * sample_rate)
    mixed_audio = np.zeros(num_samples, dtype=np.float32)

    # Render each track
    for track in timeline.tracks:
        track_audio = np.zeros(num_samples, dtype=np.float32)

        # Render all clips on this track
        for clip in track.clips:
            start_time = clip.start_time

            # Generate audio based on clip source name
            source = str(clip.source)
            clip_duration = clip.duration

            if 'drum' in source.lower():
                clip_audio = generate_drum_pattern(clip_duration, sample_rate, timeline.tempo)
            elif 'bass' in source.lower():
                clip_audio = generate_bass_line(clip_duration, sample_rate, timeline.tempo)
            elif 'synth' in source.lower():
                clip_audio = generate_synth_pad(clip_duration, sample_rate)
            elif 'vocal' in source.lower():
                clip_audio = generate_vocal_melody(clip_duration, sample_rate, timeline.tempo)
            else:
                # Generic tone
                t = np.arange(int(clip_duration * sample_rate)) / sample_rate
                clip_audio = np.sin(2 * np.pi * 440 * t) * 0.3

            if clip_audio is None:
                continue

            # Apply clip gain
            clip_audio = clip_audio * clip.gain

            # Apply fades
            if clip.fade_in > 0:
                fade_samples = int(clip.fade_in * sample_rate)
                fade_in_curve = np.linspace(0, 1, min(fade_samples, len(clip_audio)))
                clip_audio[:len(fade_in_curve)] *= fade_in_curve

            if clip.fade_out > 0:
                fade_samples = int(clip.fade_out * sample_rate)
                fade_out_curve = np.linspace(1, 0, min(fade_samples, len(clip_audio)))
                clip_audio[-len(fade_out_curve):] *= fade_out_curve

            # Place clip in track timeline
            start_sample = int(start_time * sample_rate)
            end_sample = min(start_sample + len(clip_audio), num_samples)
            clip_len = end_sample - start_sample
            track_audio[start_sample:end_sample] += clip_audio[:clip_len]

        # Apply track volume
        track_audio = track_audio * track.volume

        # Apply volume automation if present
        if 'volume' in track.automation:
            volume_auto = track.automation['volume']
            for i in range(num_samples):
                time = i / sample_rate
                auto_value = volume_auto.get_value(time)
                track_audio[i] *= auto_value

        # Apply pan automation if present
        # (For stereo, we'd need to handle left/right separately - simplified here)

        # Mix into master
        mixed_audio += track_audio

    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 0:
        mixed_audio = mixed_audio / max_val * 0.9

    return mixed_audio


# ============================================================================
# Demo Functions
# ============================================================================

def demo_midi_clip():
    """Demo 0a: MIDI clip with piano rendering"""
    print("\n" + "=" * 60)
    print("Demo 0a: MIDI Clip with Piano Rendering")
    print("=" * 60)

    if not NUMPY_AVAILABLE:
        print("Skipping - NumPy required for audio generation")
        return

    # Create MIDI clip with a beautiful melody (Clair de Lune inspired)
    midi_clip = MIDIClip()

    # Expressive melody in Db major - gentle, flowing
    melody = [
        # Opening phrase
        (61, 0.0, 0.8, 70),    # Db4
        (63, 0.8, 0.4, 65),    # Eb4
        (65, 1.2, 0.6, 75),    # F4
        (68, 1.8, 1.0, 80),    # Ab4
        (66, 2.8, 0.6, 70),    # Gb4
        (65, 3.4, 0.8, 65),    # F4
        # Second phrase - rising
        (63, 4.2, 0.4, 60),    # Eb4
        (65, 4.6, 0.4, 70),    # F4
        (68, 5.0, 0.8, 85),    # Ab4
        (70, 5.8, 1.2, 90),    # Bb4
        (68, 7.0, 0.6, 75),    # Ab4
        # Resolution
        (66, 7.6, 0.8, 70),    # Gb4
        (65, 8.4, 1.6, 65),    # F4 (held)
    ]

    for note, start, dur, vel in melody:
        midi_clip.add_note(note, vel, start, dur)

    print(f"Created MIDI clip with {len(midi_clip.notes)} notes (expressive melody)")
    print(f"  Duration: 10 seconds")
    print(f"  Key: Db major")
    print(f"  First 4 notes:")
    for i, note in enumerate(midi_clip.notes[:4], 1):
        note_name = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'][note.note % 12]
        octave = note.note // 12 - 1
        print(f"    {i}. {note_name}{octave} (MIDI {note.note}), "
              f"t={note.start_time:.1f}s, vel={note.velocity}")

    # Render to audio with piano sound
    print(f"\n  Rendering MIDI to audio with piano instrument...")
    audio = render_midi_to_audio(midi_clip, duration=10.5, instrument_type="piano")

    if audio is not None:
        output_path = OUTPUT_DIR / "midi_piano_melody.wav"
        write_audio_file(audio, 48000, output_path)
        print(f"  Saved: {output_path.name}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


def demo_midi_instruments():
    """Demo 0b: MIDI with different instrument types"""
    print("\n" + "=" * 60)
    print("Demo 0b: MIDI with Different Instruments")
    print("=" * 60)

    if not NUMPY_AVAILABLE:
        print("Skipping - NumPy required for audio generation")
        return

    # Create a beautiful chord progression with voice leading
    midi_clip = MIDIClip()

    # Dm9 chord (D, F, A, C, E) - rich voicing
    for note in [50, 53, 57, 60, 64]:  # D3, F3, A3, C4, E4
        midi_clip.add_note(note, 75, 0.0, 2.0)

    # G7 chord (G, B, D, F) - dominant
    for note in [43, 47, 50, 53]:  # G2, B2, D3, F3
        midi_clip.add_note(note, 80, 2.0, 2.0)

    # Cmaj7 chord (C, E, G, B) - resolution
    for note in [48, 52, 55, 59]:  # C3, E3, G3, B3
        midi_clip.add_note(note, 85, 4.0, 2.0)

    # Am7 chord (A, C, E, G) - minor color
    for note in [45, 48, 52, 55]:  # A2, C3, E3, G3
        midi_clip.add_note(note, 70, 6.0, 2.5)

    print(f"Created jazz chord progression with {len(midi_clip.notes)} notes")
    print(f"  Progression: Dm9 - G7 - Cmaj7 - Am7 (ii-V-I-vi)")

    # Render with different instruments
    instruments = [
        ("piano", "Piano"),
        ("synth", "Synthesizer"),
        ("bass", "Bass"),
    ]

    for inst_type, inst_name in instruments:
        print(f"\n  Rendering with {inst_name}...")
        audio = render_midi_to_audio(midi_clip, duration=9.0, instrument_type=inst_type)

        if audio is not None:
            output_path = OUTPUT_DIR / f"midi_chords_{inst_type}.wav"
            write_audio_file(audio, 48000, output_path)
            print(f"    Saved: {output_path.name}")


def demo_audio_effects():
    """Demo 0c: Audio effects processing (delay and reverb)"""
    print("\n" + "=" * 60)
    print("Demo 0c: Audio Effects Processing")
    print("=" * 60)

    if not NUMPY_AVAILABLE:
        print("Skipping - NumPy required for audio generation")
        return

    # Generate a melodic arpeggio pattern perfect for effects
    sample_rate = 48000
    duration = 6.0

    # E minor arpeggio pattern (great for delay/reverb demos)
    # MIDI notes with timing for rhythmic interest
    notes = [
        (64, 0.0, 0.3),    # E4
        (67, 0.25, 0.3),   # G4
        (71, 0.5, 0.3),    # B4
        (76, 0.75, 0.5),   # E5

        (64, 1.5, 0.3),    # E4
        (69, 1.75, 0.3),   # A4
        (72, 2.0, 0.3),    # C5
        (76, 2.25, 0.5),   # E5

        (62, 3.0, 0.3),    # D4
        (67, 3.25, 0.3),   # G4
        (71, 3.5, 0.3),    # B4
        (74, 3.75, 0.5),   # D5

        (64, 4.5, 0.3),    # E4
        (67, 4.75, 0.3),   # G4
        (71, 5.0, 0.8),    # B4 (held)
    ]

    audio = np.zeros(int(duration * sample_rate), dtype=np.float32)

    for midi_note, start_time, note_dur in notes:
        freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        start_sample = int(start_time * sample_rate)
        dur_samples = int(note_dur * sample_rate)
        end_sample = min(start_sample + dur_samples, len(audio))

        note_t = np.arange(end_sample - start_sample) / sample_rate

        # Bell-like tone with harmonics
        fundamental = np.sin(2 * np.pi * freq * note_t)
        harmonic2 = 0.4 * np.sin(2 * np.pi * freq * 2 * note_t)
        harmonic3 = 0.2 * np.sin(2 * np.pi * freq * 3 * note_t)

        envelope = np.exp(-note_t * 4.0)
        note_audio = (fundamental + harmonic2 + harmonic3) * envelope * 0.35
        audio[start_sample:end_sample] += note_audio

    print(f"Generated arpeggio melody ({duration}s) in E minor")

    # Save original
    output_path = OUTPUT_DIR / "effects_original.wav"
    write_audio_file(audio, sample_rate, output_path)
    print(f"  Saved original: {output_path.name}")

    # Apply delay effect
    print(f"\n  Applying delay effect (250ms, 40% feedback)...")
    delay_audio = apply_delay_effect(audio, sample_rate, delay_time=0.25, feedback=0.4, mix=0.4)
    output_path = OUTPUT_DIR / "effects_with_delay.wav"
    write_audio_file(delay_audio, sample_rate, output_path)
    print(f"    Saved: {output_path.name}")

    # Apply reverb effect
    print(f"\n  Applying reverb effect (medium room)...")
    reverb_audio = apply_reverb_effect(audio, sample_rate, room_size=0.6, damping=0.4, mix=0.4)
    output_path = OUTPUT_DIR / "effects_with_reverb.wav"
    write_audio_file(reverb_audio, sample_rate, output_path)
    print(f"    Saved: {output_path.name}")

    # Apply both effects
    print(f"\n  Applying delay + reverb...")
    combo_audio = apply_delay_effect(audio, sample_rate, delay_time=0.25, feedback=0.3, mix=0.3)
    combo_audio = apply_reverb_effect(combo_audio, sample_rate, room_size=0.5, damping=0.5, mix=0.3)
    output_path = OUTPUT_DIR / "effects_delay_reverb.wav"
    write_audio_file(combo_audio, sample_rate, output_path)
    print(f"    Saved: {output_path.name}")


def demo_basic_timeline():
    """Demo 1: Basic timeline creation and track management"""
    print("\n" + "=" * 60)
    print("Demo 1: Basic Timeline Creation")
    print("=" * 60)

    # Create timeline
    timeline = Timeline(sample_rate=48000, tempo=128.0)
    print(f"Created timeline: {timeline}")

    # Add tracks
    drums = timeline.add_track("Drums", "audio")
    bass = timeline.add_track("Bass", "audio")
    vocals = timeline.add_track("Vocals", "audio")

    print(f"\nAdded {len(timeline.tracks)} tracks:")
    for track in timeline.tracks:
        print(f"  - {track}")

    # Add clips
    drums_clip = Clip("drums.wav")
    drums_clip.duration = 32.0
    drums.add_clip(drums_clip, start_time=0.0)

    vocals_clip = Clip("vocals.wav")
    vocals_clip.duration = 24.0
    vocals.add_clip(vocals_clip, start_time=8.0)

    print(f"\nClips on Drums track: {len(drums.clips)}")
    print(f"Clips on Vocals track: {len(vocals.clips)}")
    print(f"Timeline duration: {timeline.get_duration():.2f}s")


def demo_clip_management():
    """Demo 2: Clip trimming and fades"""
    print("\n" + "=" * 60)
    print("Demo 2: Clip Management - Trimming and Fades")
    print("=" * 60)

    # Create clip with method chaining
    clip = Clip("audio.wav").trim(2.0, 10.0).set_fades(fade_in=0.5, fade_out=1.0)

    print(f"Clip: {clip}")
    print(f"  Offset: {clip.offset}s")
    print(f"  Duration: {clip.duration}s")
    print(f"  Fade in: {clip.fade_in}s")
    print(f"  Fade out: {clip.fade_out}s")
    print(f"  Gain: {clip.gain}")

    # Adjust gain
    clip.gain = 0.8
    print(f"\nAdjusted gain to: {clip.gain}")


def demo_automation():
    """Demo 3: Automation lanes"""
    print("\n" + "=" * 60)
    print("Demo 3: Automation Lanes")
    print("=" * 60)

    # Create track and automation
    track = Track("Vocals", "audio")
    volume_automation = track.automate("volume")

    # Add automation points for fade in/out
    volume_automation.add_point(0.0, 0.0)    # Silent at start
    volume_automation.add_point(2.0, 1.0)    # Fade in over 2s
    volume_automation.add_point(58.0, 1.0)   # Hold
    volume_automation.add_point(60.0, 0.0)   # Fade out over 2s

    print(f"Created automation lane: {volume_automation}")
    print(f"Automation points: {len(volume_automation.points)}")

    # Test interpolation at various times
    print("\nVolume at different times (linear interpolation):")
    for time in [0.0, 1.0, 2.0, 30.0, 58.0, 59.0, 60.0]:
        value = volume_automation.get_value(time)
        print(f"  t={time:5.1f}s: volume={value:.3f}")

    # Try different interpolation modes
    print("\nComparing interpolation modes at t=1.0s:")
    for mode in ["linear", "step", "cubic"]:
        volume_automation.interpolation = mode
        value = volume_automation.get_value(1.0)
        print(f"  {mode:6s}: {value:.3f}")


def demo_markers_and_loops():
    """Demo 4: Markers and loop regions"""
    print("\n" + "=" * 60)
    print("Demo 4: Markers and Loop Regions")
    print("=" * 60)

    timeline = Timeline(tempo=120.0)

    # Add structural markers
    timeline.add_marker(0.0, "Intro")
    timeline.add_marker(16.0, "Verse 1")
    timeline.add_marker(32.0, "Chorus", color="#FF0000")
    timeline.add_marker(48.0, "Verse 2")
    timeline.add_marker(64.0, "Bridge", color="#00FF00")
    timeline.add_marker(80.0, "Outro")

    print(f"Added {len(timeline.markers)} markers:")
    for marker in timeline.markers:
        print(f"  {marker}")

    # Set loop region for chorus
    timeline.set_loop_region(32.0, 48.0)
    print(f"\nLoop region: {timeline.loop_region}")

    # Get markers in a range
    markers_in_range = timeline.get_markers_in_range(30.0, 50.0)
    print(f"\nMarkers between 30s and 50s:")
    for marker in markers_in_range:
        print(f"  {marker}")


def demo_transport_control():
    """Demo 5: Transport control"""
    print("\n" + "=" * 60)
    print("Demo 5: Transport Control")
    print("=" * 60)

    timeline = Timeline()

    # Play
    print(f"Initial state: playing={timeline.is_playing}, playhead={timeline.playhead}s")

    timeline.play()
    print(f"After play(): playing={timeline.is_playing}, playhead={timeline.playhead}s")

    # Simulate playhead movement
    timeline.playhead = 10.0
    print(f"After seeking: playhead={timeline.playhead}s")

    # Pause
    timeline.pause()
    print(f"After pause(): playing={timeline.is_playing}, playhead={timeline.playhead}s")

    # Resume from current position
    timeline.play()
    print(f"After play(): playing={timeline.is_playing}, playhead={timeline.playhead}s")

    # Stop (resets playhead)
    timeline.stop()
    print(f"After stop(): playing={timeline.is_playing}, playhead={timeline.playhead}s")

    # Play from specific time
    timeline.play(from_time=20.0)
    print(f"After play(from_time=20.0): playhead={timeline.playhead}s")


def demo_recording():
    """Demo 6: Recording setup"""
    print("\n" + "=" * 60)
    print("Demo 6: Recording Setup")
    print("=" * 60)

    timeline = Timeline()

    # Add tracks
    vocals = timeline.add_track("Vocals", "audio")
    guitar = timeline.add_track("Guitar", "audio")
    bass = timeline.add_track("Bass", "audio")

    print(f"Created {len(timeline.tracks)} tracks")

    # Arm tracks for recording
    vocals.record_enable(True)
    guitar.record_enable(True)

    print("\nArmed tracks for recording:")
    for track in timeline.tracks:
        status = "ARMED" if track.armed else "not armed"
        print(f"  {track.name}: {status}")

    # Start recording
    print("\nStarting recording...")
    timeline.record()
    print(f"Recording: {timeline.is_recording}")
    print(f"Playing: {timeline.is_playing}")


def demo_complete_workflow():
    """Demo 7: Complete DAW workflow"""
    print("\n" + "=" * 60)
    print("Demo 7: Complete DAW Workflow")
    print("=" * 60)

    if not NUMPY_AVAILABLE:
        print("Skipping audio generation - NumPy not available")
        return

    # Create session
    timeline = Timeline(sample_rate=48000, tempo=128.0)
    print(f"Created timeline: {timeline}")

    # Add tracks
    drums = timeline.add_track("Drums", "audio")
    bass = timeline.add_track("Bass", "audio")
    synth = timeline.add_track("Synth", "audio")
    vocals = timeline.add_track("Vocals", "audio")

    # Add clips with trimming
    drums.add_clip(
        Clip("drums.wav").trim(0.0, 32.0),
        start_time=0.0
    )

    bass.add_clip(
        Clip("bass.wav").trim(0.0, 16.0),
        start_time=0.0
    )

    synth_clip = Clip("synth.wav")
    synth_clip.duration = 32.0
    synth.add_clip(synth_clip, start_time=0.0)

    vocals.add_clip(
        Clip("vocals.wav").trim(2.0, 26.0).set_fades(0.5, 1.0),
        start_time=8.0
    )

    # Add markers for song structure
    timeline.add_marker(0.0, "Intro")
    timeline.add_marker(8.0, "Verse 1")
    timeline.add_marker(16.0, "Chorus")
    timeline.add_marker(24.0, "Verse 2")

    # Set loop region for rehearsal
    timeline.set_loop_region(8.0, 16.0)

    # Add automation
    volume_auto = vocals.automate("volume")
    volume_auto.add_point(8.0, 0.0)
    volume_auto.add_point(9.0, 1.0)
    volume_auto.add_point(30.0, 1.0)
    volume_auto.add_point(32.0, 0.0)

    pan_auto = synth.automate("pan")
    pan_auto.add_point(0.0, -1.0)   # Start left
    pan_auto.add_point(16.0, 1.0)   # Move to right
    pan_auto.add_point(32.0, -1.0)  # Back to left

    # Set track parameters
    drums.volume = 0.9
    bass.volume = 0.7
    synth.volume = 0.6
    vocals.volume = 1.0

    # Print session summary
    print(f"\nSession Summary:")
    print(f"  Tempo: {timeline.tempo} BPM")
    print(f"  Sample Rate: {timeline.sample_rate} Hz")
    print(f"  Tracks: {len(timeline.tracks)}")
    print(f"  Total Duration: {timeline.get_duration():.2f}s")
    print(f"  Markers: {len(timeline.markers)}")
    print(f"  Loop Region: {timeline.loop_region}")

    print(f"\nTracks:")
    for track in timeline.tracks:
        print(f"  {track.name}:")
        print(f"    Volume: {track.volume:.2f}")
        print(f"    Clips: {len(track.clips)}")
        print(f"    Automation: {list(track.automation.keys())}")

    print(f"\nMarkers:")
    for marker in timeline.markers:
        print(f"  {marker}")

    # Render timeline to audio
    print(f"\nRendering timeline to audio...")
    mixed_audio = render_timeline_to_audio(timeline, sample_rate=48000)

    if mixed_audio is not None:
        output_path = OUTPUT_DIR / "complete_workflow_mix.wav"
        write_audio_file(mixed_audio, 48000, output_path)
        print(f"  Saved: {output_path}")
        print(f"  Duration: {len(mixed_audio) / 48000:.2f}s")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

        # Also save individual track stems
        print(f"\nRendering individual track stems...")
        for track in timeline.tracks:
            # Create a temporary timeline with just this track
            temp_timeline = Timeline(sample_rate=48000, tempo=128.0)
            temp_track = temp_timeline.add_track(track.name, "audio")
            temp_track.volume = track.volume
            temp_track.clips = track.clips.copy()
            temp_track.automation = track.automation.copy()

            stem_audio = render_timeline_to_audio(temp_timeline, sample_rate=48000)
            if stem_audio is not None:
                stem_path = OUTPUT_DIR / f"stem_{track.name.lower()}.wav"
                write_audio_file(stem_audio, 48000, stem_path)
                print(f"  Saved {track.name} stem: {stem_path.name}")


def demo_track_operations():
    """Demo 8: Advanced track operations"""
    print("\n" + "=" * 60)
    print("Demo 8: Advanced Track Operations")
    print("=" * 60)

    timeline = Timeline()
    track = timeline.add_track("Guitar")

    # Add multiple clips
    for i in range(4):
        clip = Clip(f"take_{i+1}.wav")
        clip.duration = 5.0
        track.add_clip(clip, start_time=i * 10.0)

    print(f"Added {len(track.clips)} clips to track")

    # Query clips at different times
    print("\nActive clips at different times:")
    for time in [2.0, 7.0, 12.0, 22.0, 35.0]:
        clips = track.get_clips_at_time(time)
        clip_names = [str(c.source) for c in clips]
        print(f"  t={time:5.1f}s: {len(clips)} clip(s) - {clip_names}")

    # Remove a clip
    clip_to_remove = track.clips[1]
    track.remove_clip(clip_to_remove)
    print(f"\nAfter removing clip: {len(track.clips)} clips remaining")


def demo_automation_modes():
    """Demo 9: Different automation interpolation modes"""
    print("\n" + "=" * 60)
    print("Demo 9: Automation Interpolation Modes")
    print("=" * 60)

    # Create automation with same points
    points = [(0.0, 0.0), (2.0, 1.0), (4.0, 0.5), (6.0, 1.0)]

    print("Automation points:")
    for time, value in points:
        print(f"  t={time:.1f}s: value={value:.2f}")

    print("\nInterpolated values at t=1.0s, t=3.0s, t=5.0s:")

    for mode in ["linear", "step", "cubic"]:
        lane = AutomationLane("test")
        lane.interpolation = mode
        for time, value in points:
            lane.add_point(time, value)

        values = [lane.get_value(t) for t in [1.0, 3.0, 5.0]]
        print(f"  {mode:6s}: {' / '.join(f'{v:.3f}' for v in values)}")


def demo_time_range():
    """Demo 10: Time range operations"""
    print("\n" + "=" * 60)
    print("Demo 10: Time Range Operations")
    print("=" * 60)

    time_range = TimeRange(10.0, 30.0)
    print(f"Time range: {time_range}")
    print(f"Duration: {time_range.duration}s")

    print("\nContainment tests:")
    test_times = [5.0, 10.0, 20.0, 30.0, 35.0]
    for time in test_times:
        contained = time_range.contains(time)
        status = "inside" if contained else "outside"
        print(f"  t={time:5.1f}s: {status}")


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("CoreMusic DAW Essentials Module - Demonstrations")
    print("=" * 60)

    # Clean output directory
    if NUMPY_AVAILABLE:
        import shutil
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    demos = [
        demo_midi_clip,
        demo_midi_instruments,
        demo_audio_effects,
        demo_basic_timeline,
        demo_clip_management,
        demo_automation,
        demo_markers_and_loops,
        demo_transport_control,
        demo_recording,
        demo_complete_workflow,
        demo_track_operations,
        demo_automation_modes,
        demo_time_range,
    ]

    for demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\nError in {demo_func.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)

    # Show created audio files
    if NUMPY_AVAILABLE and OUTPUT_DIR.exists():
        wav_files = list(OUTPUT_DIR.glob("*.wav"))
        if wav_files:
            print("\n" + "=" * 60)
            print("CREATED AUDIO FILES")
            print("=" * 60)
            print(f"\nAll audio files saved to: {OUTPUT_DIR.absolute()}")
            print("\nGenerated files:")
            for wav_file in sorted(wav_files):
                size_kb = wav_file.stat().st_size / 1024
                print(f"  - {wav_file.name} ({size_kb:.1f} KB)")
            print(f"\nTotal audio files created: {len(wav_files)}")
            print("\nYou can now listen to these files to hear the DAW workflow results!")
        else:
            print("\nNo audio files were generated.")
    elif not NUMPY_AVAILABLE:
        print("\nNote: Audio generation requires NumPy. Install with: pip install numpy")


if __name__ == "__main__":
    main()
