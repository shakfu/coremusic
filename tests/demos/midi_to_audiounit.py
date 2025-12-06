#!/usr/bin/env python3
"""Demo: Playing MIDI Files through AudioUnit Instruments

Demonstrates how to:
1. Load a MIDI file (or create one programmatically)
2. Send MIDI events to an AudioUnit instrument plugin
3. Render the audio output to a file

This shows the complete workflow for MIDI file → AudioUnit → Audio rendering.
"""

import sys
import time
from pathlib import Path

# Add src to path for demo purposes
BUILD_DIR = Path.cwd() / "build"
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import coremusic as cm
from coremusic.midi import MIDISequence, MIDIEvent
from coremusic.daw import AudioUnitPlugin

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available - some demos will be skipped")

# Output directory
OUTPUT_DIR = BUILD_DIR / "midi_audiounit_output"


def create_simple_midi_file(output_path):
    """Create a simple MIDI file for demonstration."""
    print("\n" + "=" * 70)
    print("Creating Simple MIDI File")
    print("=" * 70)

    # Create MIDI sequence
    seq = MIDISequence(tempo=120.0, time_signature=(4, 4))

    # Add melody track
    melody = seq.add_track("Melody")
    melody.channel = 0
    melody.add_program_change(0.0, 0)  # Acoustic Grand Piano

    # Add C major scale
    notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C, D, E, F, G, A, B, C
    for i, note in enumerate(notes):
        start_time = i * 0.5
        melody.add_note(start_time, note, 100, 0.4)

    # Add a chord at the end
    chord_start = len(notes) * 0.5
    for note in [60, 64, 67]:  # C major chord
        melody.add_note(chord_start, note, 90, 1.0)

    # Save MIDI file
    seq.save(str(output_path))

    print(f"Created MIDI file: {output_path}")
    print(f"  Tempo: {seq.tempo} BPM")
    print(f"  Duration: {seq.duration:.2f} seconds")
    print(f"  Tracks: {len(seq.tracks)}")
    print(f"  Notes in melody: {len([e for e in melody.events if e.is_note_on])}")

    return seq


def demo_midi_file_to_audiounit_simple():
    """Demo 1: Simple MIDI file playback through AudioUnit instrument."""
    print("\n" + "=" * 70)
    print("Demo 1: MIDI File → AudioUnit Instrument (Simple)")
    print("=" * 70)

    # Create a simple MIDI file
    midi_path = OUTPUT_DIR / "test_melody.mid"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seq = create_simple_midi_file(midi_path)

    # Load the MIDI file back
    print("\nLoading MIDI file...")
    loaded_seq = MIDISequence.load(str(midi_path))
    print(f"  Loaded {len(loaded_seq.tracks)} track(s)")

    # Get the first track
    track = loaded_seq.tracks[0]
    print(f"  Track: {track.name}, Channel: {track.channel}")
    print(f"  Events: {len(track.events)}")

    # Create AudioUnit instrument
    print("\nInitializing AudioUnit instrument (DLSMusicDevice)...")
    try:
        plugin = AudioUnitPlugin("dls ", plugin_type="instrument", manufacturer="appl")
        plugin.initialize(sample_rate=48000.0)
        print("  AudioUnit initialized successfully")

        # Play through the MIDI events
        print("\nPlaying MIDI events through AudioUnit...")
        print("  (Note: This demo shows the workflow, actual audio rendering requires")
        print("   AudioUnit render callback integration)")

        # Send all MIDI events to the plugin
        event_count = 0
        for event in track.events:
            if event.is_note_on:
                plugin.send_midi(event.data1, event.data2, note_on=True, channel=track.channel)
                event_count += 1
                print(f"    Note ON:  MIDI {event.data1}, velocity {event.data2}")
            elif event.is_note_off:
                plugin.send_midi(event.data1, event.data2, note_on=False, channel=track.channel)
                print(f"    Note OFF: MIDI {event.data1}")
            elif event.is_program_change:
                print(f"    Program Change: {event.data1}")

        print(f"\n  Sent {event_count} note events to AudioUnit")

        # Cleanup
        plugin.dispose()
        print("\n  AudioUnit disposed")

    except Exception as e:
        print(f"\nNote: AudioUnit not available - {e}")
        print("  (This is expected in some test environments)")


def demo_midi_rendering_with_timing():
    """Demo 2: Render MIDI file with proper timing using synthesized audio."""
    print("\n" + "=" * 70)
    print("Demo 2: MIDI File Rendering with Timing")
    print("=" * 70)

    if not NUMPY_AVAILABLE:
        print("Skipping - NumPy required for audio rendering")
        return

    # Create a MIDI file
    midi_path = OUTPUT_DIR / "test_melody.mid"
    if not midi_path.exists():
        seq = create_simple_midi_file(midi_path)
    else:
        seq = MIDISequence.load(str(midi_path))

    print("\nRendering MIDI to audio with timing...")

    # Get the track
    track = seq.tracks[0]

    # Calculate total duration
    duration = seq.duration + 0.5  # Add some tail room
    sample_rate = 48000
    num_samples = int(duration * sample_rate)

    # Create output audio buffer
    audio = np.zeros(num_samples, dtype=np.float32)

    # Simple synthesizer: render each note as a sine wave
    active_notes = {}  # Track active notes

    print(f"  Duration: {duration:.2f}s")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Rendering {len(track.events)} events...")

    # Process all events
    for event in track.events:
        if event.is_note_on and event.data2 > 0:  # velocity > 0
            # Note on
            start_sample = int(event.time * sample_rate)
            note_freq = 440.0 * (2.0 ** ((event.data1 - 69) / 12.0))
            velocity = event.data2 / 127.0
            active_notes[event.data1] = {
                'start': start_sample,
                'freq': note_freq,
                'velocity': velocity
            }

        elif event.is_note_off or (event.is_note_on and event.data2 == 0):
            # Note off
            if event.data1 in active_notes:
                note_info = active_notes[event.data1]
                start_sample = note_info['start']
                end_sample = int(event.time * sample_rate)

                if end_sample > start_sample:
                    # Generate note audio
                    note_len = end_sample - start_sample
                    t = np.arange(note_len) / sample_rate

                    # Simple piano-like sound
                    fundamental = np.sin(2 * np.pi * note_info['freq'] * t)
                    harmonic2 = 0.3 * np.sin(2 * np.pi * note_info['freq'] * 2 * t)
                    envelope = np.exp(-t * 3.0)

                    note_audio = (fundamental + harmonic2) * envelope * note_info['velocity'] * 0.5

                    # Add to output buffer
                    audio[start_sample:end_sample] += note_audio

                del active_notes[event.data1]

    # Normalize audio
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    # Save to file
    output_path = OUTPUT_DIR / "midi_rendered.wav"

    # Convert to stereo by duplicating
    audio_stereo = np.repeat(audio, 2)
    audio_int16 = (audio_stereo * 32767).astype(np.int16)

    # Use coremusic to write the file
    from coremusic.capi import (
        fourchar_to_int,
        extended_audio_file_create_with_url,
        extended_audio_file_write,
        extended_audio_file_dispose,
        get_linear_pcm_format_flag_is_signed_integer,
        get_linear_pcm_format_flag_is_packed,
    )

    asbd = {
        'sample_rate': float(sample_rate),
        'format_id': fourchar_to_int('lpcm'),
        'format_flags': (
            get_linear_pcm_format_flag_is_signed_integer() |
            get_linear_pcm_format_flag_is_packed()
        ),
        'bytes_per_packet': 4,
        'frames_per_packet': 1,
        'bytes_per_frame': 4,
        'channels_per_frame': 2,
        'bits_per_channel': 16
    }

    file_type = fourchar_to_int('WAVE')
    ext_audio_file_id = extended_audio_file_create_with_url(
        str(output_path), file_type, asbd
    )

    try:
        num_frames = len(audio_int16) // 2
        extended_audio_file_write(ext_audio_file_id, num_frames, audio_int16.tobytes())
    finally:
        extended_audio_file_dispose(ext_audio_file_id)

    print(f"\n  Rendered audio saved to: {output_path.name}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    print("  You can now play this file to hear the MIDI sequence!")


def demo_midi_with_daw_integration():
    """Demo 3: Using DAW module to play MIDI files through instruments."""
    print("\n" + "=" * 70)
    print("Demo 3: DAW Integration - MIDI File Playback")
    print("=" * 70)

    print("\nThis demonstrates the recommended workflow:")
    print("  1. Load MIDI file using MIDISequence")
    print("  2. Create MIDIClip from loaded data")
    print("  3. Add to DAW Timeline with instrument")
    print("  4. Render timeline to audio")

    from coremusic.daw import Timeline, MIDIClip, Clip

    # Create or load MIDI file
    midi_path = OUTPUT_DIR / "test_melody.mid"
    if not midi_path.exists():
        seq = create_simple_midi_file(midi_path)
    else:
        seq = MIDISequence.load(str(midi_path))

    print(f"\nLoaded MIDI file: {midi_path.name}")

    # Create DAW timeline
    timeline = Timeline(sample_rate=48000, tempo=seq.tempo)
    print(f"Created timeline at {timeline.tempo} BPM")

    # Add MIDI track with instrument
    piano_track = timeline.add_track("Piano", "midi")
    print(f"Added MIDI track: {piano_track.name}")

    # Set instrument (DLSMusicDevice - Apple's General MIDI synth)
    try:
        instrument = piano_track.set_instrument("dls ")
        print(f"  Set instrument: {instrument.name}")
    except Exception as e:
        print(f"  Note: Could not load instrument - {e}")

    # Convert MIDI sequence to MIDIClip
    midi_clip = MIDIClip()
    track_data = seq.tracks[0]

    for event in track_data.events:
        if event.is_note_on and event.data2 > 0:
            # Find corresponding note off
            note_off_time = None
            for later_event in track_data.events:
                if (later_event.time > event.time and
                    (later_event.is_note_off or (later_event.is_note_on and later_event.data2 == 0)) and
                    later_event.data1 == event.data1):
                    note_off_time = later_event.time
                    break

            if note_off_time:
                duration = note_off_time - event.time
                midi_clip.add_note(
                    note=event.data1,
                    velocity=event.data2,
                    start_time=event.time,
                    duration=duration,
                    channel=track_data.channel
                )

    print(f"  Converted to MIDIClip with {len(midi_clip.notes)} notes")

    # Add clip to track
    clip = Clip(midi_clip, clip_type="midi")
    clip.duration = seq.duration
    piano_track.add_clip(clip, start_time=0.0)

    print(f"  Added clip to timeline (duration: {seq.duration:.2f}s)")
    print("\nTimeline ready for playback/rendering!")
    print(f"  Total duration: {timeline.get_duration():.2f}s")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("MIDI File → AudioUnit Instrument Demonstrations")
    print("=" * 70)
    print("\nShowing different approaches to playing MIDI files through")
    print("AudioUnit instruments in CoreMusic.")

    demos = [
        demo_midi_file_to_audiounit_simple,
        demo_midi_rendering_with_timing,
        demo_midi_with_daw_integration,
    ]

    for demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\nError in {demo_func.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All Demos Completed!")
    print("=" * 70)

    # Show created files
    if OUTPUT_DIR.exists():
        files = list(OUTPUT_DIR.glob("*"))
        if files:
            print(f"\nCreated files in {OUTPUT_DIR}:")
            for f in sorted(files):
                if f.is_file():
                    size = f.stat().st_size
                    if f.suffix == '.mid':
                        print(f"  - {f.name} ({size} bytes) - MIDI file")
                    elif f.suffix == '.wav':
                        print(f"  - {f.name} ({size / 1024:.1f} KB) - Audio file")
                    else:
                        print(f"  - {f.name}")

            wav_files = list(OUTPUT_DIR.glob("*.wav"))
            if wav_files:
                print(f"\nYou can play the audio files to hear the rendered MIDI!")


if __name__ == "__main__":
    main()
