#!/usr/bin/env python3
"""Demo for DAW (Digital Audio Workstation) essentials module.

This script demonstrates the coremusic.daw module capabilities:
- Multi-track timeline creation
- Clip management
- Automation
- Markers and loop regions
- Transport control
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
BUILD_DIR = Path.cwd() / "build"
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import coremusic as cm
from coremusic.daw import (
    Timeline,
    Track,
    Clip,
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
    """Generate a simple drum pattern (kick + snare)"""
    if not NUMPY_AVAILABLE:
        return None

    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)

    # Calculate beat interval in samples
    beat_interval = int((60.0 / tempo) * sample_rate)

    # Kick drum on beats 1 and 3
    for beat in [0, 2]:
        pos = beat * beat_interval
        if pos < num_samples:
            # Simple kick: low frequency sine burst
            t = np.arange(min(1000, num_samples - pos)) / sample_rate
            kick = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 20)
            audio[pos:pos + len(kick)] += kick * 0.8

    # Snare on beats 2 and 4
    for beat in [1, 3]:
        pos = beat * beat_interval
        if pos < num_samples:
            # Simple snare: noise burst
            snare_len = min(500, num_samples - pos)
            t = np.arange(snare_len) / sample_rate
            snare = np.random.randn(snare_len) * np.exp(-t * 40)
            audio[pos:pos + snare_len] += snare * 0.4

    return audio


def generate_bass_line(duration, sample_rate=48000, tempo=128.0):
    """Generate a simple bass line"""
    if not NUMPY_AVAILABLE:
        return None

    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate

    # Simple repeating bass pattern
    beat_duration = 60.0 / tempo
    pattern_duration = beat_duration * 4

    # Bass notes (frequencies)
    notes = [55, 55, 73.42, 55]  # A1, A1, D2, A1
    audio = np.zeros(num_samples, dtype=np.float32)

    for i, freq in enumerate(notes):
        note_start = i * beat_duration
        note_end = note_start + beat_duration * 0.8

        start_sample = int(note_start * sample_rate)
        end_sample = int(note_end * sample_rate)

        if start_sample >= num_samples:
            break

        end_sample = min(end_sample, num_samples)
        note_t = np.arange(end_sample - start_sample) / sample_rate

        # Simple bass tone with envelope
        envelope = np.exp(-note_t * 2)
        bass_note = np.sin(2 * np.pi * freq * note_t) * envelope * 0.6
        audio[start_sample:end_sample] += bass_note

    return audio


def generate_synth_pad(duration, sample_rate=48000):
    """Generate a simple synth pad"""
    if not NUMPY_AVAILABLE:
        return None

    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate

    # Layered sine waves for pad sound
    freqs = [220, 220 * 1.5, 220 * 2, 220 * 3]  # A3 chord
    audio = np.zeros(num_samples, dtype=np.float32)

    for i, freq in enumerate(freqs):
        audio += np.sin(2 * np.pi * freq * t) * (0.15 / len(freqs))

    # Slow fade in/out
    fade_samples = int(2.0 * sample_rate)
    fade_in = np.linspace(0, 1, min(fade_samples, num_samples))
    fade_out = np.linspace(1, 0, min(fade_samples, num_samples))

    audio[:len(fade_in)] *= fade_in
    audio[-len(fade_out):] *= fade_out

    return audio


def generate_vocal_melody(duration, sample_rate=48000, tempo=128.0):
    """Generate a simple melodic line (simulating vocals)"""
    if not NUMPY_AVAILABLE:
        return None

    num_samples = int(duration * sample_rate)
    beat_duration = 60.0 / tempo

    # Melody notes (frequencies in Hz)
    melody = [
        (440, beat_duration * 2),      # A4
        (493.88, beat_duration),       # B4
        (523.25, beat_duration * 2),   # C5
        (493.88, beat_duration),       # B4
        (440, beat_duration * 2),      # A4
    ]

    audio = np.zeros(num_samples, dtype=np.float32)
    current_pos = 0

    for freq, note_duration in melody:
        start_sample = int(current_pos * sample_rate)
        end_sample = int((current_pos + note_duration) * sample_rate)

        if start_sample >= num_samples:
            break

        end_sample = min(end_sample, num_samples)
        note_t = np.arange(end_sample - start_sample) / sample_rate

        # Sine with vibrato and envelope
        vibrato = 1 + 0.02 * np.sin(2 * np.pi * 5 * note_t)
        envelope = np.exp(-note_t * 1.5)
        note = np.sin(2 * np.pi * freq * vibrato * note_t) * envelope * 0.5

        audio[start_sample:end_sample] += note
        current_pos += note_duration

    return audio


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
