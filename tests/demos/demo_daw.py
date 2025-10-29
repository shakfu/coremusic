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


if __name__ == "__main__":
    main()
