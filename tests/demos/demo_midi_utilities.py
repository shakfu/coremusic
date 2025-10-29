#!/usr/bin/env python3
"""Demo: MIDI Utilities

Demonstrates the MIDI utilities capabilities of CoreMusic including:
- Creating MIDI sequences programmatically
- Adding notes, control changes, and program changes
- Multi-track compositions
- Saving to Standard MIDI File format
- Loading MIDI files
- MIDI routing and transformations
- Transform functions (transpose, velocity scale, etc.)
"""

import sys
from pathlib import Path

# Add src to path for demo purposes
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from coremusic.midi import (
    MIDISequence,
    MIDIFileFormat,
    MIDIRouter,
    transpose_transform,
    velocity_scale_transform,
    channel_remap_transform,
    quantize_transform,
    MIDIEvent,
    MIDIStatus,
)


def demo_basic_sequence():
    """Demo 1: Basic MIDI sequence creation."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic MIDI Sequence Creation")
    print("=" * 70)

    print("\nCreating simple melody...")
    seq = MIDISequence(tempo=120.0)

    track = seq.add_track("Melody")
    track.channel = 0

    # C major scale
    notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C
    print("  Notes: C D E F G A B C")

    for i, note in enumerate(notes):
        track.add_note(i * 0.5, note, 100, 0.4)

    print(f"\nResults:")
    print(f"  Track: {track.name}")
    print(f"  Events: {len(track.events)}")
    print(f"  Duration: {track.duration:.2f}s")
    print(f"  Sequence duration: {seq.duration:.2f}s")


def demo_multi_track():
    """Demo 2: Multi-track composition."""
    print("\n" + "=" * 70)
    print("DEMO 2: Multi-Track Composition")
    print("=" * 70)

    print("\nCreating multi-track composition...")
    seq = MIDISequence(tempo=120.0, time_signature=(4, 4))

    # Melody track
    print("  Track 1: Melody (Piano)")
    melody = seq.add_track("Melody")
    melody.channel = 0
    melody.add_program_change(0.0, 0)  # Acoustic Grand Piano

    # Simple melody pattern
    melody_notes = [60, 64, 67, 64, 60, 64, 67, 72]
    for i, note in enumerate(melody_notes):
        melody.add_note(i * 0.5, note, 100, 0.4)

    # Bass track
    print("  Track 2: Bass (Electric Bass)")
    bass = seq.add_track("Bass")
    bass.channel = 1
    bass.add_program_change(0.0, 33)  # Electric Bass (finger)

    # Bass line
    bass_notes = [48, 48, 43, 43]
    for i, note in enumerate(bass_notes):
        bass.add_note(i * 1.0, note, 90, 0.9)

    # Drums track
    print("  Track 3: Drums")
    drums = seq.add_track("Drums")
    drums.channel = 9  # MIDI drum channel

    # Kick and snare pattern
    for beat in range(8):
        if beat % 2 == 0:
            drums.add_note(beat * 0.5, 36, 110, 0.1)  # Kick
        else:
            drums.add_note(beat * 0.5, 38, 90, 0.1)  # Snare
        # Hi-hat on every 8th note
        drums.add_note(beat * 0.5, 42, 70, 0.05)  # Closed Hi-Hat

    print(f"\nResults:")
    print(f"  Total tracks: {len(seq.tracks)}")
    print(f"  Melody events: {len(melody.events)}")
    print(f"  Bass events: {len(bass.events)}")
    print(f"  Drums events: {len(drums.events)}")
    print(f"  Total duration: {seq.duration:.2f}s")
    print(f"  Tempo: {seq.tempo} BPM")


def demo_control_changes():
    """Demo 3: Control changes and automation."""
    print("\n" + "=" * 70)
    print("DEMO 3: Control Changes and Automation")
    print("=" * 70)

    print("\nAdding control changes for automation...")
    seq = MIDISequence(tempo=120.0)

    track = seq.add_track("Synth")
    track.channel = 0

    # Add notes
    print("  Adding notes...")
    for i in range(4):
        track.add_note(i * 1.0, 60 + i * 2, 100, 0.8)

    # Add volume automation (CC 7)
    print("  Adding volume automation...")
    for i in range(5):
        volume = int(50 + i * 15)  # 50 -> 110
        track.add_control_change(i * 1.0, 7, volume)

    # Add filter cutoff automation (CC 74)
    print("  Adding filter cutoff automation...")
    for i in range(5):
        cutoff = int(20 + i * 20)  # 20 -> 100
        track.add_control_change(i * 1.0, 74, cutoff)

    # Add pitch bend
    print("  Adding pitch bend...")
    track.add_pitch_bend(0.0, 8192)  # Center
    track.add_pitch_bend(2.0, 10000)  # Bend up
    track.add_pitch_bend(4.0, 8192)  # Back to center

    print(f"\nResults:")
    print(f"  Total events: {len(track.events)}")
    print(f"  Note events: {sum(1 for e in track.events if e.is_note_on or e.is_note_off)}")
    print(f"  CC events: {sum(1 for e in track.events if e.is_control_change)}")
    print(f"  Pitch bend events: {sum(1 for e in track.events if e.status == MIDIStatus.PITCH_BEND)}")


def demo_save_and_load(tmp_dir="/tmp/coremusic_midi_demo"):
    """Demo 4: Save and load MIDI files."""
    print("\n" + "=" * 70)
    print("DEMO 4: Save and Load MIDI Files")
    print("=" * 70)

    import os

    os.makedirs(tmp_dir, exist_ok=True)

    # Create and save sequence
    print(f"\nCreating sequence and saving to {tmp_dir}/...")
    seq = MIDISequence(tempo=140.0)

    track = seq.add_track("Test")
    track.add_note(0.0, 60, 100, 0.5)
    track.add_note(0.5, 64, 100, 0.5)
    track.add_note(1.0, 67, 100, 0.5)

    output_path = f"{tmp_dir}/demo.mid"
    seq.save(output_path, format=MIDIFileFormat.MULTI_TRACK)
    print(f"  Saved: {output_path}")

    # Get file size
    file_size = os.path.getsize(output_path)
    print(f"  File size: {file_size} bytes")

    # Load sequence
    print("\nLoading sequence from file...")
    loaded_seq = MIDISequence.load(output_path)

    print(f"\nLoaded sequence:")
    print(f"  Tempo: {loaded_seq.tempo:.1f} BPM")
    print(f"  Tracks: {len(loaded_seq.tracks)}")
    print(f"  Duration: {loaded_seq.duration:.2f}s")
    print(f"  PPQ: {loaded_seq.ppq}")


def demo_router_basic():
    """Demo 5: Basic MIDI routing."""
    print("\n" + "=" * 70)
    print("DEMO 5: Basic MIDI Routing")
    print("=" * 70)

    print("\nCreating MIDI router...")
    router = MIDIRouter()

    # Add routes
    print("  Adding route: keyboard -> synth")
    router.add_route("keyboard", "synth")

    print("  Adding route: keyboard -> effects")
    router.add_route("keyboard", "effects")

    # Process event
    print("\nProcessing MIDI event...")
    event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)
    print(f"  Input: Note On, channel={event.channel}, note={event.data1}, vel={event.data2}")

    results = router.process_event("keyboard", event)

    print(f"\nRouting results:")
    print(f"  Destinations: {len(results)}")
    for dest, routed_event in results:
        print(f"    -> {dest}: Note {routed_event.data1}, vel {routed_event.data2}")


def demo_router_transforms():
    """Demo 6: MIDI routing with transforms."""
    print("\n" + "=" * 70)
    print("DEMO 6: MIDI Routing with Transforms")
    print("=" * 70)

    print("\nCreating router with transforms...")
    router = MIDIRouter()

    # Register transforms
    print("  Registering transforms...")
    router.add_transform("transpose_up", transpose_transform(12))
    router.add_transform("transpose_down", transpose_transform(-12))
    router.add_transform("softer", velocity_scale_transform(0.7))
    router.add_transform("harder", velocity_scale_transform(1.3))

    # Add routes with transforms
    print("  Route 1: keyboard -> high_synth (transpose up octave)")
    router.add_route("keyboard", "high_synth", transform="transpose_up")

    print("  Route 2: keyboard -> low_synth (transpose down octave, softer)")
    router.add_route("keyboard", "low_synth", transform="transpose_down")

    print("  Route 3: keyboard -> pad (softer)")
    router.add_route("keyboard", "pad", transform="softer")

    # Process event
    print("\nProcessing MIDI event...")
    event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)
    print(f"  Input: Note {event.data1} (C4), velocity {event.data2}")

    results = router.process_event("keyboard", event)

    print(f"\nRouting results:")
    for dest, routed_event in results:
        print(f"  -> {dest}: Note {routed_event.data1}, velocity {routed_event.data2}")


def demo_router_channel_mapping():
    """Demo 7: MIDI routing with channel mapping."""
    print("\n" + "=" * 70)
    print("DEMO 7: MIDI Routing with Channel Mapping")
    print("=" * 70)

    print("\nCreating router with channel mapping...")
    router = MIDIRouter()

    # Add route with channel mapping
    channel_map = {0: 5, 1: 6, 2: 7}
    print(f"  Channel mapping: {channel_map}")
    router.add_route("keyboard", "multi_synth", channel_map=channel_map)

    # Process events on different channels
    print("\nProcessing events on different channels...")
    for ch in [0, 1, 2]:
        event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, ch, 60, 100)
        results = router.process_event("keyboard", event)

        if results:
            dest, routed_event = results[0]
            print(f"  Channel {ch} -> {dest} channel {routed_event.channel}")


def demo_router_filtering():
    """Demo 8: MIDI routing with filtering."""
    print("\n" + "=" * 70)
    print("DEMO 8: MIDI Routing with Filtering")
    print("=" * 70)

    print("\nCreating router with filter...")
    router = MIDIRouter()

    # Only pass note events
    def note_filter(e):
        return e.is_note_on or e.is_note_off

    print("  Filter: Only pass note on/off events")
    router.add_route("keyboard", "synth", filter_func=note_filter)

    # Test with different event types
    print("\nTesting different event types...")

    test_events = [
        ("Note On", MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)),
        ("Note Off", MIDIEvent(0.0, MIDIStatus.NOTE_OFF, 0, 60, 0)),
        ("Control Change", MIDIEvent(0.0, MIDIStatus.CONTROL_CHANGE, 0, 7, 100)),
        ("Program Change", MIDIEvent(0.0, MIDIStatus.PROGRAM_CHANGE, 0, 10, 0)),
    ]

    for name, event in test_events:
        results = router.process_event("keyboard", event)
        status = "✓ PASSED" if results else "✗ FILTERED"
        print(f"  {name}: {status}")


def demo_quantization():
    """Demo 9: Time quantization."""
    print("\n" + "=" * 70)
    print("DEMO 9: Time Quantization")
    print("=" * 70)

    print("\nDemonstrating time quantization...")

    # 16th note grid at 120 BPM
    sixteenth_note = 60.0 / 120.0 / 4  # 0.125 seconds
    print(f"  Grid: 16th notes ({sixteenth_note:.3f}s)")

    transform = quantize_transform(sixteenth_note)

    # Test with slightly off-time events
    test_times = [0.1, 0.3, 0.51, 0.73, 1.02]
    print("\nQuantizing events:")

    for time in test_times:
        event = MIDIEvent(time, MIDIStatus.NOTE_ON, 0, 60, 100)
        quantized = transform(event)
        print(f"  {time:.2f}s -> {quantized.time:.2f}s")


def demo_complex_composition(tmp_dir="/tmp/coremusic_midi_demo"):
    """Demo 10: Complex multi-track composition."""
    print("\n" + "=" * 70)
    print("DEMO 10: Complex Multi-Track Composition")
    print("=" * 70)

    import os

    os.makedirs(tmp_dir, exist_ok=True)

    print("\nCreating complex composition...")
    seq = MIDISequence(tempo=128.0, time_signature=(4, 4))

    # Lead melody
    print("  Track 1: Lead Synth")
    lead = seq.add_track("Lead Synth")
    lead.channel = 0
    lead.add_program_change(0.0, 81)  # Square Lead

    lead_pattern = [60, 63, 65, 67, 65, 63, 60, 58]
    for bar in range(2):
        for i, note in enumerate(lead_pattern):
            time = bar * 4.0 + i * 0.5
            lead.add_note(time, note, 100, 0.4)

    # Pad chords
    print("  Track 2: Pad Chords")
    pad = seq.add_track("Pad")
    pad.channel = 1
    pad.add_program_change(0.0, 88)  # Pad

    chord_progressions = [
        [48, 52, 55],  # C minor
        [53, 57, 60],  # F minor
        [55, 58, 62],  # G minor
        [48, 52, 55],  # C minor
    ]

    for i, chord in enumerate(chord_progressions):
        for note in chord:
            pad.add_note(i * 2.0, note, 70, 1.9)

    # Arpeggio
    print("  Track 3: Arpeggio")
    arp = seq.add_track("Arpeggio")
    arp.channel = 2
    arp.add_program_change(0.0, 11)  # Vibraphone

    arp_pattern = [60, 64, 67, 72, 67, 64]
    for bar in range(4):
        for i, note in enumerate(arp_pattern):
            time = bar * 2.0 + i * 0.25
            arp.add_note(time, note, 85, 0.2)

    # Drums
    print("  Track 4: Drums")
    drums = seq.add_track("Drums")
    drums.channel = 9

    for beat in range(32):
        time = beat * 0.5
        if beat % 4 == 0:
            drums.add_note(time, 36, 110, 0.1)  # Kick
        if beat % 4 == 2:
            drums.add_note(time, 38, 95, 0.1)  # Snare
        drums.add_note(time, 42, 65, 0.05)  # Hi-hat

    # Save composition
    output_path = f"{tmp_dir}/composition.mid"
    seq.save(output_path)

    print(f"\nComposition saved:")
    print(f"  File: {output_path}")
    print(f"  Tracks: {len(seq.tracks)}")
    print(f"  Total events: {sum(len(t.events) for t in seq.tracks)}")
    print(f"  Duration: {seq.duration:.2f}s")
    print(f"  File size: {os.path.getsize(output_path)} bytes")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("COREMUSIC MIDI UTILITIES DEMO")
    print("=" * 70)
    print("\nThis demo showcases MIDI utilities capabilities:")
    print("- MIDI sequence creation")
    print("- Multi-track compositions")
    print("- Control changes and automation")
    print("- MIDI file I/O (Standard MIDI File format)")
    print("- MIDI routing and transformations")
    print("- Event filtering and quantization")

    try:
        demo_basic_sequence()
        demo_multi_track()
        demo_control_changes()
        demo_save_and_load()
        demo_router_basic()
        demo_router_transforms()
        demo_router_channel_mapping()
        demo_router_filtering()
        demo_quantization()
        demo_complex_composition()

        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("- MIDISequence provides easy MIDI composition")
        print("- MIDITrack manages events with automatic sorting")
        print("- Standard MIDI File I/O for compatibility")
        print("- MIDIRouter enables flexible MIDI routing")
        print("- Transform functions for common operations")
        print("- Filter functions for event selection")
        print("- Time quantization for rhythmic alignment")
        print("- Full multi-track composition support")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
