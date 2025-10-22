#!/usr/bin/env python3
"""Demo: Link + MIDI Integration

Demonstrates synchronizing MIDI with Ableton Link including:
- MIDI Clock messages synchronized to Link tempo
- Beat-accurate MIDI sequencing
- Multi-device MIDI synchronization via Link
- Time conversion utilities

Requirements:
- MIDI output device or virtual MIDI port
- Optional: Another Link-enabled application for tempo sync
"""

import time
import sys
import coremusic as cm
from coremusic import link_midi


def list_midi_destinations():
    """List available MIDI destinations"""
    print("Available MIDI Destinations:")
    print("-" * 40)

    num_destinations = cm.capi.midi_get_number_of_destinations()

    if num_destinations == 0:
        print("  (No MIDI destinations found)")
        return []

    destinations = []
    for i in range(num_destinations):
        try:
            dest_id = cm.capi.midi_get_destination(i)
            name = cm.capi.midi_endpoint_get_name(dest_id)
            destinations.append((i, dest_id, name))
            print(f"  [{i}] {name}")
        except Exception as e:
            print(f"  [{i}] Error: {e}")

    return destinations


def demo_midi_clock():
    """Demo 1: MIDI Clock Synchronization"""
    print("\n" + "=" * 60)
    print("Demo 1: MIDI Clock Synchronization")
    print("=" * 60)
    print()
    print("This demo sends MIDI clock messages synchronized to Link.")
    print("Start a Link-enabled app and change tempo to see sync.\n")

    # Setup MIDI
    destinations = list_midi_destinations()
    if not destinations:
        print("\n✗ No MIDI destinations available")
        print("  Create a virtual MIDI port (e.g., IAC Driver) to try this demo")
        return

    print()
    dest_idx = input(f"Select destination [0-{len(destinations)-1}]: ")
    try:
        dest_idx = int(dest_idx)
        _, dest_id, dest_name = destinations[dest_idx]
    except (ValueError, IndexError):
        print("Invalid selection")
        return

    print(f"\nSelected: {dest_name}")

    # Create MIDI client and port
    try:
        client = cm.capi.midi_client_create("Link MIDI Clock Demo")
        port = cm.capi.midi_output_port_create(client, "Clock Out")
    except Exception as e:
        print(f"\n✗ Error creating MIDI client/port: {e}")
        return

    # Create Link session
    print("\nCreating Link session at 120 BPM...")
    with cm.link.LinkSession(bpm=120.0) as session:
        print(f"✓ Link session: {session}")

        # Create MIDI clock
        print(f"✓ Creating MIDI clock...")
        clock = link_midi.LinkMIDIClock(session, port, dest_id)

        # Start clock
        print("\n▶ Starting MIDI clock...")
        clock.start()
        print("  Sending MIDI Start and Clock messages")
        print("  (24 clock messages per quarter note)\n")

        # Monitor for 10 seconds
        print("Monitoring for 10 seconds...")
        print("(Change tempo in another Link app to see sync)\n")

        for i in range(20):
            state = session.capture_app_session_state()
            beat = state.beat_at_time(session.clock.micros(), 4.0)

            print(f"  Beat: {beat:7.2f} | "
                  f"Tempo: {state.tempo:6.1f} BPM | "
                  f"Peers: {session.num_peers}", end='\r')

            time.sleep(0.5)

        # Stop clock
        print("\n\n■ Stopping MIDI clock...")
        clock.stop()
        print("  Sent MIDI Stop message")

    # Cleanup
    cm.capi.midi_port_dispose(port)
    cm.capi.midi_client_dispose(client)

    print("\n✓ Demo complete\n")


def demo_midi_sequencer():
    """Demo 2: Beat-Accurate MIDI Sequencing"""
    print("\n" + "=" * 60)
    print("Demo 2: Beat-Accurate MIDI Sequencing")
    print("=" * 60)
    print()
    print("This demo plays a simple MIDI sequence synchronized to Link beats.\n")

    # Setup MIDI
    destinations = list_midi_destinations()
    if not destinations:
        print("\n✗ No MIDI destinations available")
        return

    print()
    dest_idx = input(f"Select destination [0-{len(destinations)-1}]: ")
    try:
        dest_idx = int(dest_idx)
        _, dest_id, dest_name = destinations[dest_idx]
    except (ValueError, IndexError):
        print("Invalid selection")
        return

    print(f"\nSelected: {dest_name}")

    # Create MIDI client and port
    try:
        client = cm.capi.midi_client_create("Link MIDI Sequencer Demo")
        port = cm.capi.midi_output_port_create(client, "Seq Out")
    except Exception as e:
        print(f"\n✗ Error creating MIDI client/port: {e}")
        return

    # Create Link session
    print("\nCreating Link session at 120 BPM...")
    with cm.link.LinkSession(bpm=120.0) as session:
        print(f"✓ Link session: {session}")

        # Create sequencer
        print(f"✓ Creating MIDI sequencer...")
        seq = link_midi.LinkMIDISequencer(session, port, dest_id, quantum=4.0)

        # Schedule a simple pattern
        print("\nScheduling MIDI pattern...")
        print("  Beat 0.0: C4  (quarter note)")
        print("  Beat 1.0: E4  (quarter note)")
        print("  Beat 2.0: G4  (quarter note)")
        print("  Beat 3.0: C5  (quarter note)")

        # C major arpeggio
        seq.schedule_note(beat=0.0, channel=0, note=60, velocity=100, duration=0.9)  # C4
        seq.schedule_note(beat=1.0, channel=0, note=64, velocity=100, duration=0.9)  # E4
        seq.schedule_note(beat=2.0, channel=0, note=67, velocity=100, duration=0.9)  # G4
        seq.schedule_note(beat=3.0, channel=0, note=72, velocity=100, duration=0.9)  # C5

        print(f"\n✓ Scheduled {len(seq.events)} MIDI events")

        # Start sequencer
        print("\n▶ Starting sequencer...")
        seq.start()

        # Monitor playback
        print("\nPlaying sequence (one bar = 4 beats)...\n")

        for i in range(20):  # 10 seconds
            state = session.capture_app_session_state()
            current_time = session.clock.micros()
            beat = state.beat_at_time(current_time, 4.0)
            phase = state.phase_at_time(current_time, 4.0)

            # Show current beat indicator
            beat_int = int(beat) % 4
            indicators = ["●" if i == beat_int else "○" for i in range(4)]
            beat_display = " ".join(indicators)

            print(f"  {beat_display}  Beat: {beat:7.2f} | Phase: {phase:4.2f}", end='\r')
            time.sleep(0.5)

        # Stop sequencer
        print("\n\n■ Stopping sequencer...")
        seq.stop()

    # Cleanup
    cm.capi.midi_port_dispose(port)
    cm.capi.midi_client_dispose(client)

    print("\n✓ Demo complete\n")


def demo_time_conversion():
    """Demo 3: Link Beat <-> MIDI Time Conversion"""
    print("\n" + "=" * 60)
    print("Demo 3: Link Beat <-> MIDI Time Conversion")
    print("=" * 60)
    print()
    print("This demo shows conversion between Link beats and host time.\n")

    with cm.link.LinkSession(bpm=120.0) as session:
        print(f"Link session: {session}")
        print(f"Current tempo: {session.capture_app_session_state().tempo:.1f} BPM\n")

        # Show conversions for several beats
        print("Beat Position -> Host Time (ticks):")
        print("-" * 40)

        for beat in [0.0, 1.0, 2.0, 4.0, 8.0]:
            host_time = link_midi.link_beat_to_host_time(session, beat, quantum=4.0)
            print(f"  Beat {beat:4.1f} -> {host_time:15d} ticks")

        print()

        # Show reverse conversion
        print("Host Time (ticks) -> Beat Position:")
        print("-" * 40)

        current_ticks = session.clock.ticks()
        for offset in [0, 500000, 1000000, 2000000]:  # Various tick offsets
            ticks = current_ticks + offset
            beat = link_midi.host_time_to_link_beat(session, ticks, quantum=4.0)
            print(f"  {ticks:15d} ticks -> Beat {beat:7.2f}")

        print()

        # Show round-trip accuracy
        print("Round-Trip Accuracy Test:")
        print("-" * 40)

        for original_beat in [0.0, 10.0, 100.0]:
            host_time = link_midi.link_beat_to_host_time(session, original_beat, quantum=4.0)
            converted_beat = link_midi.host_time_to_link_beat(session, host_time, quantum=4.0)
            error = abs(converted_beat - original_beat)

            print(f"  Beat {original_beat:6.1f} -> {host_time:15d} -> Beat {converted_beat:6.3f}")
            print(f"    Error: {error:.6f} beats")

    print("\n✓ Demo complete\n")


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print(" Link + MIDI Integration Demonstration")
    print("=" * 60)
    print()
    print("This demo showcases Ableton Link + CoreMIDI integration:")
    print("  1. MIDI Clock Synchronization")
    print("  2. Beat-Accurate MIDI Sequencing")
    print("  3. Time Conversion Utilities")
    print()
    print("Requirements:")
    print("  • MIDI output device or virtual MIDI port")
    print("  • Optional: Link-enabled app for tempo sync")
    print()

    while True:
        print("\nSelect demo:")
        print("  [1] MIDI Clock Synchronization")
        print("  [2] Beat-Accurate MIDI Sequencing")
        print("  [3] Time Conversion Utilities")
        print("  [q] Quit")
        print()

        choice = input("Choice: ").strip().lower()

        if choice == '1':
            demo_midi_clock()
        elif choice == '2':
            demo_midi_sequencer()
        elif choice == '3':
            demo_time_conversion()
        elif choice == 'q':
            break
        else:
            print("Invalid choice")

    print("\n" + "=" * 60)
    print(" All Demos Complete!")
    print("=" * 60)
    print()
    print("Key Features Demonstrated:")
    print("  • MIDI Clock synchronized to Link tempo")
    print("  • Beat-accurate MIDI event scheduling")
    print("  • Link beat <-> host time conversion")
    print("  • Multi-device synchronization capability")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(0)
