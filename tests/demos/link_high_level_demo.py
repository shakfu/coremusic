#!/usr/bin/env python3
"""Demo: High-Level Ableton Link API (Phase 3)

Demonstrates the Pythonic high-level API for Ableton Link including:
- Context manager support
- Clean module exports
- Property-based API
- Common usage patterns
"""

import time
import coremusic as cm


def demo_context_manager():
    """Demo 1: Context Manager Pattern"""
    print("=" * 60)
    print("Demo 1: Context Manager Pattern")
    print("=" * 60)
    print()

    print("Using LinkSession with context manager...")
    print("Link is automatically enabled on enter, disabled on exit\n")

    # Context manager automatically enables/disables Link
    with cm.link.LinkSession(bpm=120.0) as session:
        print(f"✓ Inside context: {session}")
        print(f"  Enabled: {session.enabled}")
        print(f"  Tempo: {session.capture_app_session_state().tempo:.1f} BPM")
        print(f"  Peers: {session.num_peers}")
        time.sleep(0.5)

    print(f"\n✓ Outside context - Link disabled\n")


def demo_simple_tempo_monitoring():
    """Demo 2: Simple Tempo Monitoring"""
    print("=" * 60)
    print("Demo 2: Simple Tempo Monitoring")
    print("=" * 60)
    print()

    print("Monitoring Link tempo for 3 seconds...")
    print("(Start another Link-enabled app to see peer connection)\n")

    with cm.link.LinkSession(bpm=120.0) as session:
        for i in range(6):
            state = session.capture_app_session_state()
            print(f"  [{i+1}/6] Tempo: {state.tempo:6.1f} BPM | "
                  f"Peers: {session.num_peers}", end='\r')
            time.sleep(0.5)

    print("\n\n✓ Monitoring complete\n")


def demo_beat_tracking():
    """Demo 3: Beat Tracking"""
    print("=" * 60)
    print("Demo 3: Beat Tracking")
    print("=" * 60)
    print()

    print("Tracking beats for 5 seconds...")
    print("(Visual indicator shows downbeats)\n")

    with cm.link.LinkSession(bpm=120.0) as session:
        clock = session.clock

        for i in range(10):
            state = session.capture_app_session_state()
            current_time = clock.micros()

            beat = state.beat_at_time(current_time, quantum=4.0)
            phase = state.phase_at_time(current_time, quantum=4.0)

            # Show downbeat indicator
            indicator = "●" if int(beat) % 4 == 0 else "○"

            print(f"  {indicator} Beat: {beat:7.2f} | Phase: {phase:4.2f}/4", end='\r')
            time.sleep(0.5)

    print("\n\n✓ Beat tracking complete\n")


def demo_tempo_changes():
    """Demo 4: Dynamic Tempo Changes"""
    print("=" * 60)
    print("Demo 4: Dynamic Tempo Changes")
    print("=" * 60)
    print()

    print("Demonstrating tempo changes...")
    print("Changing tempo every 2 seconds\n")

    tempos = [120.0, 140.0, 100.0, 160.0]

    with cm.link.LinkSession(bpm=tempos[0]) as session:
        for i, target_tempo in enumerate(tempos, 1):
            # Set new tempo
            state = session.capture_app_session_state()
            current_time = session.clock.micros()
            state.set_tempo(target_tempo, current_time)
            session.commit_app_session_state(state)

            print(f"  [{i}/{len(tempos)}] Set tempo to {target_tempo:.1f} BPM")

            # Monitor for 2 seconds
            for j in range(4):
                state = session.capture_app_session_state()
                print(f"      Current: {state.tempo:6.1f} BPM", end='\r')
                time.sleep(0.5)
            print()

    print("\n✓ Tempo changes complete\n")


def demo_transport_control():
    """Demo 5: Transport Control"""
    print("=" * 60)
    print("Demo 5: Transport Control")
    print("=" * 60)
    print()

    print("Demonstrating transport start/stop...")
    print("(Enable start/stop sync in another Link app to see sync)\n")

    with cm.link.LinkSession(bpm=120.0) as session:
        # Enable transport sync
        session.start_stop_sync_enabled = True
        print(f"  Transport sync: {'Enabled' if session.start_stop_sync_enabled else 'Disabled'}")

        # Start transport
        print("\n  Starting transport...")
        state = session.capture_app_session_state()
        current_time = session.clock.micros()
        state.set_is_playing(True, current_time)
        session.commit_app_session_state(state)

        time.sleep(0.1)

        # Monitor playing state
        for i in range(4):
            state = session.capture_app_session_state()
            status = "▶ PLAYING" if state.is_playing else "■ STOPPED"
            print(f"  {status}", end='\r')
            time.sleep(0.5)

        # Stop transport
        print("\n\n  Stopping transport...")
        state = session.capture_app_session_state()
        current_time = session.clock.micros()
        state.set_is_playing(False, current_time)
        session.commit_app_session_state(state)

        time.sleep(0.1)

        # Verify stopped
        state = session.capture_app_session_state()
        status = "▶ PLAYING" if state.is_playing else "■ STOPPED"
        print(f"  {status}\n")

    print("✓ Transport control complete\n")


def demo_pythonic_patterns():
    """Demo 6: Pythonic API Patterns"""
    print("=" * 60)
    print("Demo 6: Pythonic API Patterns")
    print("=" * 60)
    print()

    print("Demonstrating Pythonic API features...\n")

    # Property-based API
    print("1. Property-based API:")
    session = cm.link.LinkSession(bpm=120.0)
    print(f"   session.enabled = {session.enabled}")
    print(f"   session.num_peers = {session.num_peers}")
    print(f"   session.start_stop_sync_enabled = {session.start_stop_sync_enabled}")

    # Readable __repr__
    print(f"\n2. Informative __repr__:")
    print(f"   {repr(session)}")

    # Named arguments
    print(f"\n3. Named arguments:")
    session.enabled = True
    state = session.capture_app_session_state()
    current_time = session.clock.micros()
    print(f"   state.set_tempo(bpm=140.0, time_micros={current_time})")
    state.set_tempo(bpm=140.0, time_micros=current_time)
    session.commit_app_session_state(state)

    time.sleep(0.1)

    state = session.capture_app_session_state()
    print(f"   Result: {state.tempo:.1f} BPM")

    # Context manager
    print(f"\n4. Context manager pattern:")
    print(f"   with cm.link.LinkSession(bpm=120.0) as session:")
    print(f"       # Link is automatically enabled")
    print(f"       pass")
    print(f"   # Link is automatically disabled")

    session.enabled = False
    print("\n✓ Pythonic patterns demonstrated\n")


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print(" Ableton Link High-Level API Demonstration")
    print("=" * 60)
    print()
    print("This demo showcases the Pythonic high-level API for Link")
    print("including context managers, properties, and clean patterns.")
    print()
    input("Press Enter to start...")
    print()

    try:
        demo_context_manager()
        input("Press Enter for next demo...")
        print()

        demo_simple_tempo_monitoring()
        input("Press Enter for next demo...")
        print()

        demo_beat_tracking()
        input("Press Enter for next demo...")
        print()

        demo_tempo_changes()
        input("Press Enter for next demo...")
        print()

        demo_transport_control()
        input("Press Enter for next demo...")
        print()

        demo_pythonic_patterns()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")

    print("=" * 60)
    print(" All Demos Complete!")
    print("=" * 60)
    print()
    print("Key Takeaways:")
    print("  • Use 'with cm.link.LinkSession()' for automatic enable/disable")
    print("  • Access Link via 'cm.link' from main coremusic package")
    print("  • Properties and named arguments for Pythonic code")
    print("  • Clean, intuitive API for professional applications")
    print()


if __name__ == "__main__":
    main()
