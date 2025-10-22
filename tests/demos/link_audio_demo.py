#!/usr/bin/env python3
"""Demo: Ableton Link Integration with AudioPlayer

Demonstrates using Ableton Link tempo synchronization with AudioPlayer.
Shows how to:
- Create a Link session
- Attach it to AudioPlayer
- Query Link timing information
- Display synchronized beat/tempo information while playing audio
"""

import time
import sys
from coremusic import AudioPlayer, link


def main():
    """Demo Link integration with AudioPlayer"""
    print("=== Ableton Link AudioPlayer Integration Demo ===\n")

    # Check if audio file was provided
    if len(sys.argv) < 2:
        print("Usage: python link_audio_demo.py <audio_file.wav>")
        print("\nExample: python link_audio_demo.py tests/amen.wav")
        return 1

    audio_file = sys.argv[1]

    # Create Link session
    print("1. Creating Link session at 120 BPM...")
    session = link.LinkSession(bpm=120.0)
    session.enabled = True
    print(f"   Link session: {session}")
    print(f"   Connected peers: {session.num_peers}")

    # Create AudioPlayer with Link integration
    print("\n2. Creating AudioPlayer with Link integration...")
    player = AudioPlayer(link_session=session)
    print(f"   AudioPlayer created")
    print(f"   Link session attached: {player.link_session is not None}")

    # Load audio file
    print(f"\n3. Loading audio file: {audio_file}")
    try:
        player.load_file(audio_file)
        print("   ✓ Audio file loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading file: {e}")
        return 1

    # Setup audio output
    print("\n4. Setting up audio output...")
    try:
        player.setup_output()
        print("   ✓ Audio output configured")
    except Exception as e:
        print(f"   ✗ Error setting up output: {e}")
        return 1

    # Display initial Link timing
    print("\n5. Initial Link timing:")
    timing = player.get_link_timing(quantum=4.0)
    if timing:
        print(f"   Tempo: {timing['tempo']:.1f} BPM")
        print(f"   Beat: {timing['beat']:.2f}")
        print(f"   Phase: {timing['phase']:.2f} (within 4-beat bar)")
        print(f"   Transport: {'Playing' if timing['is_playing'] else 'Stopped'}")
    else:
        print("   No Link timing available")

    # Start playback
    print("\n6. Starting playback with Link sync...")
    player.play()
    player.start()
    print("   ✓ Audio playing")

    # Monitor playback and show Link timing
    print("\n7. Monitoring playback (10 seconds)...")
    print("   [Press Ctrl+C to stop early]\n")

    try:
        for i in range(20):  # 10 seconds (0.5s intervals)
            timing = player.get_link_timing(quantum=4.0)
            progress = player.get_progress()

            if timing:
                beat = timing['beat']
                phase = timing['phase']
                tempo = timing['tempo']

                # Show beat with visual indicator
                beat_indicator = "●" if (int(beat) % 4 == 0) else "○"

                print(f"   {beat_indicator} Beat: {beat:7.2f} | "
                      f"Phase: {phase:4.2f} | "
                      f"Tempo: {tempo:6.1f} BPM | "
                      f"Progress: {progress*100:5.1f}%", end='\r')

            time.sleep(0.5)

            if not player.is_playing():
                print("\n   Audio finished playing")
                break

    except KeyboardInterrupt:
        print("\n\n   Stopped by user")

    # Stop playback
    print("\n\n8. Stopping playback...")
    player.stop()
    print("   ✓ Playback stopped")

    # Show final Link stats
    print("\n9. Final Link state:")
    print(f"   Session: {session}")
    print(f"   Connected peers: {session.num_peers}")
    timing = player.get_link_timing()
    if timing:
        print(f"   Final tempo: {timing['tempo']:.1f} BPM")
        print(f"   Final beat: {timing['beat']:.2f}")

    print("\n=== Demo Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
