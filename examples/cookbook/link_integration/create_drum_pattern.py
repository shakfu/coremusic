#!/usr/bin/env python3
"""Complete Example: Drum Machine."""

# --8<-- [start:example]
import time

from coremusic import capi, link
from coremusic.midi import link as link_midi


def create_drum_pattern():
    """Create a simple drum pattern"""
    pattern = []

    # 4 bars of 4/4 time
    for bar in range(4):
        bar_start = bar * 4.0

        # Kick on beats 1 and 3
        pattern.append((bar_start + 0.0, 36, 100))  # Beat 1
        pattern.append((bar_start + 2.0, 36, 100))  # Beat 3

        # Snare on beats 2 and 4
        pattern.append((bar_start + 1.0, 38, 100))  # Beat 2
        pattern.append((bar_start + 3.0, 38, 100))  # Beat 4

        # Hi-hat every half beat
        for eighth in range(8):
            pattern.append((bar_start + eighth * 0.5, 42, 80))

    return pattern

# Setup MIDI
client = capi.midi_client_create("Drum Machine")
port = capi.midi_output_port_create(client, "Drums")
dest = capi.midi_destination_create(client, "Link Demo Destination")

# Create Link session
with link.LinkSession(bpm=120.0) as session:
    # Create sequencer
    sequencer = link_midi.LinkMIDISequencer(session, port, dest)

    # Load pattern
    pattern = create_drum_pattern()
    for beat, note, velocity in pattern:
        sequencer.schedule_note(
            beat=beat,
            channel=9,  # MIDI channel 10 (index 9) for drums
            note=note,
            velocity=velocity,
            duration=0.1
        )

    print(f"Loaded {len(pattern)} drum hits")
    print(f"Link tempo: {session.capture_app_session_state().tempo:.1f} BPM")
    print(f"Connected peers: {session.num_peers}")

    # Start playback
    sequencer.start()
    print("Drum machine started!")

    # Run for 16 bars
    time.sleep(16 * 4 * 60.0 / 120.0)  # 16 bars at 120 BPM

    # Stop
    sequencer.stop()
    print("Drum machine stopped")

# Cleanup
# Cleanup: disposing the client also disposes its ports and endpoints
capi.midi_client_dispose(client)
# --8<-- [end:example]
