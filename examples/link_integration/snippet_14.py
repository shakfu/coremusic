#!/usr/bin/env python3
"""Looping MIDI Patterns."""

# --8<-- [start:example]
import time

from coremusic import capi, link
from coremusic.midi import link as link_midi

client = capi.midi_client_create("Loop Sequencer")
port = capi.midi_output_port_create(client, "Loop Out")
destination = capi.midi_destination_create(client, "Link Demo Destination")

with link.LinkSession(bpm=120.0) as session:
    seq = link_midi.LinkMIDISequencer(session, port, destination)

    # Create a 4-beat pattern
    pattern = [
        (0.0, 60, 100),  # Beat 0: C4
        (0.5, 62, 80),  # Beat 0.5: D4
        (1.0, 64, 100),  # Beat 1: E4
        (2.0, 67, 100),  # Beat 2: G4
        (3.0, 65, 100),  # Beat 3: F4
        (3.5, 64, 80),  # Beat 3.5: E4
    ]

    # Schedule pattern for multiple bars
    num_bars = 4
    for bar in range(num_bars):
        for beat, note, velocity in pattern:
            absolute_beat = (bar * 4.0) + beat
            seq.schedule_note(
                beat=absolute_beat,
                channel=0,
                note=note,
                velocity=velocity,
                duration=0.4,
            )

    print(f"Scheduled {num_bars} bars of pattern")

    seq.start()
    time.sleep(num_bars * 2)  # 2 seconds per bar at 120 BPM
    seq.stop()

# Cleanup: disposing the client also disposes its ports and endpoints
capi.midi_client_dispose(client)
# --8<-- [end:example]
