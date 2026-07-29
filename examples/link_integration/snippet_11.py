#!/usr/bin/env python3
"""MIDI Clock Synchronization."""

# --8<-- [start:example]
import time

from coremusic import capi, link
from coremusic.midi import link as link_midi

# Setup MIDI
client = capi.midi_client_create("MIDI Clock")
port = capi.midi_output_port_create(client, "Clock Out")
destination = capi.midi_destination_create(client, "Link Demo Destination")

# Create Link session and MIDI clock
with link.LinkSession(bpm=120.0) as session:
    # Create MIDI clock synchronized to Link
    clock = link_midi.LinkMIDIClock(session, port, destination)

    # Start sending MIDI clock
    clock.start()
    print("Sending MIDI Clock at 120 BPM")
    print("(24 clock messages per quarter note)")

    # Run for 10 seconds
    for _i in range(20):
        state = session.capture_app_session_state()
        print(f"Tempo: {state.tempo:6.1f} BPM | Peers: {session.num_peers}", end="\r")
        time.sleep(0.5)

    # Stop clock
    clock.stop()
    print("\nMIDI Clock stopped")

# Cleanup
# Cleanup: disposing the client also disposes its ports and endpoints
capi.midi_client_dispose(client)
# --8<-- [end:example]
