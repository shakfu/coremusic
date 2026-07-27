#!/usr/bin/env python3
"""Send MIDI Clock."""

# --8<-- [start:example]
import coremusic.capi as capi
from coremusic import link

from coremusic.midi import link as link_midi
import time

# Create MIDI output
client = capi.midi_client_create("Link MIDI Clock")
port = capi.midi_output_port_create(client, "Clock Out")
dest = capi.midi_destination_create(client, "Link Demo Destination")

# Create Link session
with link.LinkSession(bpm=120.0) as session:
    # Create MIDI clock synchronized to Link
    clock = link_midi.LinkMIDIClock(session, port, dest)

    # Start sending MIDI clock
    clock.start()
    print("Sending MIDI clock at 120 BPM")

    # Let it run
    time.sleep(1)

    # Change tempo
    state = session.capture_app_session_state()
    state.set_tempo(140.0, session.clock.micros())
    session.commit_app_session_state(state)
    print("Changed tempo to 140 BPM")

    time.sleep(1)

    # Stop clock
    clock.stop()

# Cleanup: disposing the client also disposes its ports and endpoints
capi.midi_client_dispose(client)
# --8<-- [end:example]
