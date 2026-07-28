#!/usr/bin/env python3
"""MIDI CC Automation Synchronized to Link."""

# --8<-- [start:example]
import time

from coremusic import capi, link
from coremusic.midi import link as link_midi

# Setup MIDI
client = capi.midi_client_create("CC Automation")
port = capi.midi_output_port_create(client, "CC Out")
destination = capi.midi_destination_create(client, "Link Demo Destination")

with link.LinkSession(bpm=120.0) as session:
    seq = link_midi.LinkMIDISequencer(session, port, destination)

    # Schedule filter cutoff sweep over 4 beats
    # CC #74 (Filter Cutoff) from 0 to 127
    for beat in range(4):
        for substep in range(8):
            position = beat + (substep / 8.0)
            value = int((position / 4.0) * 127)
            seq.schedule_cc(
                beat=position,
                channel=0,
                controller=74,  # Filter Cutoff
                value=value
            )

    print(f"Scheduled {len(seq.events)} CC events")

    seq.start()
    time.sleep(0.5)
    seq.stop()

# Cleanup: disposing the client also disposes its ports and endpoints
capi.midi_client_dispose(client)
# --8<-- [end:example]
