#!/usr/bin/env python3
"""Timing Precision."""

from coremusic import capi
from coremusic.midi import note_on

client = capi.midi_client_create("Timing")
port = capi.midi_output_port_create(client, "Out")
dest = capi.midi_destination_create(client, "Timing Dest")
input_port = capi.midi_input_port_create(client, "In")
note = note_on("C4", 100)


# --8<-- [start:example]
from coremusic import capi

# Schedule a note 50 ms from now
when = capi.midi_current_host_time() + capi.midi_seconds_to_host_time(0.05)
capi.midi_send_data(port, dest, note, when)

# Convert an incoming packet timestamp back to seconds
for host_time, _payload in capi.midi_input_poll(input_port):
    seconds = capi.midi_host_time_to_seconds(host_time)
# --8<-- [end:example]

capi.midi_client_dispose(client)
