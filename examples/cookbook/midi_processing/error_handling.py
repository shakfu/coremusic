#!/usr/bin/env python3
"""Error Handling."""

from coremusic import capi

client = capi.midi_client_create("Error Handling")
port = capi.midi_output_port_create(client, "Out")
data = bytes([0x90, 60, 100])


# --8<-- [start:example]
from coremusic import capi

# There may be no destinations at all, and midi_get_destination raises rather
# than handing back a bad handle
try:
    dest = capi.midi_get_destination(0)
except ValueError:
    print("No MIDI destinations available")
    dest = capi.midi_destination_create(client, "Fallback Output")

try:
    capi.midi_send_data(port, dest, data)
except RuntimeError as e:
    print(f"Failed to send MIDI: {e}")
# --8<-- [end:example]

capi.midi_client_dispose(client)
