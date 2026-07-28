#!/usr/bin/env python3
"""Control Changes."""

# --8<-- [start:example]
import time

from coremusic import capi

client = capi.midi_client_create("CC Controller")
output_port = capi.midi_output_port_create(client, "Output")
# Use the first destination, or publish one of our own when nothing else is
# connected, so this runs with or without hardware attached
if capi.midi_get_number_of_destinations() > 0:
    dest = capi.midi_get_destination(0)
else:
    dest = capi.midi_destination_create(client, "Cookbook Output")

# Start a note
note_on = bytes([0x90, 60, 100])
capi.midi_send_data(output_port, dest, note_on)

# Fade volume (CC 7) from 127 to 0
for volume in range(127, -1, -5):
    cc = bytes([0xB0, 7, volume])  # Channel 1, CC 7 (Volume), value
    capi.midi_send_data(output_port, dest, cc)
    time.sleep(0.05)

# Stop note
note_off = bytes([0x80, 60, 0])
capi.midi_send_data(output_port, dest, note_off)

# Cleanup
capi.midi_port_dispose(output_port)
capi.midi_client_dispose(client)
# --8<-- [end:example]
