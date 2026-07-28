#!/usr/bin/env python3
"""Send Messages."""

# --8<-- [start:example]
import time

from coremusic import capi
from coremusic.midi import note_off, note_on

# Create MIDI client and output port
client = capi.midi_client_create("MIDI Output")
output_port = capi.midi_output_port_create(client, "Output")

# Use the first destination, or publish one of our own when nothing else is
# connected, so this runs with or without hardware attached
if capi.midi_get_number_of_destinations() > 0:
    dest = capi.midi_get_destination(0)
else:
    dest = capi.midi_destination_create(client, "Cookbook Output")

# Send Note On (middle C, velocity 100, channel 0)
capi.midi_send_data(output_port, dest, note_on("C4", 100))
print("Sent Note On")

time.sleep(1.0)

# Send Note Off
capi.midi_send_data(output_port, dest, note_off("C4"))
print("Sent Note Off")

# Cleanup
capi.midi_port_dispose(output_port)
capi.midi_client_dispose(client)
# --8<-- [end:example]
