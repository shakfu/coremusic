#!/usr/bin/env python3
"""Control Changes."""

# --8<-- [start:example]
import time

from coremusic import capi
from coremusic.constants import MIDIControlChange
from coremusic.midi import control_change, note_off, note_on

client = capi.midi_client_create("CC Controller")
output_port = capi.midi_output_port_create(client, "Output")
# Use the first destination, or publish one of our own when nothing else is
# connected, so this runs with or without hardware attached
if capi.midi_get_number_of_destinations() > 0:
    dest = capi.midi_get_destination(0)
else:
    dest = capi.midi_destination_create(client, "Cookbook Output")

# Start a note
capi.midi_send_data(output_port, dest, note_on("C4", 100))

# Fade volume (CC 7) from 127 to 0
for volume in range(127, -1, -5):
    cc = control_change(MIDIControlChange.VOLUME, volume)
    capi.midi_send_data(output_port, dest, cc)
    time.sleep(0.05)

# Stop note
capi.midi_send_data(output_port, dest, note_off("C4"))

# Cleanup
capi.midi_port_dispose(output_port)
capi.midi_client_dispose(client)
# --8<-- [end:example]
