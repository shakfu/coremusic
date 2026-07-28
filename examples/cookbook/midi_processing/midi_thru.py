#!/usr/bin/env python3
"""Midi Thru."""

# --8<-- [start:example]
import time

from coremusic import capi

# Create client with input and output ports
client = capi.midi_client_create("MIDI Thru")

# Output port
output_port = capi.midi_output_port_create(client, "Output")
# Use the first destination, or publish one of our own when nothing else is
# connected, so this runs with or without hardware attached
if capi.midi_get_number_of_destinations() > 0:
    dest = capi.midi_get_destination(0)
else:
    dest = capi.midi_destination_create(client, "Cookbook Output")

# Input port
input_port = capi.midi_input_port_create(client, "Input")
# Connect to the first source, or to one we publish ourselves when nothing
# else is connected
if capi.midi_get_number_of_sources() > 0:
    source = capi.midi_get_source(0)
else:
    source = capi.midi_source_create(client, "Cookbook Input")
capi.midi_port_connect_source(input_port, source)

print("MIDI thru active... (1s)")
deadline = time.monotonic() + 1.0
try:
    while time.monotonic() < deadline:
        if not capi.midi_input_wait(input_port, 0.1):
            continue

        # Packets can be forwarded verbatim; no need to split them first.
        for _host_time, payload in capi.midi_input_poll(input_port):
            capi.midi_send_data(output_port, dest, payload)
except KeyboardInterrupt:
    print("\nStopped")

# Cleanup
capi.midi_port_disconnect_source(input_port, source)
capi.midi_port_dispose(input_port)
capi.midi_port_dispose(output_port)
capi.midi_client_dispose(client)
# --8<-- [end:example]
