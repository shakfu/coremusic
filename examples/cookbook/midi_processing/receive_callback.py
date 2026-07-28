#!/usr/bin/env python3
"""Receive Callback."""

# --8<-- [start:example]
import queue

from coremusic import capi

incoming: queue.SimpleQueue = queue.SimpleQueue()

def midi_callback(data: bytes, host_time: int) -> None:
    # Hand off immediately; do the real work on your own thread.
    incoming.put((host_time, data))

client = capi.midi_client_create("MIDI Input")
input_port = capi.midi_input_port_create(client, "Input", midi_callback)
# Connect to the first source, or to one we publish ourselves when nothing
# else is connected
if capi.midi_get_number_of_sources() > 0:
    source = capi.midi_get_source(0)
else:
    source = capi.midi_source_create(client, "Cookbook Input")
capi.midi_port_connect_source(input_port, source)
# --8<-- [end:example]
