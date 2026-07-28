#!/usr/bin/env python3
"""Receive Poll."""

# --8<-- [start:example]
import time

from coremusic import capi
from coremusic.midi import MIDIMessageSplitter

client = capi.midi_client_create("MIDI Input")
input_port = capi.midi_input_port_create(client, "Input")

# Connect to the first source, or to one we publish ourselves when nothing
# else is connected
if capi.midi_get_number_of_sources() > 0:
    source = capi.midi_get_source(0)
else:
    source = capi.midi_source_create(client, "Cookbook Input")
capi.midi_port_connect_source(input_port, source)

splitter = MIDIMessageSplitter()

print("Listening for MIDI for 1s...")
deadline = time.monotonic() + 1.0
try:
    while time.monotonic() < deadline:
        # Blocks until a packet arrives or the timeout expires.
        if not capi.midi_input_wait(input_port, 0.1):
            continue

        for host_time, payload in capi.midi_input_poll(input_port):
            seconds = capi.midi_host_time_to_seconds(host_time)

            for data in splitter.push(payload):
                status = data[0]
                message_type = status & 0xF0
                channel = status & 0x0F

                if message_type == 0x90 and data[2] > 0:  # Note On
                    print(f"Note On: ch={channel}, note={data[1]}, vel={data[2]}")
                elif message_type == 0x80 or message_type == 0x90:  # Note Off
                    print(f"Note Off: ch={channel}, note={data[1]}")
                elif message_type == 0xB0:  # Control Change
                    print(f"CC: ch={channel}, ctrl={data[1]}, val={data[2]}")
except KeyboardInterrupt:
    print("\nStopped")

# Cleanup
capi.midi_port_disconnect_source(input_port, source)
capi.midi_port_dispose(input_port)
capi.midi_client_dispose(client)
# --8<-- [end:example]
