#!/usr/bin/env python3
"""Transpose."""

# --8<-- [start:example]
import time

from coremusic import capi
from coremusic.midi import MIDIMessageSplitter


class Transposer:
    def __init__(self, output_port, dest, semitones):
        self.output_port = output_port
        self.dest = dest
        self.semitones = semitones
        self.splitter = MIDIMessageSplitter()

    def process(self, payload: bytes) -> None:
        for message in self.splitter.push(payload):
            data = bytearray(message)
            message_type = data[0] & 0xF0

            # Transpose note on/off messages
            if message_type in (0x80, 0x90) and len(data) >= 3:
                original_note = data[1]
                transposed_note = max(0, min(127, original_note + self.semitones))
                data[1] = transposed_note

                print(f"Transposed: {original_note} -> {transposed_note}")

            # Forward the (possibly modified) message
            capi.midi_send_data(self.output_port, self.dest, bytes(data))

# Transpose up one octave
client = capi.midi_client_create("Transposer")
output_port = capi.midi_output_port_create(client, "Output")
# Use the first destination, or publish one of our own when nothing else is
# connected, so this runs with or without hardware attached
if capi.midi_get_number_of_destinations() > 0:
    dest = capi.midi_get_destination(0)
else:
    dest = capi.midi_destination_create(client, "Cookbook Output")

transposer = Transposer(output_port, dest, semitones=12)

input_port = capi.midi_input_port_create(client, "Input")
# Connect to the first source, or to one we publish ourselves when nothing
# else is connected
if capi.midi_get_number_of_sources() > 0:
    source = capi.midi_get_source(0)
else:
    source = capi.midi_source_create(client, "Cookbook Input")
capi.midi_port_connect_source(input_port, source)

# Run for a second; a real router would loop until told to stop
deadline = time.monotonic() + 1.0
while time.monotonic() < deadline:
    if capi.midi_input_wait(input_port, 0.1):
        for _host_time, payload in capi.midi_input_poll(input_port):
            transposer.process(payload)
# --8<-- [end:example]
