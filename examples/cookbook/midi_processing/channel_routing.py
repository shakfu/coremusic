#!/usr/bin/env python3
"""Channel Routing."""

# --8<-- [start:example]
import time

from coremusic import capi
from coremusic.midi import MIDIMessageSplitter


class ChannelRouter:
    def __init__(self, output_port, dest, input_channel, output_channel):
        self.output_port = output_port
        self.dest = dest
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.splitter = MIDIMessageSplitter()

    def process(self, payload: bytes) -> None:
        for message in self.splitter.push(payload):
            status = message[0]

            # Channel voice messages only; system messages have no channel.
            if not 0x80 <= status <= 0xEF:
                continue

            if (status & 0x0F) == self.input_channel:
                data = bytearray(message)
                data[0] = (status & 0xF0) | self.output_channel
                capi.midi_send_data(self.output_port, self.dest, bytes(data))


# Route channel 0 -> channel 1
client = capi.midi_client_create("Channel Router")
output_port = capi.midi_output_port_create(client, "Output")
# Use the first destination, or publish one of our own when nothing else is
# connected, so this runs with or without hardware attached
if capi.midi_get_number_of_destinations() > 0:
    dest = capi.midi_get_destination(0)
else:
    dest = capi.midi_destination_create(client, "Cookbook Output")

router = ChannelRouter(output_port, dest, input_channel=0, output_channel=1)

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
            router.process(payload)
# --8<-- [end:example]
