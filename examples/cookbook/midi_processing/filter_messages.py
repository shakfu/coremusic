#!/usr/bin/env python3
"""Filter Messages."""

# --8<-- [start:example]
from coremusic import capi
from coremusic.midi import MIDIMessageSplitter


class MIDIFilter:
    def __init__(self, filter_notes=False, filter_cc=False):
        self.filter_notes = filter_notes
        self.filter_cc = filter_cc
        self.splitter = MIDIMessageSplitter()

    def process(self, payload: bytes) -> None:
        for data in self.splitter.push(payload):
            message_type = data[0] & 0xF0

            # Filter note messages
            if message_type in (0x80, 0x90) and self.filter_notes:
                continue

            # Filter CC messages
            if message_type == 0xB0 and self.filter_cc:
                continue

            print(f"MIDI: {[hex(b) for b in data]}")

# Create filter that blocks notes but allows CC
midi_filter = MIDIFilter(filter_notes=True, filter_cc=False)

client = capi.midi_client_create("Filtered Input")
input_port = capi.midi_input_port_create(client, "Input")

# Connect, then feed polled packets to midi_filter.process()...
# --8<-- [end:example]
