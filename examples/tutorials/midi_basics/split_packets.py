#!/usr/bin/env python3
"""Split polled packets into individual MIDI messages."""

import time

from coremusic.midi import MIDIClient

with MIDIClient("Splitter Demo") as client:
    input_port = client.create_input_port("Input")
    source = client.create_virtual_source("Splitter Demo Source")
    input_port.connect_source(source)
    time.sleep(0.2)

    # Two messages in one packet, which is exactly what a splitter is for.
    source.send(bytes([0x90, 60, 100, 0x80, 60, 0]))
    input_port.wait(2.0)

    # --8<-- [start:example]
    from coremusic.midi import MIDIMessageSplitter

    splitter = MIDIMessageSplitter()

    for _host_time, data in input_port.poll():
        for message in splitter.push(data):
            print(message.hex())
    # --8<-- [end:example]
