#!/usr/bin/env python3
"""Receive MIDI by polling an input port."""

# --8<-- [start:example]
import time

from coremusic.midi import MIDIClient, get_sources

with MIDIClient("MIDI Receiver") as client:
    input_port = client.create_input_port("Input")

    # Connect to all sources
    for source in get_sources():
        input_port.connect_source(source)

    # Listen for a second. A real program would loop until it is told to stop.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if input_port.wait(0.1):
            for host_time, data in input_port.poll():
                print(host_time, data.hex())
# --8<-- [end:example]
