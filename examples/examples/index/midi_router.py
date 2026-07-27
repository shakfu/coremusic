#!/usr/bin/env python3
"""Forward everything arriving on an input port to a destination."""

# --8<-- [start:example]
import time

from coremusic.midi import MIDIClient, get_destinations, get_sources

client = MIDIClient("Router")
input_port = client.create_input_port("Input")
output_port = client.create_output_port("Output")

for source in get_sources():
    input_port.connect_source(source)

destinations = get_destinations()
destination = (
    destinations[0] if destinations else client.create_virtual_destination("Routed")
)

# Route MIDI data for a second. A real router would run until stopped.
deadline = time.monotonic() + 1.0
while time.monotonic() < deadline:
    if input_port.wait(0.1):
        for _host_time, midi_data in input_port.poll():
            output_port.send_data(destination, midi_data)

client.dispose()
# --8<-- [end:example]
