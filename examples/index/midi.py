#!/usr/bin/env python3
"""Send a Note On to a MIDI destination."""

# --8<-- [start:example]
from coremusic.midi import MIDIClient, get_destinations, note_on

# Create MIDI client
client = MIDIClient("My MIDI App")
try:
    output_port = client.create_output_port("Output")

    # Aim at an endpoint published by the system, or create a virtual one
    destinations = get_destinations()
    if destinations:
        destination = destinations[0]
    else:
        destination = client.create_virtual_destination("Synth")

    # Send MIDI data: Note On, middle C, full velocity
    output_port.send_data(destination, note_on("C4", 127))
finally:
    client.dispose()
# --8<-- [end:example]
