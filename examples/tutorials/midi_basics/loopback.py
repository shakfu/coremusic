#!/usr/bin/env python3
"""Send to a virtual destination owned by the same client."""

# --8<-- [start:example]
from coremusic.midi import MIDIClient, note_on

with MIDIClient("Loopback") as client:
    destination = client.create_virtual_destination("Loopback In")
    port = client.create_output_port("Loopback Out")

    port.send_data(destination, note_on("C4", 100))

    assert destination.wait(1.0)
    print(destination.poll())
# --8<-- [end:example]
