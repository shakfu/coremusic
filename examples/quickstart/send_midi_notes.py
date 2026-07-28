#!/usr/bin/env python3
"""Send a note on/off pair."""

# --8<-- [start:example]
import time

from coremusic.midi import MIDIClient, get_destinations, note_off, note_on

client = MIDIClient("My App")
port = client.create_output_port("Output")

destinations = get_destinations()
dest = destinations[0] if destinations else client.create_virtual_destination("Synth")

# Send Note On (middle C, velocity 100)
port.send_data(dest, note_on("C4", 100))
time.sleep(0.5)

# Send Note Off
port.send_data(dest, note_off("C4"))

client.dispose()
# --8<-- [end:example]
