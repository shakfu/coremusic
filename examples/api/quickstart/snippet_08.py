#!/usr/bin/env python3
"""Create MIDI Client."""

# --8<-- [start:example]
from coremusic.midi import MIDIClient

client = MIDIClient("My App")
try:
    # Create ports
    output_port = client.create_output_port("Output")
    input_port = client.create_input_port("Input")

    # Use ports...
finally:
    client.dispose()
# --8<-- [end:example]
