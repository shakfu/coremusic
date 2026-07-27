#!/usr/bin/env python3
"""Create an output port, and choose a destination endpoint to send to."""

# --8<-- [start:port]
from coremusic.midi import MIDIClient


def setup_midi_output():
    """Set up MIDI output."""
    client = MIDIClient("MIDI Sender")
    output_port = client.create_output_port("Output")

    return client, output_port


client, port = setup_midi_output()
# --8<-- [end:port]

# --8<-- [start:destination]
from coremusic.midi import find_destination, get_destinations

destinations = get_destinations()
destination = destinations[0] if destinations else None

# Or search by name (exact match first, then case-insensitive substring)
iac = find_destination("IAC Driver")
# --8<-- [end:destination]

client.dispose()
