#!/usr/bin/env python3
"""Receive MIDI through a callback instead of polling."""

# --8<-- [start:example]
import time

from coremusic.midi import MIDIClient, get_sources


def midi_callback(data, host_time):
    """Callback for incoming MIDI data."""
    print(f"Received {data.hex()} at {host_time}")


with MIDIClient("MIDI Receiver") as client:
    input_port = client.create_input_port("Input", callback=midi_callback)

    for source in get_sources():
        input_port.connect_source(source)

    # The callback fires on the CoreMIDI receive thread while we wait here.
    time.sleep(1.0)
# --8<-- [end:example]
