#!/usr/bin/env python3
"""Send a note on/off pair to the first available destination."""

# --8<-- [start:example]
import time

from coremusic.midi import MIDIClient, get_destinations


def send_note(port, destination, note, velocity=100, duration=0.5, channel=0):
    """Send a note on/off pair."""
    # Status byte: 0x90 + channel (Note On on channel)
    note_on = bytes([0x90 + channel, note, velocity])

    # Status byte: 0x80 + channel (Note Off on channel)
    note_off = bytes([0x80 + channel, note, 0])

    port.send_data(destination, note_on)
    print(f"Note On: {note} velocity={velocity}")

    time.sleep(duration)

    port.send_data(destination, note_off)
    print(f"Note Off: {note}")


# Send middle C
with MIDIClient("Note Sender") as client:
    port = client.create_output_port("Output")

    destinations = get_destinations()
    if not destinations:
        print("No MIDI destinations available")
    else:
        send_note(port, destinations[0], note=60, velocity=100, duration=0.5)
# --8<-- [end:example]
