#!/usr/bin/env python3
"""Send Control Change messages."""

# --8<-- [start:example]
from coremusic.midi import MIDIClient, get_destinations


def send_cc(port, destination, controller, value, channel=0):
    """Send Control Change message."""
    # CC message: 0xB0 + channel, controller number, value
    port.send_data(destination, bytes([0xB0 + channel, controller, value]))
    print(f"CC {controller}: {value}")


# Common CC numbers:
# CC 1  = Modulation wheel
# CC 7  = Volume
# CC 10 = Pan
# CC 64 = Sustain pedal
# CC 123 = All Notes Off

with MIDIClient("CC Sender") as client:
    port = client.create_output_port("Output")

    destinations = get_destinations()
    if not destinations:
        print("No MIDI destinations available")
    else:
        # Send modulation
        send_cc(port, destinations[0], controller=1, value=64)

        # Send volume
        send_cc(port, destinations[0], controller=7, value=100)
# --8<-- [end:example]
