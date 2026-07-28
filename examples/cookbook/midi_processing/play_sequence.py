#!/usr/bin/env python3
"""Play Sequence."""

# --8<-- [start:example]
import time

from coremusic import capi


def play_note(port, dest, channel, note, velocity, duration):
    """Play a single note"""
    # Note On
    note_on = bytes([0x90 | channel, note, velocity])
    capi.midi_send_data(port, dest, note_on)

    # Wait
    time.sleep(duration)

    # Note Off
    note_off = bytes([0x80 | channel, note, 0])
    capi.midi_send_data(port, dest, note_off)

# Setup
client = capi.midi_client_create("Sequencer")
output_port = capi.midi_output_port_create(client, "Output")
# Use the first destination, or publish one of our own when nothing else is
# connected, so this runs with or without hardware attached
if capi.midi_get_number_of_destinations() > 0:
    dest = capi.midi_get_destination(0)
else:
    dest = capi.midi_destination_create(client, "Cookbook Output")

# Play C major scale
scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C

for note in scale:
    play_note(output_port, dest, channel=0, note=note, velocity=100, duration=0.5)
    time.sleep(0.1)  # Gap between notes

# Cleanup
capi.midi_port_dispose(output_port)
capi.midi_client_dispose(client)
# --8<-- [end:example]
