#!/usr/bin/env python3
"""Playback."""

# The file played back below is what the previous recipe records; write a short
# one here so this runs on its own.
import json

with open("recorded_midi.json", "w") as _f:
    json.dump(
        [
            {"time": 0.0, "data": [0x90, 60, 100]},
            {"time": 0.25, "data": [0x80, 60, 0]},
        ],
        _f,
    )


# --8<-- [start:example]
import json
import time

from coremusic import capi


def playback_midi(filename, output_port, dest):
    """Play back recorded MIDI"""
    # Load recorded messages
    with open(filename) as f:
        messages = json.load(f)

    if not messages:
        print("No messages to play")
        return

    print(f"Playing back {len(messages)} messages...")
    start_time = time.time()

    for msg in messages:
        # Wait until scheduled time
        target_time = start_time + msg['time']
        wait_time = target_time - time.time()

        if wait_time > 0:
            time.sleep(wait_time)

        # Send message
        data = bytes(msg['data'])
        capi.midi_send_data(output_port, dest, data)

    print("Playback complete")

# Setup playback
client = capi.midi_client_create("Playback")
output_port = capi.midi_output_port_create(client, "Output")
# Use the first destination, or publish one of our own when nothing else is
# connected, so this runs with or without hardware attached
if capi.midi_get_number_of_destinations() > 0:
    dest = capi.midi_get_destination(0)
else:
    dest = capi.midi_destination_create(client, "Cookbook Output")

# Play recording
playback_midi("recorded_midi.json", output_port, dest)

# Cleanup
capi.midi_port_dispose(output_port)
capi.midi_client_dispose(client)
# --8<-- [end:example]
