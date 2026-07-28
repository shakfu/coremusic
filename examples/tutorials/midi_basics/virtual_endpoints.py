#!/usr/bin/env python3
"""Publish virtual endpoints that other applications can connect to."""

# --8<-- [start:destination]
import time

from coremusic.midi import MIDIClient

with MIDIClient("My Synth") as client:
    # Other applications now see "My Synth Input" as a MIDI output
    destination = client.create_virtual_destination("My Synth Input")

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if destination.wait(0.1):
            for _host_time, data in destination.poll():
                print(data.hex())
# --8<-- [end:destination]

with MIDIClient("My Synth (callback)") as client:
    # --8<-- [start:destination-callback]
    destination = client.create_virtual_destination(
        "My Synth Input", callback=lambda data, host_time: print(data.hex())
    )
    # --8<-- [end:destination-callback]
    time.sleep(0.2)

# --8<-- [start:source]
from coremusic.midi import MIDIClient

with MIDIClient("My Controller") as client:
    # Other applications now see "My Controller Out" as a MIDI input
    source = client.create_virtual_source("My Controller Out")

    source.send(bytes([0x90, 60, 100]))  # Note On
    source.send(bytes([0x80, 60, 0]))    # Note Off
# --8<-- [end:source]
