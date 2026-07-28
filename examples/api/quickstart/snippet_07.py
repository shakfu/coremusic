#!/usr/bin/env python3
"""List MIDI Devices."""

# --8<-- [start:example]
from coremusic import capi

# Count devices
num_devices = capi.midi_get_number_of_devices()
num_sources = capi.midi_get_number_of_sources()
num_destinations = capi.midi_get_number_of_destinations()

print(f"Devices: {num_devices}")
print(f"Sources: {num_sources}")
print(f"Destinations: {num_destinations}")
# --8<-- [end:example]
