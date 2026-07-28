#!/usr/bin/env python3
"""Find Device."""

# --8<-- [start:example]
from coremusic import capi


def find_midi_source(device_name):
    """Find MIDI source by name"""
    num_sources = capi.midi_get_number_of_sources()

    for i in range(num_sources):
        source = capi.midi_get_source(i)
        name = capi.midi_object_get_string_property(source, "name")
        if device_name.lower() in name.lower():
            return source, name

    return None, None

# Find a specific device
source, name = find_midi_source("Keyboard")
if source:
    print(f"Found: {name}")
else:
    print("Device not found")
# --8<-- [end:example]
