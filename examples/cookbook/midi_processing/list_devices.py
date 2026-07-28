#!/usr/bin/env python3
"""List Devices."""

# --8<-- [start:example]
from coremusic import capi

# List MIDI sources (input devices)
num_sources = capi.midi_get_number_of_sources()
print(f"MIDI Sources: {num_sources}")

for i in range(num_sources):
    source = capi.midi_get_source(i)
    name = capi.midi_object_get_string_property(source, "name")
    print(f"  {i}: {name}")

# List MIDI destinations (output devices)
num_dests = capi.midi_get_number_of_destinations()
print(f"\nMIDI Destinations: {num_dests}")

for i in range(num_dests):
    dest = capi.midi_get_destination(i)
    name = capi.midi_object_get_string_property(dest, "name")
    print(f"  {i}: {name}")
# --8<-- [end:example]
