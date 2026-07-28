#!/usr/bin/env python3
"""List MIDI devices, sources, and destinations."""

# --8<-- [start:example]
from coremusic import capi
from coremusic.midi import get_destinations, get_sources


def list_midi_devices():
    """List all available MIDI devices, sources, and destinations."""
    device_count = capi.midi_get_number_of_devices()
    sources = get_sources()
    destinations = get_destinations()

    print(f"MIDI Devices: {device_count}")
    print(f"MIDI Sources: {len(sources)}")
    print(f"MIDI Destinations: {len(destinations)}")
    print()

    # A device is the physical unit; its entities carry the endpoints that
    # sources and destinations refer to.
    for i in range(device_count):
        device = capi.midi_get_device(i)
        try:
            print(f"Device {i}: {capi.midi_object_get_name(device)}")
        except Exception:
            print(f"Device {i}: <error reading name>")


if __name__ == "__main__":
    list_midi_devices()
# --8<-- [end:example]
