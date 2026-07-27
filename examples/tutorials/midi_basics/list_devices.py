#!/usr/bin/env python3
"""List the MIDI sources and destinations published by the system."""

# --8<-- [start:example]
from coremusic.midi import get_destinations, get_sources


def list_midi_devices():
    """List all MIDI sources and destinations."""
    sources = get_sources()
    destinations = get_destinations()

    print("MIDI System Overview:")
    print(f"  Sources (inputs): {len(sources)}")
    print(f"  Destinations (outputs): {len(destinations)}")
    print()

    print("MIDI Sources (Inputs):")
    for i, source in enumerate(sources):
        print(f"  [{i}] {source.name or '<unknown>'}")

    print()

    print("MIDI Destinations (Outputs):")
    for i, destination in enumerate(destinations):
        print(f"  [{i}] {destination.name or '<unknown>'}")


list_midi_devices()
# --8<-- [end:example]
