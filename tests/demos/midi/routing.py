#!/usr/bin/env python3
"""MIDI routing with transforms.

Usage:
    python routing.py
"""

from coremusic.midi import (
    MIDIRouter,
    MIDIEvent,
    MIDIStatus,
    transpose_transform,
    velocity_scale_transform,
)


def main():
    router = MIDIRouter()

    # Add transforms
    router.add_transform("transpose_up", transpose_transform(12))
    router.add_transform("softer", velocity_scale_transform(0.7))

    # Add routes
    router.add_route("keyboard", "high_synth", transform="transpose_up")
    router.add_route("keyboard", "pad", transform="softer")

    # Test event
    event = MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100)
    print(f"Input: Note {event.data1}, velocity {event.data2}")

    results = router.process_event("keyboard", event)
    for dest, routed_event in results:
        print(f"  -> {dest}: Note {routed_event.data1}, velocity {routed_event.data2}")


if __name__ == "__main__":
    main()
