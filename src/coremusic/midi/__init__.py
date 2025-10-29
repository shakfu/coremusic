#!/usr/bin/env python3
"""CoreMusic MIDI package.

This package contains MIDI-related modules:
- link: Ableton Link + MIDI integration (LinkMIDIClock, LinkMIDISequencer)
- utilities: High-level MIDI file I/O, sequencing, and routing
"""

# Import link module
from . import link

# Import utilities classes
from .utilities import (
    MIDIEvent,
    MIDIFileFormat,
    MIDIRouter,
    MIDISequence,
    MIDIStatus,
    MIDITrack,
    Route,
    channel_remap_transform,
    quantize_transform,
    transpose_transform,
    velocity_curve_transform,
    velocity_scale_transform,
)

__all__ = [
    # Submodules
    "link",
    # Core classes
    "MIDIEvent",
    "MIDITrack",
    "MIDISequence",
    "MIDIRouter",
    "Route",
    # Enums
    "MIDIFileFormat",
    "MIDIStatus",
    # Transform functions
    "transpose_transform",
    "velocity_scale_transform",
    "velocity_curve_transform",
    "channel_remap_transform",
    "quantize_transform",
]
