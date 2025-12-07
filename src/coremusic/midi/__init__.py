#!/usr/bin/env python3
"""CoreMusic MIDI package.

This package contains MIDI-related modules:
- link: Ableton Link + MIDI integration (LinkMIDIClock, LinkMIDISequencer)
- utilities: High-level MIDI file I/O, sequencing, and routing
- transform: MIDI transformation pipeline (Transpose, Quantize, Humanize, etc.)
"""

# Import link module
from . import link

# Import transform module
from . import transform

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

# Import transform classes
from .transform import (
    # Base classes
    MIDITransformer,
    Pipeline,
    # Pitch transformers
    Transpose,
    Invert,
    Harmonize,
    # Time transformers
    Quantize,
    TimeStretch,
    TimeShift,
    Reverse,
    # Velocity transformers
    VelocityScale,
    VelocityCurve,
    Humanize,
    # Filter transformers
    NoteFilter,
    EventTypeFilter,
    # Track transformers
    ChannelRemap,
    TrackMerge,
    # Arpeggio
    Arpeggiate,
    # Convenience functions
    transpose,
    quantize,
    humanize,
    reverse,
    scale_velocity,
)

__all__ = [
    # Submodules
    "link",
    "transform",
    # Core classes
    "MIDIEvent",
    "MIDITrack",
    "MIDISequence",
    "MIDIRouter",
    "Route",
    # Enums
    "MIDIFileFormat",
    "MIDIStatus",
    # Legacy transform functions (from utilities)
    "transpose_transform",
    "velocity_scale_transform",
    "velocity_curve_transform",
    "channel_remap_transform",
    "quantize_transform",
    # Transform classes (new pipeline API)
    "MIDITransformer",
    "Pipeline",
    "Transpose",
    "Invert",
    "Harmonize",
    "Quantize",
    "TimeStretch",
    "TimeShift",
    "Reverse",
    "VelocityScale",
    "VelocityCurve",
    "Humanize",
    "NoteFilter",
    "EventTypeFilter",
    "ChannelRemap",
    "TrackMerge",
    "Arpeggiate",
    # Convenience functions
    "transpose",
    "quantize",
    "humanize",
    "reverse",
    "scale_velocity",
]
