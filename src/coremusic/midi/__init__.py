#!/usr/bin/env python3
"""CoreMusic MIDI package.

This package contains MIDI-related modules:
- core: MIDI client and port classes (MIDIClient, MIDIPort, MIDIInputPort, MIDIOutputPort)
- player: Music sequencing and playback (MusicPlayer, MusicSequence, MusicTrack)
- link: Ableton Link + MIDI integration (LinkMIDIClock, LinkMIDISequencer)
- utilities: High-level MIDI file I/O, sequencing, and routing
- transform: MIDI transformation pipeline (Transpose, Quantize, Humanize, etc.)
"""

# Import transform module
# Import link module
from . import link, transform

# Import domain object classes
from .core import MIDIClient, MIDIInputPort, MIDIOutputPort, MIDIPort
from .player import MusicPlayer, MusicSequence, MusicTrack

# Import transform classes
from .transform import (  # Base classes; Pitch transformers; Time transformers; Velocity transformers; Filter transformers; Track transformers; Arpeggio; Convenience functions
    Arpeggiate,
    ChannelRemap,
    EventTypeFilter,
    Harmonize,
    Humanize,
    Invert,
    MIDITransformer,
    NoteFilter,
    Pipeline,
    Quantize,
    Reverse,
    TimeShift,
    TimeStretch,
    TrackMerge,
    Transpose,
    VelocityCurve,
    VelocityScale,
    humanize,
    quantize,
    reverse,
    scale_velocity,
    transpose,
)

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
    # MIDI client and ports
    "MIDIClient",
    "MIDIPort",
    "MIDIInputPort",
    "MIDIOutputPort",
    # Music player
    "MusicPlayer",
    "MusicSequence",
    "MusicTrack",
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
