#!/usr/bin/env python3
"""CoreMusic: Complete Python bindings for Apple CoreAudio and CoreMIDI.

This package provides comprehensive Python bindings for Apple's CoreAudio and
CoreMIDI ecosystems, exposing the complete CoreAudio and CoreMIDI C APIs
through Python.

The primary interface is the object-oriented API with automatic resource
management and context manager support. The low-level functional C API is
available via the `capi` submodule for advanced use cases.

Usage:
    import coremusic as cm              # Object-oriented API
    import coremusic.capi as capi       # Functional C API
    import coremusic.scipy_utils as spu # SciPy integration (optional)
"""

# Import object-oriented API (primary interface)
from .objects import *

# Import async I/O classes
from .async_io import *

# Import high-level utilities
from .utilities import *


__all__ = [
    # Base class
    "CoreAudioObject",
    # Exception hierarchy
    "CoreAudioError",
    "AudioFileError",
    "AudioQueueError",
    "AudioUnitError",
    "AudioConverterError",
    "MIDIError",
    "MusicPlayerError",
    "AudioDeviceError",
    "AUGraphError",
    # Audio formats and data structures
    "AudioFormat",
    # Audio File Framework
    "AudioFile",
    "AudioFileStream",
    "ExtendedAudioFile",
    # AudioConverter Framework
    "AudioConverter",
    # Audio Queue Framework
    "AudioBuffer",
    "AudioQueue",
    # Audio Component & AudioUnit Framework
    "AudioComponentDescription",
    "AudioComponent",
    "AudioUnit",
    # MIDI Framework
    "MIDIClient",
    "MIDIPort",
    "MIDIInputPort",
    "MIDIOutputPort",
    # Audio Device & Hardware
    "AudioDevice",
    "AudioDeviceManager",
    # AUGraph Framework
    "AUGraph",
    # CoreAudioClock - Synchronization and Timing
    "AudioClock",
    "ClockTimeFormat",
    # Audio Player
    "AudioPlayer",
    # NumPy availability flag
    "NUMPY_AVAILABLE",
]
