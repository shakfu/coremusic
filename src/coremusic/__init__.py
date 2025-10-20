#!/usr/bin/env python3
"""CoreMusic: Complete Python bindings for Apple CoreAudio and CoreMIDI.

This package provides comprehensive Python bindings for Apple's CoreAudio and
CoreMIDI ecosystems, exposing the complete CoreAudio and CoreMIDI C APIs
through Python. It offers both traditional functional APIs and modern
object-oriented interfaces, with automatic resource management and context
manager support.
"""

# Import functional API
from .capi import *

# Import object-oriented API
from .objects import *

# Import async I/O classes
from .async_io import *

# Import high-level utilities
from .utilities import *


__all__ = [
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

    # NumPy availability flag
    "NUMPY_AVAILABLE",
]