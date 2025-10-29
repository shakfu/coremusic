#!/usr/bin/env python3
"""CoreMusic: Python bindings for Apple CoreAudio, CoreMIDI, and Ableton Link.

This package provides Python bindings for Apple's CoreAudio and CoreMIDI ecosystems,
plus Ableton Link tempo synchronization, exposing the APIs through Python.

The primary interface is the object-oriented API with automatic resource
management and context manager support. This is itself built=up from The low-level
functional C API which is available via the `capi` submodule for advanced use cases.

Usage:
    import coremusic as cm                # Object-oriented API
    import coremusic.capi as capi         # Functional C API
    import coremusic.utils.scipy as spu   # SciPy integration (optional)
    from coremusic import link            # Ableton Link synchronization
    from coremusic.midi import link       # Link + MIDI integration
    from coremusic.audio import async_io  # Async audio I/O
"""

# Import object-oriented API (primary interface)
from .objects import *

# Import async I/O classes from new location
from .audio.async_io import *

# Import high-level utilities
from .audio.utilities import *

# Import OSStatus error translation utilities
from . import os_status

# Import Ableton Link classes
from . import link

# Import Link + MIDI integration from new location
from .midi import link as link_midi

# Import subpackages
from . import audio
from . import midi
from . import utils

# Import AudioUnit hosting
from .audio_unit_host import (
    AudioUnitHost,
    AudioUnitPlugin,
    AudioUnitParameter,
    AudioUnitPreset,
    PluginAudioFormat,
    AudioFormatConverter,
    AudioUnitChain,
    PresetManager,
)


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
    # Ableton Link - Tempo Synchronization
    "link",  # link module containing LinkSession, SessionState, Clock
    "link_midi",  # Link + MIDI integration (LinkMIDIClock, LinkMIDISequencer)
    # AudioUnit Plugin Hosting
    "AudioUnitHost",
    "AudioUnitPlugin",
    "AudioUnitParameter",
    "AudioUnitPreset",
    "PluginAudioFormat",
    "AudioFormatConverter",
    "AudioUnitChain",
    "PresetManager",
    # NumPy availability flag
    "NUMPY_AVAILABLE",
]
