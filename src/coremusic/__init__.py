#!/usr/bin/env python3
"""CoreMusic: Complete Python bindings for Apple CoreAudio, CoreMIDI, and Ableton Link.

This package provides comprehensive Python bindings for Apple's CoreAudio and
CoreMIDI ecosystems, plus Ableton Link tempo synchronization, exposing the
complete APIs through Python.

The primary interface is the object-oriented API with automatic resource
management and context manager support. The low-level functional C API is
available via the `capi` submodule for advanced use cases.

Usage:
    import coremusic as cm              # Object-oriented API
    import coremusic.capi as capi       # Functional C API
    import coremusic.scipy_utils as spu # SciPy integration (optional)
    from coremusic import link          # Ableton Link synchronization
"""

# Import object-oriented API (primary interface)
from .objects import *

# Import async I/O classes
from .async_io import *

# Import high-level utilities
from .utilities import *

# Import Ableton Link classes
from . import link

# Import Link + MIDI integration
from . import link_midi

# Import AudioUnit hosting
from .audio_unit_host import (
    AudioUnitHost,
    AudioUnitPlugin,
    AudioUnitParameter,
    AudioUnitPreset,
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
    # NumPy availability flag
    "NUMPY_AVAILABLE",
]
