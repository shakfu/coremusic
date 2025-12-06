#!/usr/bin/env python3
"""CoreMusic: Python bindings for Apple CoreAudio, CoreMIDI, and Ableton Link.

This package provides Python bindings for Apple's CoreAudio and CoreMIDI ecosystems,
plus Ableton Link tempo synchronization, exposing the APIs through Python.

The primary interface is the object-oriented API with automatic resource
management and context manager support. This is itself built-up from the low-level
functional C API which is available via the `capi` submodule for advanced use cases.

Usage:
    import coremusic as cm                # Object-oriented API
    import coremusic.capi as capi         # Functional C API
    import coremusic.utils.scipy as spu   # SciPy integration (optional)
    from coremusic import link            # Ableton Link synchronization
    from coremusic.midi import link       # Link + MIDI integration
    from coremusic.audio import async_io  # Async audio I/O

    # Constants (preferred over capi getter functions):
    from coremusic.constants import AudioFileProperty, AudioFormatID
    format_id = AudioFileProperty.DATA_FORMAT
"""

# Import object-oriented API (primary interface)
from .objects import *

# Import async I/O classes from new location
from .audio.async_io import *

# Import high-level utilities
from .audio.utilities import *

# Import OSStatus error translation utilities
from . import os_status
from .os_status import (
    check_os_status,
    check_return_status,
    raises_on_error,
    handle_exceptions,
    format_os_status_error,
)

# Import buffer management utilities
from . import buffer_utils
from .buffer_utils import (
    AudioStreamBasicDescription,
    pack_audio_buffer,
    unpack_audio_buffer,
    calculate_buffer_size,
    optimal_buffer_size,
)

# Import Ableton Link classes
from . import link

# Import Link + MIDI integration from new location
from .midi import link as link_midi

# Import subpackages
from . import audio
from . import midi
from . import utils
from . import daw
from . import constants

# Import commonly-used constants for convenience
from .constants import (
    # Audio File
    AudioFileProperty,
    AudioFileType,
    AudioFilePermission,
    # Audio Format
    AudioFormatID,
    LinearPCMFormatFlag,
    # Audio Converter
    AudioConverterProperty,
    AudioConverterQuality,
    # Extended Audio File
    ExtendedAudioFileProperty,
    # AudioUnit
    AudioUnitProperty,
    AudioUnitScope,
    AudioUnitElement,
    AudioUnitRenderActionFlags,
    AudioUnitParameterUnit,
    # Audio Queue
    AudioQueueProperty,
    AudioQueueParameter,
    # Audio Object/Device
    AudioObjectProperty,
    AudioDeviceProperty,
    # MIDI
    MIDIStatus,
    MIDIControlChange,
    MIDIObjectProperty,
)

# Import DAW classes for convenience
from .daw import (
    Timeline,
    Track,
    Clip,
    TimelineMarker,
    TimeRange,
    AutomationLane,
)

# Import AudioUnit hosting
from .audio.audiounit_host import (
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
    # MusicPlayer Framework
    "MusicPlayer",
    "MusicSequence",
    "MusicTrack",
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
    # DAW Essentials
    "daw",  # daw module
    "Timeline",
    "Track",
    "Clip",
    "TimelineMarker",
    "TimeRange",
    "AutomationLane",
    # Error handling decorators
    "check_os_status",
    "check_return_status",
    "raises_on_error",
    "handle_exceptions",
    "format_os_status_error",
    # Buffer management
    "buffer_utils",  # buffer_utils module
    "AudioStreamBasicDescription",
    "pack_audio_buffer",
    "unpack_audio_buffer",
    "calculate_buffer_size",
    "optimal_buffer_size",
    # Constants module and enum classes (preferred over capi getter functions)
    "constants",  # constants module with all enum classes
    "AudioFileProperty",
    "AudioFileType",
    "AudioFilePermission",
    "AudioFormatID",
    "LinearPCMFormatFlag",
    "AudioConverterProperty",
    "AudioConverterQuality",
    "ExtendedAudioFileProperty",
    "AudioUnitProperty",
    "AudioUnitScope",
    "AudioUnitElement",
    "AudioUnitRenderActionFlags",
    "AudioUnitParameterUnit",
    "AudioQueueProperty",
    "AudioQueueParameter",
    "AudioObjectProperty",
    "AudioDeviceProperty",
    "MIDIStatus",
    "MIDIControlChange",
    "MIDIObjectProperty",
]
