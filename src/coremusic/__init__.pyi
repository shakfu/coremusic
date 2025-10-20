"""Type stubs for coremusic package

CoreMusic provides comprehensive Python bindings for Apple's CoreAudio ecosystem.
This package exposes both a functional C-style API and a modern object-oriented API.
"""

# Re-export everything from capi (functional API)
from .capi import (
    # Base classes
    CoreAudioObject as CoreAudioObject,
    AudioPlayer as AudioPlayer,
    # Utility functions
    fourchar_to_int as fourchar_to_int,
    int_to_fourchar as int_to_fourchar,
    # All other functions are available but not explicitly listed here
    # See capi.pyi for complete function signatures
)

# Re-export everything from objects (object-oriented API)
from .objects import (
    # Exception hierarchy
    CoreAudioError as CoreAudioError,
    AudioFileError as AudioFileError,
    AudioQueueError as AudioQueueError,
    AudioUnitError as AudioUnitError,
    AudioConverterError as AudioConverterError,
    MIDIError as MIDIError,
    MusicPlayerError as MusicPlayerError,
    AudioDeviceError as AudioDeviceError,
    AUGraphError as AUGraphError,
    # Audio formats and data structures
    AudioFormat as AudioFormat,
    # Audio File Framework
    AudioFile as AudioFile,
    AudioFileStream as AudioFileStream,
    ExtendedAudioFile as ExtendedAudioFile,
    # AudioConverter Framework
    AudioConverter as AudioConverter,
    # Audio Queue Framework
    AudioBuffer as AudioBuffer,
    AudioQueue as AudioQueue,
    # Audio Component & AudioUnit Framework
    AudioComponentDescription as AudioComponentDescription,
    AudioComponent as AudioComponent,
    AudioUnit as AudioUnit,
    # MIDI Framework
    MIDIClient as MIDIClient,
    MIDIPort as MIDIPort,
    MIDIInputPort as MIDIInputPort,
    MIDIOutputPort as MIDIOutputPort,
    # Audio Device & Hardware
    AudioDevice as AudioDevice,
    AudioDeviceManager as AudioDeviceManager,
    # AUGraph Framework
    AUGraph as AUGraph,
    # NumPy availability flag
    NUMPY_AVAILABLE as NUMPY_AVAILABLE,
)

# Package metadata
__version__: str
__all__: list[str]
