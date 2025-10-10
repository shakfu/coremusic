# Import functional API and CoreAudioObject Base class
from .capi import *

# Import pure Python OO classes
from .objects import (
    # Exception hierarchy
    CoreAudioError,
    AudioFileError,
    AudioQueueError,
    AudioUnitError,
    MIDIError,
    MusicPlayerError,
    AudioDeviceError,

    # Audio formats and data structures
    AudioFormat,

    # Audio File Framework
    AudioFile,
    AudioFileStream,

    # Audio Queue Framework
    AudioBuffer,
    AudioQueue,

    # Audio Component & AudioUnit Framework
    AudioComponentDescription,
    AudioComponent,
    AudioUnit,

    # MIDI Framework
    MIDIClient,
    MIDIPort,
    MIDIInputPort,
    MIDIOutputPort,

    # Audio Device & Hardware
    AudioDevice,
    AudioDeviceManager,
)

# NumPy availability flag
try:
    from .objects import NUMPY_AVAILABLE
except ImportError:
    NUMPY_AVAILABLE = False
