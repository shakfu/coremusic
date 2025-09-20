# Import functional API
from .capi import *

# Import object-oriented API
try:
    # Import base Cython extension class
    from .objects import CoreAudioObject

    # Import pure Python OO classes
    from .oo import (
        # Exception hierarchy
        CoreAudioError,
        AudioFileError,
        AudioQueueError,
        AudioUnitError,
        MIDIError,
        MusicPlayerError,

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
    )
except ImportError as e:
    # If object-oriented API import fails, still provide functional API
    import warnings
    warnings.warn(f"Object-oriented API not available: {e}", UserWarning)
