"""Type stubs for coremusic package

CoreMusic provides comprehensive Python bindings for Apple's CoreAudio ecosystem.
This package exposes both a functional C-style API and a modern object-oriented API.
"""

# Re-export everything from capi (functional API)
from .capi import AudioPlayer as AudioPlayer
from .capi import \
    CoreAudioObject as \
    CoreAudioObject  # Base classes; Utility functions; All other functions are available but not explicitly listed here; See capi.pyi for complete function signatures
from .capi import fourchar_to_int as fourchar_to_int
from .capi import int_to_fourchar as int_to_fourchar
# Re-export everything from objects (object-oriented API)
from .objects import NUMPY_AVAILABLE as NUMPY_AVAILABLE
from .objects import AudioBuffer as AudioBuffer
from .objects import AudioComponent as AudioComponent
from .objects import AudioComponentDescription as AudioComponentDescription
from .objects import AudioConverter as AudioConverter
from .objects import AudioConverterError as AudioConverterError
from .objects import AudioDevice as AudioDevice
from .objects import AudioDeviceError as AudioDeviceError
from .objects import AudioDeviceManager as AudioDeviceManager
from .objects import AudioFile as AudioFile
from .objects import AudioFileError as AudioFileError
from .objects import AudioFileStream as AudioFileStream
from .objects import AudioFormat as AudioFormat
from .objects import AudioQueue as AudioQueue
from .objects import AudioQueueError as AudioQueueError
from .objects import AudioUnit as AudioUnit
from .objects import AudioUnitError as AudioUnitError
from .objects import AUGraph as AUGraph
from .objects import AUGraphError as AUGraphError
from .objects import \
    CoreAudioError as \
    CoreAudioError  # Exception hierarchy; Audio formats and data structures; Audio File Framework; AudioConverter Framework; Audio Queue Framework; Audio Component & AudioUnit Framework; MIDI Framework; Audio Device & Hardware; AUGraph Framework; NumPy availability flag
from .objects import ExtendedAudioFile as ExtendedAudioFile
from .objects import MIDIClient as MIDIClient
from .objects import MIDIError as MIDIError
from .objects import MIDIInputPort as MIDIInputPort
from .objects import MIDIOutputPort as MIDIOutputPort
from .objects import MIDIPort as MIDIPort
from .objects import MusicPlayerError as MusicPlayerError

# Package metadata
__version__: str
__all__: list[str]
