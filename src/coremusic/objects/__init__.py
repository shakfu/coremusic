"""Object-oriented Python classes for coremusic.

This package provides Pythonic, object-oriented wrappers around the CoreAudio
functional API. These classes handle resource management automatically and
provide a more intuitive interface for CoreAudio development.

The classes are organized into submodules by functionality:
- exceptions: Exception hierarchy
- base: Base classes and shared utilities
- audio: Audio file and format classes
- audiounit: AudioUnit classes
- midi: MIDI classes
- devices: Audio device classes
- augraph: AUGraph classes
- clock: Audio clock classes
- music: Music player classes

For convenience, all public classes are re-exported from this package.

Usage:
    # Import from package (recommended)
    from coremusic.objects import AudioFile, AudioFormat

    # Or import from main module (also works)
    from coremusic import AudioFile, AudioFormat
"""

# Import from submodules
from .base import NUMPY_AVAILABLE, AudioPlayer, CoreAudioObject
from .exceptions import (
    AudioConverterError,
    AudioDeviceError,
    AudioFileError,
    AudioQueueError,
    AudioUnitError,
    AUGraphError,
    CoreAudioError,
    MIDIError,
    MusicPlayerError,
)

# Import audio classes
from .audio import (
    AudioBuffer,
    AudioConverter,
    AudioFile,
    AudioFileStream,
    AudioFormat,
    AudioQueue,
    ExtendedAudioFile,
)

# Import AudioUnit classes
from .audiounit import (
    AudioComponent,
    AudioComponentDescription,
    AudioUnit,
)

# Import MIDI classes
from .midi import (
    MIDIClient,
    MIDIInputPort,
    MIDIOutputPort,
    MIDIPort,
)

# Import device classes
from .devices import (
    AudioDevice,
    AudioDeviceManager,
)

# Import AUGraph classes
from .augraph import (
    AUGraph,
)

# Import clock classes
from .clock import (
    AudioClock,
    ClockTimeFormat,
)

# Import music player classes
from .music import (
    MusicPlayer,
    MusicSequence,
    MusicTrack,
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
    # NumPy availability flag
    "NUMPY_AVAILABLE",
]
