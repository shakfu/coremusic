"""CoreAudio constant enumerations for CoreMusic.

This module provides Pythonic Enum classes for CoreAudio constants, offering
better IDE support, type safety, and discoverability compared to individual
getter functions. All existing getter functions remain available for backward
compatibility.

The constants are organized by category:
- AudioFileProperty: Audio file properties
- AudioFileType: Audio file types
- AudioFormatID: Audio format identifiers
- LinearPCMFormatFlag: Linear PCM format flags
- AudioConverterProperty: Audio converter properties
- AudioConverterQuality: Converter quality settings
- AudioUnitProperty: AudioUnit properties
- AudioUnitScope: AudioUnit scopes
- AudioUnitRenderActionFlags: Render action flags
- MIDIStatus: MIDI status bytes

Usage::

    import coremusic as cm
    from coremusic.constants import AudioFileProperty, AudioFormatID

    # Use enum values
    format_property = AudioFileProperty.DATA_FORMAT
    format_id = AudioFormatID.LINEAR_PCM

    # Convert to integer for API calls
    property_id = int(format_property)  # or format_property.value

    # Compare with integers
    if some_value == AudioFileProperty.DATA_FORMAT:
        print("It's the data format property")

    # Backward compatible: getter functions still work
    property_id = cm.capi.get_audio_file_property_data_format()
"""

from coremusic.constants.audio import (
    AudioConverterProperty,
    AudioConverterQuality,
    AudioFilePermission,
    AudioFileProperty,
    AudioFileType,
    AudioFormatID,
    ExtendedAudioFileProperty,
    LinearPCMFormatFlag,
)
from coremusic.constants.audiounit import (
    AudioUnitElement,
    AudioUnitParameterUnit,
    AudioUnitProperty,
    AudioUnitRenderActionFlags,
    AudioUnitScope,
)
from coremusic.constants.device import AudioDeviceProperty, AudioObjectProperty
from coremusic.constants.midi import MIDIControlChange, MIDIObjectProperty, MIDIStatus
from coremusic.constants.queue import AudioQueueParameter, AudioQueueProperty

__all__ = [
    # Audio File
    "AudioFileProperty",
    "AudioFileType",
    "AudioFilePermission",
    # Audio Format
    "AudioFormatID",
    "LinearPCMFormatFlag",
    # Audio Converter
    "AudioConverterProperty",
    "AudioConverterQuality",
    # Extended Audio File
    "ExtendedAudioFileProperty",
    # Audio Unit
    "AudioUnitProperty",
    "AudioUnitScope",
    "AudioUnitElement",
    "AudioUnitRenderActionFlags",
    "AudioUnitParameterUnit",
    # Audio Queue
    "AudioQueueProperty",
    "AudioQueueParameter",
    # Audio Object/Device
    "AudioObjectProperty",
    "AudioDeviceProperty",
    # MIDI
    "MIDIStatus",
    "MIDIControlChange",
    "MIDIObjectProperty",
]
