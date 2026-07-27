#!/usr/bin/env python3
"""Exception Hierarchy."""

# --8<-- [start:example]
from coremusic.audio import AudioFile
from coremusic.exceptions import AUGraphError, AudioConverterError, AudioDeviceError, AudioFileError, AudioQueueError, AudioUnitError, CoreAudioError, MIDIError, MusicPlayerError

try:
    with AudioFile("missing.wav") as audio:
        pass
except AudioFileError as e:
    print(f"Audio file error: {e}")
except CoreAudioError as e:
    print(f"CoreAudio error: {e}")

# Specific exception types:
# - AudioFileError
# - AudioQueueError
# - AudioUnitError
# - AudioConverterError
# - MIDIError
# - MusicPlayerError
# - AudioDeviceError
# - AUGraphError
# --8<-- [end:example]
