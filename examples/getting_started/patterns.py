#!/usr/bin/env python3
"""Resource management and error handling, both APIs."""

# --8<-- [start:context-managers]
from coremusic.audio import AudioFile

# Good - automatic cleanup
with AudioFile("audio.wav") as audio:
    data = audio.read_packets(0, 1000)

# Also good - explicit but safe
audio = AudioFile("audio.wav")
try:
    audio.open()
    data = audio.read_packets(0, 1000)
finally:
    audio.close()
# --8<-- [end:context-managers]

# --8<-- [start:error-handling]
from coremusic.audio import AudioFile
from coremusic.exceptions import AudioFileError

try:
    with AudioFile("audio.wav") as audio:
        data = audio.read_packets(0, 1000)
except AudioFileError as e:
    print(f"Audio file error: {e}")
except FileNotFoundError:
    print("File not found")
# --8<-- [end:error-handling]

# --8<-- [start:functional-cleanup]
import coremusic.capi as capi

# Functional API - manual cleanup required
audio_file = capi.audio_file_open_url("audio.wav")
try:
    # Use the file
    data = capi.audio_file_read_packets(audio_file, 0, 1000)
finally:
    # Always close, even if errors occur
    capi.audio_file_close(audio_file)
# --8<-- [end:functional-cleanup]
