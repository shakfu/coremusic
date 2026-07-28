#!/usr/bin/env python3
"""Opening an audio file, with and without a context manager."""

# --8<-- [start:context-manager]
from coremusic.audio import AudioFile

# Open with context manager (automatic cleanup)
with AudioFile("audio.wav") as audio:
    # File is automatically opened
    print(f"Opened: {audio.path}")

# File is automatically closed here
# --8<-- [end:context-manager]

# --8<-- [start:manual]
from coremusic import capi

# Open file manually
file_id = capi.audio_file_open_url("audio.wav")
try:
    # Work with file
    print(f"Opened file: {file_id}")
finally:
    # Always close, even on error
    capi.audio_file_close(file_id)
# --8<-- [end:manual]
