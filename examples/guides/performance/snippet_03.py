#!/usr/bin/env python3
"""Releasing file handles."""

# --8<-- [start:example]
from coremusic.audio import AudioFile

# Good: Automatic cleanup via context manager
with AudioFile("audio.wav") as audio:
    data, count = audio.read_packets(0, 1024)
# File automatically closed here

# Also good: Explicit disposal
audio = AudioFile("audio.wav")
audio.open()
try:
    data, count = audio.read_packets(0, 1024)
finally:
    audio.dispose()  # Explicit cleanup
# --8<-- [end:example]
