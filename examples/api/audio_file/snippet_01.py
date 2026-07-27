#!/usr/bin/env python3
"""AudioFile Class."""

# --8<-- [start:example]
from coremusic.audio import AudioFile

# Context manager usage (recommended)
with AudioFile("audio.wav") as audio:
    print(f"Duration: {audio.duration:.2f}s")
    data, count = audio.read_packets(0, 1000)

# Explicit management
audio = AudioFile("audio.wav")
audio.open()
try:
    data = audio.read_packets(0, 1000)
finally:
    audio.close()
# --8<-- [end:example]
