#!/usr/bin/env python3
"""Open an audio file and read packets from it."""

# --8<-- [start:example]
from coremusic.audio import AudioFile

# Object-oriented API (recommended)
with AudioFile("audio.wav") as audio:
    print(f"Duration: {audio.duration:.2f}s")
    print(f"Sample rate: {audio.format.sample_rate}Hz")
    data, count = audio.read_packets(0, 1000)
# --8<-- [end:example]
