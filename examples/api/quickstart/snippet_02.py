#!/usr/bin/env python3
"""Read Audio File."""

# --8<-- [start:example]
from coremusic.audio import AudioFile

# Context manager (recommended)
with AudioFile("audio.wav") as audio:
    print(f"Duration: {audio.duration:.2f}s")
    print(f"Sample rate: {audio.format.sample_rate}Hz")
    data, count = audio.read_packets(0, 1024)
# --8<-- [end:example]
