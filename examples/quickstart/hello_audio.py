#!/usr/bin/env python3
"""Open an audio file and print its format."""

# --8<-- [start:example]
from coremusic.audio import AudioFile

# Open and inspect an audio file
with AudioFile("audio.wav") as audio:
    print(f"Duration: {audio.duration:.2f} seconds")
    print(f"Sample Rate: {audio.format.sample_rate} Hz")
    print(f"Channels: {audio.format.channels_per_frame}")
    print(f"Bit Depth: {audio.format.bits_per_channel}")
# --8<-- [end:example]
