#!/usr/bin/env python3
"""AudioFormat Class."""

# --8<-- [start:example]
from coremusic.audio import AudioFile, AudioFormat

# Access format from audio file
with AudioFile("audio.wav") as audio:
    fmt = audio.format
    print(f"Sample rate: {fmt.sample_rate}Hz")
    print(f"Channels: {fmt.channels_per_frame}")
    print(f"Bit depth: {fmt.bits_per_channel}")

# Create custom format
format = AudioFormat(
    sample_rate=44100.0,
    format_id='lpcm',
    channels_per_frame=2,
    bits_per_channel=16
)
# --8<-- [end:example]
