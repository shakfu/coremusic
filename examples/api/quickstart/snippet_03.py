#!/usr/bin/env python3
"""Get Audio Format."""

# --8<-- [start:example]
from coremusic.audio import AudioFile

with AudioFile("audio.wav") as audio:
    fmt = audio.format
    print(f"Format ID: {fmt.format_id}")           # 'lpcm'
    print(f"Sample rate: {fmt.sample_rate}")       # 44100.0
    print(f"Channels: {fmt.channels_per_frame}")   # 2
    print(f"Bits: {fmt.bits_per_channel}")         # 16
# --8<-- [end:example]
