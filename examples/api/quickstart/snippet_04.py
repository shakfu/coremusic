#!/usr/bin/env python3
"""Reading with automatic format conversion."""

# --8<-- [start:example]
from coremusic.audio import AudioFormat, ExtendedAudioFile

with ExtendedAudioFile("input.wav") as ext_audio:
    # Set client format for automatic conversion
    ext_audio.client_format = AudioFormat.pcm(sample_rate=48000.0, channels=2)

    # Read converted data
    data, count = ext_audio.read(8192)
# --8<-- [end:example]
