#!/usr/bin/env python3
"""PCM format flags."""

# --8<-- [start:example]
from coremusic.constants import LinearPCMFormatFlag

# Standard packed float PCM
flags = LinearPCMFormatFlag.IS_FLOAT | LinearPCMFormatFlag.IS_PACKED

# Standard packed signed-integer PCM
int_flags = LinearPCMFormatFlag.IS_SIGNED_INTEGER | LinearPCMFormatFlag.IS_PACKED

# AudioFormat.pcm() builds these for you
from coremusic.audio import AudioFormat

float_format = AudioFormat.pcm(44100.0, channels=2, bits=32, is_float=True)
assert float_format.format_flags == flags
# --8<-- [end:example]
