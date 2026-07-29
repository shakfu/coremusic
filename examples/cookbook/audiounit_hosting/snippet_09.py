#!/usr/bin/env python3
"""Supported Formats."""

# --8<-- [start:example]
from coremusic.audio.audiounit_host import PluginAudioFormat

# Float formats (32-bit and 64-bit)
fmt_f32 = PluginAudioFormat(44100.0, 2, PluginAudioFormat.FLOAT32)
fmt_f64 = PluginAudioFormat(44100.0, 2, PluginAudioFormat.FLOAT64)

# Integer formats (16-bit and 32-bit)
fmt_i16 = PluginAudioFormat(44100.0, 2, PluginAudioFormat.INT16)
fmt_i32 = PluginAudioFormat(44100.0, 2, PluginAudioFormat.INT32)

# Non-interleaved (planar) format
fmt_planar = PluginAudioFormat(
    44100.0,
    2,
    PluginAudioFormat.FLOAT32,
    interleaved=False,  # Separate buffers per channel
)
# --8<-- [end:example]
