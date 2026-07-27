#!/usr/bin/env python3
"""Custom Audio Formats."""

input_data = bytes(1024 * 2 * 2)  # 1024 stereo frames, int16
# --8<-- [start:example]
from coremusic.audio.audiounit_host import AudioUnitPlugin, PluginAudioFormat

with AudioUnitPlugin.from_name("AUDelay") as plugin:
    # Create custom audio format (16-bit integer, 48kHz)
    fmt = PluginAudioFormat(
        sample_rate=48000.0,
        channels=2,
        sample_format=PluginAudioFormat.INT16,
        interleaved=True
    )

    # Set plugin to use this format
    plugin.set_audio_format(fmt)

    # Process audio (automatic conversion to/from float32 internally)
    output = plugin.process(input_data, num_frames=1024, audio_format=fmt)
# --8<-- [end:example]
