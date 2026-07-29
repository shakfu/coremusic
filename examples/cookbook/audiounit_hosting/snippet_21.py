#!/usr/bin/env python3
"""Performance."""

total_frames = 4096
audio_data = bytes(total_frames * 2 * 4)

# --8<-- [start:example]
from coremusic.audio.audiounit_host import AudioUnitPlugin, PluginAudioFormat

with AudioUnitPlugin.from_name("AUMatrixReverb") as plugin:
    # 1. Set format once, not per-process call
    fmt = PluginAudioFormat(44100.0, 2, PluginAudioFormat.FLOAT32)
    plugin.set_audio_format(fmt)

    # 2. Process in chunks (1024-4096 frames typical)
    chunk_size = 2048

    # 3. Pre-allocate buffers when possible
    for i in range(0, total_frames, chunk_size):
        frames_to_process = min(chunk_size, total_frames - i)
        output = plugin.process(
            audio_data[i : i + frames_to_process],
            num_frames=frames_to_process,
            audio_format=fmt,
        )
# --8<-- [end:example]
