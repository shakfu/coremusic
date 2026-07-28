#!/usr/bin/env python3
"""Render a file through an effect and write the result."""

# --8<-- [start:example]
import numpy as np

from coremusic import capi
from coremusic.audio import AudioFile, AudioFormat, ExtendedAudioFile
from coremusic.audio.audiounit_host import AudioUnitPlugin


def process_audio_with_effect(input_path, output_path, effect_name):
    """Process an audio file through an effect, block by block."""
    with AudioFile(input_path) as audio:
        samples = audio.read_as_numpy().astype(np.float32) / 32768.0
        channels = audio.format.channels_per_frame
        sample_rate = audio.format.sample_rate

    block_frames = 512
    processed_blocks = []

    with AudioUnitPlugin.from_name(effect_name, component_type="aufx") as plugin:
        for start in range(0, len(samples), block_frames):
            block = samples[start:start + block_frames]
            if len(block) < block_frames:
                # The plugin wants full blocks; pad the tail
                block = np.pad(block, ((0, block_frames - len(block)), (0, 0)))

            out = plugin.process(block.tobytes(), num_frames=block_frames)
            processed_blocks.append(np.frombuffer(out, dtype=np.float32))

    result = np.concatenate(processed_blocks)

    out_format = AudioFormat.pcm(
        sample_rate, channels=channels, bits=32, is_float=True
    )
    with ExtendedAudioFile.create(
        output_path, capi.fourchar_to_int('WAVE'), out_format
    ) as output:
        output.write(len(result) // channels, result.tobytes())

    print(f"Processed {input_path} -> {output_path}")


process_audio_with_effect("input.wav", "processed.wav", "AUDelay")
# --8<-- [end:example]

# --8<-- [start:cli]
# The CLI does the same thing in one line:
#   coremusic plugin process AUDelay input.wav -o processed.wav
# --8<-- [end:cli]
