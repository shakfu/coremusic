#!/usr/bin/env python3
"""Normalize a file to a target peak."""

# --8<-- [start:example]
import numpy as np

from coremusic import capi
from coremusic.audio import AudioFile, AudioFormat, ExtendedAudioFile


def normalize_audio(input_path, output_path, target_peak=0.9):
    """Normalize audio to target peak level"""
    with AudioFile(input_path) as input_file:
        # Read the whole file, as float in [-1, 1]
        samples = input_file.read_as_numpy().astype(np.float32) / 32768.0
        source_format = input_file.format

        # Find current peak
        current_peak = np.max(np.abs(samples))

        # Calculate and apply gain
        if current_peak > 0:
            gain = target_peak / current_peak
            samples *= gain
            print(f"Applied gain: {gain:.3f}x ({20 * np.log10(gain):.2f}dB)")

    # Write output as float32, which is what the samples now are
    out_format = AudioFormat.pcm(
        source_format.sample_rate,
        channels=source_format.channels_per_frame,
        bits=32,
        is_float=True,
    )
    with ExtendedAudioFile.create(
        output_path, capi.fourchar_to_int('WAVE'), out_format
    ) as output_file:
        output_file.write(len(samples), samples.tobytes())


# Usage
normalize_audio("input.wav", "normalized.wav", target_peak=0.9)
# --8<-- [end:example]
