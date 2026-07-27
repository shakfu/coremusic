#!/usr/bin/env python3
"""Fade a file in and out."""

# --8<-- [start:example]
import numpy as np

import coremusic.capi as capi
from coremusic.audio import AudioFile, AudioFormat, ExtendedAudioFile


def apply_fades(input_path, output_path, fade_in=0.5, fade_out=0.5):
    """Apply fade in and fade out to audio"""
    with AudioFile(input_path) as input_file:
        samples = input_file.read_as_numpy().astype(np.float32) / 32768.0
        sample_rate = input_file.format.sample_rate
        channels = input_file.format.channels_per_frame

    # read_as_numpy returns (frames, channels) for multichannel files
    audio = samples if samples.ndim == 2 else samples.reshape(-1, 1)

    fade_in_frames = min(int(fade_in * sample_rate), len(audio))
    fade_out_frames = min(int(fade_out * sample_rate), len(audio))

    # Apply fades
    audio[:fade_in_frames] *= np.linspace(0, 1, fade_in_frames)[:, np.newaxis]
    audio[len(audio) - fade_out_frames:] *= (
        np.linspace(1, 0, fade_out_frames)[:, np.newaxis]
    )

    out_format = AudioFormat.pcm(
        sample_rate, channels=channels, bits=32, is_float=True
    )
    with ExtendedAudioFile.create(
        output_path, capi.fourchar_to_int('WAVE'), out_format
    ) as output_file:
        output_file.write(len(audio), audio.tobytes())


# Usage
apply_fades("input.wav", "faded.wav", fade_in=0.5, fade_out=0.5)
# --8<-- [end:example]
