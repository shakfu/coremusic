#!/usr/bin/env python3
"""Sum several tracks into one."""

# --8<-- [start:example]
import numpy as np

import coremusic.capi as capi
from coremusic.audio import AudioFile, AudioFormat, ExtendedAudioFile


def mix_tracks(track_files, output_path, levels=None):
    """Mix multiple audio tracks with individual levels"""
    if levels is None:
        levels = [1.0] * len(track_files)

    # Load all tracks
    tracks = []
    max_frames = 0
    source_format = None

    for file_path, level in zip(track_files, levels):
        with AudioFile(file_path) as audio:
            samples = audio.read_as_numpy().astype(np.float32) / 32768.0
            samples *= level  # Apply level
            tracks.append(samples)
            max_frames = max(max_frames, len(samples))
            source_format = source_format or audio.format

    # Pad tracks to same length
    for i, track in enumerate(tracks):
        if len(track) < max_frames:
            pad = [(0, max_frames - len(track))] + [(0, 0)] * (track.ndim - 1)
            tracks[i] = np.pad(track, pad)

    # Mix (sum all tracks)
    mixed = np.sum(tracks, axis=0)

    # Normalize to prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak

    out_format = AudioFormat.pcm(
        source_format.sample_rate,
        channels=source_format.channels_per_frame,
        bits=32,
        is_float=True,
    )
    with ExtendedAudioFile.create(
        output_path, capi.fourchar_to_int('WAVE'), out_format
    ) as output_file:
        output_file.write(len(mixed), mixed.tobytes())


# Usage
tracks = ["drums.wav", "input.wav"]
levels = [1.0, 0.8]
mix_tracks(tracks, "mixed.wav", levels=levels)
# --8<-- [end:example]
