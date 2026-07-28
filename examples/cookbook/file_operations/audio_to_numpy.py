#!/usr/bin/env python3
"""Convert to NumPy Array."""

# --8<-- [start:example]
import numpy as np

from coremusic.audio import AudioFile


def audio_to_numpy(filepath):
    """Convert audio file to NumPy array."""
    with AudioFile(filepath) as audio:
        # Read raw data
        data, count = audio.read_packets(0, audio.packet_count)

        # Get format info
        fmt = audio.format
        dtype = np.int16 if fmt.bits_per_channel == 16 else np.int32

        # Convert to numpy array
        samples = np.frombuffer(data, dtype=dtype)

        # Reshape for channels
        if fmt.channels_per_frame > 1:
            samples = samples.reshape(-1, fmt.channels_per_frame)

        return samples, fmt.sample_rate

# Usage
samples, sample_rate = audio_to_numpy("audio.wav")
print(f"Shape: {samples.shape}, Sample rate: {sample_rate}Hz")
# --8<-- [end:example]
