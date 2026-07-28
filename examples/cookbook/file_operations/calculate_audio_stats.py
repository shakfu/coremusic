#!/usr/bin/env python3
"""Calculate Audio Statistics."""

# --8<-- [start:example]
import numpy as np

from coremusic.audio import AudioFile


def calculate_audio_stats(filepath):
    """Calculate audio statistics."""
    with AudioFile(filepath) as audio:
        # Read audio data
        data, count = audio.read_packets(0, audio.packet_count)

        # Convert to numpy
        fmt = audio.format
        dtype = np.int16 if fmt.bits_per_channel == 16 else np.int32
        samples = np.frombuffer(data, dtype=dtype)

        # Calculate statistics
        stats = {
            'mean': float(np.mean(samples)),
            'std': float(np.std(samples)),
            'min': int(np.min(samples)),
            'max': int(np.max(samples)),
            'rms': float(np.sqrt(np.mean(samples**2))),
        }

        # Calculate peak amplitude
        max_value = 2**(fmt.bits_per_channel - 1) - 1
        stats['peak_amplitude'] = max(abs(stats['min']), abs(stats['max'])) / max_value

        return stats

# Usage
stats = calculate_audio_stats("audio.wav")
print(f"Peak amplitude: {stats['peak_amplitude']:.2%}")
print(f"RMS: {stats['rms']:.2f}")
# --8<-- [end:example]
