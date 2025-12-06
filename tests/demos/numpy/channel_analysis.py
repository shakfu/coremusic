#!/usr/bin/env python3
"""Analyze individual audio channels.

Usage:
    python channel_analysis.py [audio_file]
"""

import sys
import os
import coremusic as cm


def main():
    if not cm.NUMPY_AVAILABLE:
        print("NumPy not installed. Install with: pip install numpy")
        sys.exit(1)

    import numpy as np

    audio_path = sys.argv[1] if len(sys.argv) > 1 else "tests/amen.wav"
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)

    with cm.AudioFile(audio_path) as audio:
        format = audio.format
        data = audio.read_as_numpy()

        print(f"File: {audio_path}")
        print(f"Channels: {format.channels_per_frame}")

        for ch in range(format.channels_per_frame):
            channel_data = data[:, ch] if format.channels_per_frame > 1 else data
            print(f"\nChannel {ch + 1}:")
            print(f"  Range: [{channel_data.min()}, {channel_data.max()}]")
            print(f"  Mean: {channel_data.mean():.4f}")
            print(f"  Std: {channel_data.std():.4f}")
            print(f"  RMS: {np.sqrt(np.mean(channel_data**2)):.4f}")


if __name__ == "__main__":
    main()
