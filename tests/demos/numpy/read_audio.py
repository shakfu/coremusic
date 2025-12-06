#!/usr/bin/env python3
"""Read audio file as NumPy array.

Usage:
    python read_audio.py [audio_file]
"""

import sys
import os
import coremusic as cm


def main():
    if not cm.NUMPY_AVAILABLE:
        print("NumPy not installed. Install with: pip install numpy")
        sys.exit(1)

    audio_path = sys.argv[1] if len(sys.argv) > 1 else "tests/amen.wav"
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)

    with cm.AudioFile(audio_path) as audio:
        format = audio.format
        data = audio.read_as_numpy()

        print(f"File: {audio_path}")
        print(f"Format: {format.sample_rate:.0f} Hz, {format.channels_per_frame} ch, {format.bits_per_channel} bit")
        print(f"Duration: {audio.duration:.2f}s")
        print(f"Array shape: {data.shape}")
        print(f"Array dtype: {data.dtype}")
        print(f"Data range: [{data.min()}, {data.max()}]")


if __name__ == "__main__":
    main()
