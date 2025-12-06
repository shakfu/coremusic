#!/usr/bin/env python3
"""Slice audio using onset detection.

Usage:
    python onset_slicing.py [audio_file] [sensitivity]
"""

import sys

try:
    import numpy as np
    from coremusic.audio.slicing import AudioSlicer
except ImportError:
    print("Requires NumPy and SciPy")
    sys.exit(1)


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "tests/amen.wav"
    sensitivity = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    slicer = AudioSlicer(audio_path, method="onset", sensitivity=sensitivity)
    slices = slicer.detect_slices(min_slice_duration=0.05, max_slices=16)

    print(f"Detected {len(slices)} slices (sensitivity={sensitivity})")
    for i, s in enumerate(slices[:8], 1):
        print(f"  {i}. {s.start:.3f}s - {s.end:.3f}s ({s.duration:.3f}s)")
    if len(slices) > 8:
        print(f"  ... and {len(slices) - 8} more")


if __name__ == "__main__":
    main()
