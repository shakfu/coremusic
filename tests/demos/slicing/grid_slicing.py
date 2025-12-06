#!/usr/bin/env python3
"""Slice audio into equal grid divisions.

Usage:
    python grid_slicing.py [audio_file] [divisions]
"""

import sys

try:
    from coremusic.audio.slicing import AudioSlicer
except ImportError:
    print("Requires NumPy and SciPy")
    sys.exit(1)


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "tests/amen.wav"
    divisions = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    slicer = AudioSlicer(audio_path, method="grid")
    slices = slicer.detect_slices(divisions=divisions)

    print(f"Created {len(slices)} equal slices")
    for i, s in enumerate(slices, 1):
        print(f"  {i}. {s.start:.3f}s - {s.end:.3f}s ({s.duration:.3f}s)")


if __name__ == "__main__":
    main()
