#!/usr/bin/env python3
"""Recombine audio slices with different strategies.

Usage:
    python recombine.py [audio_file]
"""

import sys

try:
    import numpy as np
    from coremusic.audio.slicing import AudioSlicer, SliceCollection, SliceRecombinator
except ImportError:
    print("Requires NumPy and SciPy")
    sys.exit(1)


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "tests/amen.wav"

    slicer = AudioSlicer(audio_path, method="grid")
    slices = slicer.detect_slices(divisions=16)
    collection = SliceCollection(slices)
    recombinator = SliceRecombinator(collection)
    sample_rate = slices[0].sample_rate

    methods = ["original", "reverse", "random"]
    for method in methods:
        audio = recombinator.recombine(method=method, crossfade_duration=0.005)
        duration = len(audio) / sample_rate
        print(f"{method}: {duration:.3f}s")


if __name__ == "__main__":
    main()
