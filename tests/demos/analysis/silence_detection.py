#!/usr/bin/env python3
"""Detect silence regions in audio file.

Usage:
    python silence_detection.py [audio_file] [threshold_db] [min_duration]
"""

import sys
import os
import coremusic as cm


def main():
    if not cm.NUMPY_AVAILABLE:
        print("NumPy required for silence detection")
        sys.exit(1)

    audio_path = sys.argv[1] if len(sys.argv) > 1 else "tests/amen.wav"
    threshold_db = float(sys.argv[2]) if len(sys.argv) > 2 else -40.0
    min_duration = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1

    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)

    print(f"File: {audio_path}")
    print(f"Threshold: {threshold_db} dB, Min duration: {min_duration}s\n")

    silence_regions = cm.AudioAnalyzer.detect_silence(
        audio_path, threshold_db=threshold_db, min_duration=min_duration
    )

    if silence_regions:
        print(f"Found {len(silence_regions)} silence region(s):")
        for i, (start, end) in enumerate(silence_regions, 1):
            print(f"  {i}. {start:.2f}s - {end:.2f}s ({end - start:.2f}s)")
    else:
        print("No silence regions found")


if __name__ == "__main__":
    main()
