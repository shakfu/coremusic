#!/usr/bin/env python3
"""Calculate peak and RMS levels.

Usage:
    python peak_rms.py [audio_file]
"""

import sys
import os
import coremusic as cm


def main():
    if not cm.NUMPY_AVAILABLE:
        print("NumPy required for analysis")
        sys.exit(1)

    import numpy as np

    audio_path = sys.argv[1] if len(sys.argv) > 1 else "tests/amen.wav"
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)

    peak = cm.AudioAnalyzer.get_peak_amplitude(audio_path)
    rms = cm.AudioAnalyzer.calculate_rms(audio_path)

    peak_db = 20 * np.log10(peak) if peak > 0 else float("-inf")
    rms_db = 20 * np.log10(rms) if rms > 0 else float("-inf")

    print(f"File: {audio_path}")
    print(f"Peak: {peak:.4f} ({peak_db:.2f} dB)")
    print(f"RMS: {rms:.4f} ({rms_db:.2f} dB)")
    print(f"Crest Factor: {peak/rms:.2f}" if rms > 0 else "Crest Factor: N/A")


if __name__ == "__main__":
    main()
