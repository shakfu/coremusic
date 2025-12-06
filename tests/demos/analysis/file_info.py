#!/usr/bin/env python3
"""Extract audio file information.

Usage:
    python file_info.py [audio_file]
"""

import sys
import os
import coremusic as cm


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "tests/amen.wav"
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)

    info = cm.AudioAnalyzer.get_file_info(audio_path)

    print(f"File: {info['path']}")
    print(f"Duration: {info['duration']:.2f} seconds")
    print(f"Sample Rate: {info['sample_rate']} Hz")
    print(f"Format: {info['format_id']}")
    print(f"Channels: {info['channels']} ({'stereo' if info['is_stereo'] else 'mono'})")
    print(f"Bits per channel: {info['bits_per_channel']}")

    if cm.NUMPY_AVAILABLE and 'peak_amplitude' in info:
        import numpy as np
        rms_db = 20 * np.log10(info["rms"]) if info["rms"] > 0 else float("-inf")
        print(f"Peak: {info['peak_amplitude']:.4f}")
        print(f"RMS: {info['rms']:.4f} ({rms_db:.2f} dB)")


if __name__ == "__main__":
    main()
