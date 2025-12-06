#!/usr/bin/env python3
"""Show NumPy dtype mappings for audio formats.

Usage:
    python format_dtypes.py
"""

import coremusic as cm


def main():
    if not cm.NUMPY_AVAILABLE:
        print("NumPy not installed. Install with: pip install numpy")
        return

    formats = [
        ("16-bit PCM", cm.AudioFormat(44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16)),
        ("24-bit PCM", cm.AudioFormat(44100.0, "lpcm", channels_per_frame=2, bits_per_channel=24)),
        ("32-bit int PCM", cm.AudioFormat(44100.0, "lpcm", format_flags=0, channels_per_frame=2, bits_per_channel=32)),
        ("32-bit float PCM", cm.AudioFormat(44100.0, "lpcm", format_flags=1, channels_per_frame=2, bits_per_channel=32)),
        ("8-bit signed PCM", cm.AudioFormat(44100.0, "lpcm", format_flags=0, channels_per_frame=1, bits_per_channel=8)),
    ]

    print("Audio Format to NumPy dtype mappings:\n")
    print(f"{'Format':<24} {'NumPy dtype':<12}")
    print("-" * 40)

    for name, format in formats:
        try:
            dtype = format.to_numpy_dtype()
            print(f"{name:<24} {str(dtype):<12}")
        except ValueError as e:
            print(f"{name:<24} Error: {e}")


if __name__ == "__main__":
    main()
