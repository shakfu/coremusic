#!/usr/bin/env python3
"""Convert stereo audio to mono.

Usage:
    python stereo_to_mono.py <input_file> <output_file>
"""

import sys
import os
import coremusic as cm
import coremusic.capi as capi


def main():
    if len(sys.argv) < 3:
        print("Usage: python stereo_to_mono.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        sys.exit(1)

    with cm.AudioFile(input_file) as audio:
        input_format = audio.format
        print(f"Input: {input_format.sample_rate:.0f} Hz, {input_format.channels_per_frame} ch")

    output_format = cm.AudioFormatPresets.wav_44100_mono()
    print(f"Output: {output_format.sample_rate:.0f} Hz, {output_format.channels_per_frame} ch")

    capi.convert_audio_file(input_file, output_file, output_format)
    print(f"Converted: {output_file} ({os.path.getsize(output_file)} bytes)")


if __name__ == "__main__":
    main()
