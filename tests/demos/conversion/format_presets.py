#!/usr/bin/env python3
"""Show available audio format presets.

Usage:
    python format_presets.py
"""

from coremusic.audio.utilities import AudioFormatPresets


def main():
    presets = [
        ("CD Quality WAV", AudioFormatPresets.wav_44100_stereo()),
        ("WAV 44.1kHz mono", AudioFormatPresets.wav_44100_mono()),
        ("Pro Audio WAV", AudioFormatPresets.wav_48000_stereo()),
        ("High-Res WAV", AudioFormatPresets.wav_96000_stereo()),
    ]

    print("Available Format Presets:\n")
    for name, format in presets:
        print(f"{name}:")
        print(f"  Sample Rate: {format.sample_rate:.0f} Hz")
        print(f"  Channels: {format.channels_per_frame}")
        print(f"  Bits: {format.bits_per_channel}")
        print()


if __name__ == "__main__":
    main()
