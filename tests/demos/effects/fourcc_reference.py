#!/usr/bin/env python3
"""AudioUnit FourCC codes reference.

Usage:
    python fourcc_reference.py
"""


def main():
    print("AudioUnit FourCC Codes Reference\n")

    print("Types:")
    print("  'auou' - Output units")
    print("  'aumu' - Music effects")
    print("  'aufx' - Audio effects")
    print("  'aumi' - Mixer units")
    print("  'aumf' - Music instruments")
    print("  'aufc' - Format converters")

    print("\nOutput ('auou') Subtypes:")
    print("  'def ' - Default output")
    print("  'sys ' - System output")

    print("\nMixer ('aumi') Subtypes:")
    print("  '3dem' - 3D Mixer")
    print("  'mxmx' - Matrix Mixer")
    print("  'mcmx' - Multichannel Mixer")

    print("\nEffects ('aufx') Subtypes:")
    print("  'eqal' - Graphic EQ")
    print("  'dcmp' - Dynamics Processor")
    print("  'dely' - Delay")
    print("  'dist' - Distortion")

    print("\nManufacturer:")
    print("  'appl' - Apple (built-in)")


if __name__ == "__main__":
    main()
