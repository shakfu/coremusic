#!/usr/bin/env python3
"""Show default input and output audio devices.

Usage:
    python default_devices.py
"""

import coremusic as cm


def main():
    output = cm.AudioDeviceManager.get_default_output_device()
    if output:
        print(f"Default Output: {output.name}")
        print(f"  Sample Rate: {output.sample_rate:.0f} Hz")
        config = output.get_stream_configuration("output")
        print(f"  Config: {config}")
    else:
        print("No default output device")

    print()

    input_dev = cm.AudioDeviceManager.get_default_input_device()
    if input_dev:
        print(f"Default Input: {input_dev.name}")
        print(f"  Sample Rate: {input_dev.sample_rate:.0f} Hz")
        config = input_dev.get_stream_configuration("input")
        print(f"  Config: {config}")
    else:
        print("No default input device")


if __name__ == "__main__":
    main()
