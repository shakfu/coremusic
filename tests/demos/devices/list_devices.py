#!/usr/bin/env python3
"""List all audio devices on the system.

Usage:
    python list_devices.py
"""

import coremusic as cm


def main():
    devices = cm.AudioDeviceManager.get_devices()

    print(f"Found {len(devices)} audio device(s)\n")

    for device in devices:
        print(f"{device.name}")
        print(f"  Manufacturer: {device.manufacturer}")
        print(f"  UID:          {device.uid}")
        print(f"  Sample Rate:  {device.sample_rate:.0f} Hz")
        print(f"  Transport:    {device.transport_type}")
        print()


if __name__ == "__main__":
    main()
