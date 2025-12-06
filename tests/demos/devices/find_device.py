#!/usr/bin/env python3
"""Find audio device by name or UID.

Usage:
    python find_device.py <name_or_uid>
"""

import sys
import coremusic as cm


def main():
    if len(sys.argv) < 2:
        print("Usage: python find_device.py <name_or_uid>")
        sys.exit(1)

    query = sys.argv[1]

    # Try by name first
    device = cm.AudioDeviceManager.find_device_by_name(query)
    if device:
        print(f"Found by name: {device.name}")
        print(f"  UID: {device.uid}")
        print(f"  Sample Rate: {device.sample_rate:.0f} Hz")
        return

    # Try by UID
    device = cm.AudioDeviceManager.find_device_by_uid(query)
    if device:
        print(f"Found by UID: {device.name}")
        print(f"  UID: {device.uid}")
        print(f"  Sample Rate: {device.sample_rate:.0f} Hz")
        return

    print(f"Device not found: {query}")
    sys.exit(1)


if __name__ == "__main__":
    main()
