#!/usr/bin/env python3
"""Choose an output device, and set its volume."""

# --8<-- [start:list]
from coremusic.audio import AudioDeviceManager


def list_output_devices():
    """List available output devices."""
    devices = AudioDeviceManager.get_output_devices()

    print("Output Devices:")
    for device in devices:
        default = AudioDeviceManager.get_default_output_device()
        marker = " (default)" if default and device.uid == default.uid else ""
        print(f"  {device.name} [{device.uid}]{marker}")

    return devices


list_output_devices()
# --8<-- [end:list]

# --8<-- [start:select]
from coremusic.audio import AudioDeviceManager


def play_to_device(filepath, device_name):
    """Send playback to a specific device by making it the default."""
    device = AudioDeviceManager.find_device_by_name(device_name)

    if device is None or not device.has_output():
        print(f"Device not found: {device_name}")
        return

    print(f"Playing to: {device.name}")

    previous = AudioDeviceManager.get_default_output_device()
    AudioDeviceManager.set_default_output_device(device)
    try:
        from coremusic.shortcuts import play

        play(filepath)
    finally:
        if previous is not None:
            AudioDeviceManager.set_default_output_device(previous)
# --8<-- [end:select]

# --8<-- [start:volume]
from coremusic.audio import AudioDeviceManager

device = AudioDeviceManager.get_default_output_device()

# Not every device exposes a software volume control; get_volume() returns
# None when it does not.
level = device.get_volume() if device else None
if level is not None:
    print(f"Output volume: {level:.2f}")
    device.set_volume(level)
else:
    print("This device has no software volume control.")
# --8<-- [end:volume]
