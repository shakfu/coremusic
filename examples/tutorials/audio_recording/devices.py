#!/usr/bin/env python3
"""Find input devices, and record from a chosen one."""

# --8<-- [start:list]
from coremusic.audio import AudioDeviceManager


def list_input_devices():
    """List all available input devices."""
    devices = AudioDeviceManager.get_input_devices()

    print("Input Devices:")
    print("-" * 50)

    for device in devices:
        print(f"Name: {device.name}")
        print(f"  UID: {device.uid}")
        print(f"  Channels: {device.channel_count('input')}")
        print(f"  Sample Rate: {device.sample_rate}")
        print()

    return devices


input_devices = list_input_devices()
# --8<-- [end:list]

# --8<-- [start:select]
from coremusic.audio import AudioDeviceManager
from coremusic.capi import AudioRecorder


def record_from_device(device_name, output_path, duration):
    """Record from a specific audio device."""
    device = AudioDeviceManager.find_device_by_name(device_name)

    if device is None or not device.has_input():
        print(f"Device not found: {device_name}")
        return

    print(f"Recording from: {device.name}")

    # Capture follows the default input device, so select it first
    previous = AudioDeviceManager.get_default_input_device()
    AudioDeviceManager.set_default_input_device(device)
    try:
        recorder = AudioRecorder(
            sample_rate=device.sample_rate,
            channels=min(device.channel_count("input"), 2),
        )
        recorder.setup_input(duration=duration)
        recorder.start()
        while recorder.is_recording():
            recorder.run_loop(0.1)
        recorder.stop()
        recorder.save_to_file(output_path)
        print(f"Saved to: {output_path}")
    finally:
        if previous is not None:
            AudioDeviceManager.set_default_input_device(previous)


if input_devices:
    record_from_device(input_devices[0].name, "device_recording.wav", duration=0.5)
# --8<-- [end:select]
