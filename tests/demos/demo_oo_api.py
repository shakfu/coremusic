#!/usr/bin/env python3
"""
Demo of CoreMusic Object-Oriented API
Shows AudioDevice discovery and NumPy integration features
"""

import coremusic as cm
import os


def demo_audio_devices():
    """Demonstrate AudioDevice and AudioDeviceManager functionality"""
    print("=" * 70)
    print("AUDIO DEVICE DISCOVERY & MANAGEMENT")
    print("=" * 70)

    # Get all audio devices
    print("\n1. All Available Audio Devices:")
    print("-" * 70)
    devices = cm.AudioDeviceManager.get_devices()
    print(f"Found {len(devices)} audio device(s)")

    for i, device in enumerate(devices, 1):
        print(f"\n  Device {i}:")
        print(f"    Name:         {device.name}")
        print(f"    Manufacturer: {device.manufacturer}")
        print(f"    UID:          {device.uid}")
        print(f"    Sample Rate:  {device.sample_rate:.0f} Hz")
        print(f"    Is Alive:     {device.is_alive}")
        print(f"    Is Hidden:    {device.is_hidden}")
        print(f"    Transport:    {device.transport_type}")

    # Get default output device
    print("\n2. Default Output Device:")
    print("-" * 70)
    output_device = cm.AudioDeviceManager.get_default_output_device()
    if output_device:
        print(f"  Name:         {output_device.name}")
        print(f"  Sample Rate:  {output_device.sample_rate:.0f} Hz")

        # Get stream configuration
        output_config = output_device.get_stream_configuration("output")
        print(f"  Output Config: {output_config}")
    else:
        print("  No default output device found")

    # Get default input device
    print("\n3. Default Input Device:")
    print("-" * 70)
    input_device = cm.AudioDeviceManager.get_default_input_device()
    if input_device:
        print(f"  Name:         {input_device.name}")
        print(f"  Sample Rate:  {input_device.sample_rate:.0f} Hz")

        # Get stream configuration
        input_config = input_device.get_stream_configuration("input")
        print(f"  Input Config:  {input_config}")
    else:
        print("  No default input device found")

    # Device search
    print("\n4. Device Search:")
    print("-" * 70)
    if output_device:
        # Search by name
        device_name = output_device.name
        found = cm.AudioDeviceManager.find_device_by_name(device_name)
        print(f"  Search by name '{device_name}': {'Found' if found else 'Not found'}")

        # Search by UID
        device_uid = output_device.uid
        if device_uid:
            found = cm.AudioDeviceManager.find_device_by_uid(device_uid)
            print(
                f"  Search by UID '{device_uid}': {'Found' if found else 'Not found'}"
            )


def demo_numpy_integration():
    """Demonstrate NumPy integration with AudioFile"""
    print("\n" + "=" * 70)
    print("NUMPY INTEGRATION")
    print("=" * 70)

    # Check if NumPy is available
    print(f"\n1. NumPy Available: {cm.NUMPY_AVAILABLE}")

    if not cm.NUMPY_AVAILABLE:
        print("   NumPy not installed. Install with: pip install numpy")
        return

    import numpy as np

    # Find test audio file
    test_file = os.path.join(os.path.dirname(__file__), "tests", "amen.wav")
    if not os.path.exists(test_file):
        print(f"\n   Test audio file not found: {test_file}")
        return

    print(f"\n2. Opening Audio File:")
    print("-" * 70)
    print(f"   File: {test_file}")

    with cm.AudioFile(test_file) as audio:
        # Get format information
        format = audio.format
        print(f"\n   Format Information:")
        print(f"     Sample Rate:   {format.sample_rate:.0f} Hz")
        print(f"     Format ID:     {format.format_id}")
        print(f"     Channels:      {format.channels_per_frame}")
        print(f"     Bits/Channel:  {format.bits_per_channel}")
        print(f"     Is PCM:        {format.is_pcm}")
        print(f"     Is Stereo:     {format.is_stereo}")

        # Get NumPy dtype for this format
        dtype = format.to_numpy_dtype()
        print(f"     NumPy dtype:   {dtype}")

        # Read audio as NumPy array
        print(f"\n3. Reading Audio Data as NumPy Array:")
        print("-" * 70)

        # Read first 1000 frames
        data = audio.read_as_numpy(start_packet=0, packet_count=1000)
        print(f"   Array shape:   {data.shape} (frames, channels)")
        print(f"   Array dtype:   {data.dtype}")
        print(f"   Data range:    [{data.min()}, {data.max()}]")
        print(f"   Mean value:    {data.mean():.2f}")
        print(f"   Std deviation: {data.std():.2f}")

        # Channel analysis
        print(f"\n4. Channel Analysis:")
        print("-" * 70)
        if format.is_stereo:
            left_channel = data[:, 0]
            right_channel = data[:, 1]

            print(f"   Left Channel:")
            print(f"     Range: [{left_channel.min()}, {left_channel.max()}]")
            print(f"     Mean:  {left_channel.mean():.2f}")
            print(f"     Std:   {left_channel.std():.2f}")

            print(f"   Right Channel:")
            print(f"     Range: [{right_channel.min()}, {right_channel.max()}]")
            print(f"     Mean:  {right_channel.mean():.2f}")
            print(f"     Std:   {right_channel.std():.2f}")

        # Read entire file
        print(f"\n5. Reading Entire File:")
        print("-" * 70)
        full_data = audio.read_as_numpy()
        duration = audio.duration

        print(f"   Duration:      {duration:.2f} seconds")
        print(f"   Total frames:  {full_data.shape[0]}")
        print(f"   Total samples: {full_data.size}")
        print(f"   Memory size:   {full_data.nbytes / 1024:.2f} KB")

        # Audio statistics
        print(f"\n6. Audio Statistics:")
        print("-" * 70)
        print(f"   Peak amplitude:    {abs(full_data).max()}")
        print(f"   RMS:               {np.sqrt(np.mean(full_data**2)):.2f}")
        print(f"   Zero crossings:    {np.sum(np.diff(np.sign(full_data[:, 0])) != 0)}")


def demo_audio_format():
    """Demonstrate AudioFormat NumPy dtype conversion"""
    print("\n" + "=" * 70)
    print("AUDIO FORMAT NUMPY DTYPE CONVERSION")
    print("=" * 70)

    if not cm.NUMPY_AVAILABLE:
        print("\n  NumPy not available")
        return

    import numpy as np

    formats = [
        (
            "16-bit PCM",
            cm.AudioFormat(44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16),
        ),
        (
            "24-bit PCM",
            cm.AudioFormat(44100.0, "lpcm", channels_per_frame=2, bits_per_channel=24),
        ),
        (
            "32-bit int PCM",
            cm.AudioFormat(
                44100.0,
                "lpcm",
                format_flags=0,
                channels_per_frame=2,
                bits_per_channel=32,
            ),
        ),
        (
            "32-bit float PCM",
            cm.AudioFormat(
                44100.0,
                "lpcm",
                format_flags=1,
                channels_per_frame=2,
                bits_per_channel=32,
            ),
        ),
        (
            "8-bit signed PCM",
            cm.AudioFormat(
                44100.0,
                "lpcm",
                format_flags=0,
                channels_per_frame=1,
                bits_per_channel=8,
            ),
        ),
    ]

    print("\n  Format Type              NumPy dtype")
    print("  " + "-" * 50)
    for name, format in formats:
        try:
            dtype = format.to_numpy_dtype()
            print(f"  {name:24} {dtype}")
        except ValueError as e:
            print(f"  {name:24} Error: {e}")


def demo_complete_workflow():
    """Demonstrate complete audio workflow combining all features"""
    print("\n" + "=" * 70)
    print("COMPLETE WORKFLOW: DEVICE INFO + AUDIO ANALYSIS")
    print("=" * 70)

    if not cm.NUMPY_AVAILABLE:
        print("\n  NumPy not available - skipping workflow demo")
        return

    import numpy as np

    # Get audio hardware info
    output_device = cm.AudioDeviceManager.get_default_output_device()
    if not output_device:
        print("\n  No output device available")
        return

    print(f"\n  Target Output Device: {output_device.name}")
    print(f"  Device Sample Rate:   {output_device.sample_rate:.0f} Hz")

    # Load audio file
    test_file = os.path.join(os.path.dirname(__file__), "tests", "amen.wav")
    if not os.path.exists(test_file):
        print(f"  Audio file not found: {test_file}")
        return

    with cm.AudioFile(test_file) as audio:
        audio_format = audio.format
        print(
            f"  Audio File Format:    {audio_format.sample_rate:.0f} Hz, {audio_format.channels_per_frame} channels"
        )

        # Check compatibility
        if audio_format.sample_rate != output_device.sample_rate:
            print(f"  ⚠ Sample rate mismatch - resampling would be needed")
        else:
            print(f"  ✓ Sample rates match")

        # Analyze audio content
        data = audio.read_as_numpy(packet_count=5000)
        print(f"\n  Audio Analysis (first {data.shape[0]} frames):")
        print(f"    Peak level:     {abs(data).max()}")
        print(f"    Average level:  {abs(data).mean():.2f}")
        print(f"    Dynamic range:  {data.max() - data.min()}")


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("COREMUSIC OBJECT-ORIENTED API DEMO")
    print("Python bindings for CoreAudio on macOS")
    print("=" * 70)

    try:
        # Demo 1: Audio Device Discovery
        demo_audio_devices()

        # Demo 2: NumPy Integration
        demo_numpy_integration()

        # Demo 3: AudioFormat NumPy dtype conversion
        demo_audio_format()

        # Demo 4: Complete workflow
        demo_complete_workflow()

        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print()

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
