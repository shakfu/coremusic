#!/usr/bin/env python3
"""Demonstration of advanced AudioUnit features in coremusic.

This script showcases the new advanced AudioUnit capabilities:
- Stream format configuration (get/set)
- Property helpers (sample_rate, latency, cpu_load, max_frames_per_slice)
- Parameter discovery
- Pythonic property access
"""

import coremusic as cm


def print_section(title):
    """Print a formatted section header"""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main():
    print("=" * 80)
    print("CoreMusic Advanced AudioUnit Features Demo")
    print("=" * 80)

    # ========================================================================
    # 1. Stream Format Configuration
    # ========================================================================
    print_section("1. Stream Format Configuration")

    unit = cm.AudioUnit.default_output()
    try:
        unit.initialize()

        # Get current output stream format
        output_format = unit.get_stream_format('output', 0)
        print("Output Stream Format (Hardware-determined):")
        print(f"  Sample Rate: {output_format.sample_rate} Hz")
        print(f"  Format ID: {output_format.format_id}")
        print(f"  Channels: {output_format.channels_per_frame}")
        print(f"  Bits per Channel: {output_format.bits_per_channel}")
        print(f"  Bytes per Frame: {output_format.bytes_per_frame}")
        print(f"  Bytes per Packet: {output_format.bytes_per_packet}")
        print(f"  Frames per Packet: {output_format.frames_per_packet}")
        print(f"  Format Flags: {output_format.format_flags}")
        print(f"  Is PCM: {output_format.is_pcm}")
        print(f"  Is Stereo: {output_format.is_stereo}")
        print()

        # Try to get input format
        try:
            input_format = unit.get_stream_format('input', 0)
            print("Input Stream Format:")
            print(f"  Sample Rate: {input_format.sample_rate} Hz")
            print(f"  Channels: {input_format.channels_per_frame}")
        except cm.AudioUnitError:
            print("Input Stream Format: Not configured (output-only unit)")

    finally:
        unit.dispose()

    # ========================================================================
    # 2. AudioUnit Properties
    # ========================================================================
    print_section("2. AudioUnit Properties (Pythonic Access)")

    unit = cm.AudioUnit.default_output()
    try:
        unit.initialize()

        # Sample Rate
        print(f"Sample Rate: {unit.sample_rate:.1f} Hz")

        # Latency
        latency_ms = unit.latency * 1000
        print(f"Latency: {latency_ms:.2f} ms ({unit.latency:.6f} seconds)")

        # CPU Load
        cpu_percent = unit.cpu_load * 100
        print(f"CPU Load: {cpu_percent:.2f}%")

        # Max Frames Per Slice
        print(f"Max Frames Per Slice: {unit.max_frames_per_slice} frames")

        # Calculate buffer duration
        if unit.sample_rate > 0 and unit.max_frames_per_slice > 0:
            buffer_ms = (unit.max_frames_per_slice / unit.sample_rate) * 1000
            print(f"Buffer Duration: {buffer_ms:.2f} ms")

    finally:
        unit.dispose()

    # ========================================================================
    # 3. Configuring AudioUnit Before Initialization
    # ========================================================================
    print_section("3. Pre-Initialization Configuration")

    unit = cm.AudioUnit.default_output()
    try:
        print("Setting max_frames_per_slice BEFORE initialization:")
        unit.max_frames_per_slice = 1024
        print(f"  Set to: 1024 frames")

        unit.initialize()

        # Check what the hardware actually set it to
        actual_frames = unit.max_frames_per_slice
        print(f"  Actual value after init: {actual_frames} frames")
        print(f"  (Hardware may adjust based on capabilities)")

    finally:
        unit.dispose()

    # ========================================================================
    # 4. Parameter Discovery
    # ========================================================================
    print_section("4. Parameter Discovery")

    unit = cm.AudioUnit.default_output()
    try:
        unit.initialize()

        # Get parameters for all scopes
        for scope in ['global', 'input', 'output']:
            params = unit.get_parameter_list(scope)
            print(f"{scope.capitalize()} Scope Parameters:")
            if params:
                print(f"  Found {len(params)} parameters: {params}")
            else:
                print(f"  No parameters (typical for default output)")
            print()

    finally:
        unit.dispose()

    # ========================================================================
    # 5. Context Manager with Configuration
    # ========================================================================
    print_section("5. Context Manager Usage")

    print("Creating and configuring AudioUnit with context manager:")
    unit = cm.AudioUnit.default_output()
    unit.max_frames_per_slice = 2048

    print(f"Before context: is_initialized = {unit.is_initialized}")

    with unit:
        print(f"Inside context: is_initialized = {unit.is_initialized}")
        print(f"  Sample Rate: {unit.sample_rate:.1f} Hz")
        print(f"  Max Frames: {unit.max_frames_per_slice}")
        print(f"  Latency: {unit.latency * 1000:.2f} ms")

    print(f"After context: is_disposed = {unit.is_disposed}")

    # ========================================================================
    # 6. Custom Stream Format Configuration
    # ========================================================================
    print_section("6. Creating Custom Audio Formats")

    # Create various format configurations
    formats = [
        cm.AudioFormat(
            sample_rate=44100.0,
            format_id='lpcm',
            format_flags=12,  # Signed integer, packed
            channels_per_frame=2,
            bits_per_channel=16,
            bytes_per_frame=4,
            bytes_per_packet=4,
            frames_per_packet=1
        ),
        cm.AudioFormat(
            sample_rate=48000.0,
            format_id='lpcm',
            format_flags=9,  # Float, packed
            channels_per_frame=2,
            bits_per_channel=32,
            bytes_per_frame=8,
            bytes_per_packet=8,
            frames_per_packet=1
        ),
        cm.AudioFormat(
            sample_rate=96000.0,
            format_id='lpcm',
            format_flags=12,  # Signed integer, packed
            channels_per_frame=6,  # 5.1 surround
            bits_per_channel=24,
            bytes_per_frame=18,
            bytes_per_packet=18,
            frames_per_packet=1
        ),
    ]

    print("Common Audio Format Configurations:")
    for i, fmt in enumerate(formats, 1):
        print(f"\n{i}. {fmt}")
        print(f"   PCM: {fmt.is_pcm}, Stereo: {fmt.is_stereo}, Mono: {fmt.is_mono}")
        if fmt.sample_rate > 0:
            # Calculate data rate
            bytes_per_second = fmt.sample_rate * fmt.bytes_per_frame
            mbps = (bytes_per_second * 8) / (1024 * 1024)
            print(f"   Data Rate: {mbps:.2f} Mbps ({bytes_per_second/1024:.1f} KB/s)")

    # ========================================================================
    # Summary
    # ========================================================================
    print_section("Summary")

    print("Advanced AudioUnit Features:")
    print()
    print("1. Stream Format Control:")
    print("   - get_stream_format(scope, element) -> AudioFormat")
    print("   - set_stream_format(format, scope, element)")
    print("   - Supports 'input', 'output', and 'global' scopes")
    print()
    print("2. Pythonic Property Access:")
    print("   - unit.sample_rate (get/set)")
    print("   - unit.latency (read-only)")
    print("   - unit.cpu_load (read-only)")
    print("   - unit.max_frames_per_slice (get/set)")
    print()
    print("3. Parameter Discovery:")
    print("   - get_parameter_list(scope) -> List[int]")
    print("   - Returns parameter IDs for a given scope")
    print()
    print("4. Configuration Workflow:")
    print("   - Configure before initialization")
    print("   - Hardware may adjust values based on capabilities")
    print("   - Use context managers for automatic cleanup")
    print()
    print("5. AudioFormat Dataclass:")
    print("   - Pythonic representation of AudioStreamBasicDescription")
    print("   - Helper properties (is_pcm, is_stereo, is_mono)")
    print("   - Easy conversion to/from C structures")
    print()

    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
