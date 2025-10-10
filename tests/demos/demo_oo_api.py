#!/usr/bin/env python3
"""Demonstration of the improved object-oriented coremusic API.

This script shows the Pythonic OO layer built on top of the functional API.
"""

import coremusic as cm

def main():
    print("=" * 80)
    print("CoreMusic Object-Oriented API Demo")
    print("=" * 80)
    print()

    # Demonstrate dual API availability
    print("1. Dual API Access (both functional and OO APIs available)")
    print("-" * 80)
    fourcc = cm.fourchar_to_int('WAVE')
    print(f"   Functional API: fourchar_to_int('WAVE') = {fourcc}")
    format_obj = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)
    print(f"   OO API: {format_obj}")
    print()

    # Demonstrate AudioFile with context manager
    print("2. AudioFile with Context Manager (automatic resource management)")
    print("-" * 80)
    audio_path = "tests/amen.wav"

    with cm.AudioFile(audio_path) as audio:
        print(f"   File: {audio_path}")
        print(f"   Status: {audio}")
        print()

        # Access format property (now properly parsed from ASBD)
        format = audio.format
        print(f"   Format Details:")
        print(f"     Sample Rate: {format.sample_rate} Hz")
        print(f"     Format ID: {format.format_id}")
        print(f"     Channels: {format.channels_per_frame}")
        print(f"     Bits per Channel: {format.bits_per_channel}")
        print(f"     Bytes per Frame: {format.bytes_per_frame}")
        print(f"     Is PCM: {format.is_pcm}")
        print(f"     Is Stereo: {format.is_stereo}")
        print()

        # Access duration property (now properly calculated)
        duration = audio.duration
        print(f"   Duration: {duration:.3f} seconds")
        print()

        # Read some packets
        data, packet_count = audio.read_packets(0, 100)
        print(f"   Read {packet_count} packets ({len(data)} bytes)")

    print("   File automatically closed after context exit")
    print()

    # Demonstrate AudioUnit with default output
    print("3. AudioUnit Creation (simplified factory methods)")
    print("-" * 80)
    try:
        unit = cm.AudioUnit.default_output()
        print(f"   Created default output AudioUnit")
        print(f"   Is initialized: {unit.is_initialized}")

        # Clean up
        unit.dispose()
        print(f"   Disposed: {unit.is_disposed}")
    except cm.AudioUnitError as e:
        print(f"   AudioUnit error: {e}")
    print()

    # Demonstrate AudioQueue creation
    print("4. AudioQueue (Pythonic object creation)")
    print("-" * 80)
    try:
        format = cm.AudioFormat(
            sample_rate=44100.0,
            format_id='lpcm',
            channels_per_frame=2,
            bits_per_channel=16
        )
        queue = cm.AudioQueue.new_output(format)
        print(f"   Created AudioQueue with format: {format}")

        # Clean up
        queue.dispose()
        print(f"   Disposed: {queue.is_disposed}")
    except cm.AudioQueueError as e:
        print(f"   AudioQueue error: {e}")
    print()

    # Demonstrate MIDI client
    print("5. MIDI Client (automatic port management)")
    print("-" * 80)
    try:
        client = cm.MIDIClient("Demo Client")
        print(f"   Created MIDI client: {client.name}")

        input_port = client.create_input_port("Input")
        print(f"   Created input port: {input_port.name}")

        output_port = client.create_output_port("Output")
        print(f"   Created output port: {output_port.name}")

        # Cleanup
        client.dispose()
        print(f"   Disposed client and all ports automatically")
    except cm.MIDIError as e:
        print(f"   MIDI error: {e}")
    print()

    # Demonstrate exception hierarchy
    print("6. Exception Hierarchy (Pythonic error handling)")
    print("-" * 80)
    print(f"   CoreAudioError -> base exception")
    print(f"   AudioFileError -> file operations")
    print(f"   AudioQueueError -> queue operations")
    print(f"   AudioUnitError -> audio unit operations")
    print(f"   MIDIError -> MIDI operations")
    print(f"   MusicPlayerError -> music player operations")
    print()

    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print()
    print("Key Improvements:")
    print("  - Context managers for automatic resource cleanup")
    print("  - Pythonic property access (audio.format, audio.duration)")
    print("  - Type-safe AudioFormat dataclass")
    print("  - Factory methods (AudioUnit.default_output())")
    print("  - Proper exception hierarchy")
    print("  - Backward compatible with functional API")
    print()

if __name__ == '__main__':
    main()
