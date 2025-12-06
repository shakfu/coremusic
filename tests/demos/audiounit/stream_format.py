#!/usr/bin/env python3
"""Show AudioUnit stream format configuration.

Usage:
    python stream_format.py
"""

import coremusic as cm


def main():
    unit = cm.AudioUnit.default_output()
    try:
        unit.initialize()

        format = unit.get_stream_format("output", 0)
        print(f"Output Stream Format:")
        print(f"  Sample Rate: {format.sample_rate} Hz")
        print(f"  Channels: {format.channels_per_frame}")
        print(f"  Bits: {format.bits_per_channel}")
        print(f"  Format: {format.format_id}")
        print(f"  Is PCM: {format.is_pcm}")

        print(f"\nUnit Properties:")
        print(f"  Sample Rate: {unit.sample_rate:.0f} Hz")
        print(f"  Latency: {unit.latency * 1000:.2f} ms")
        print(f"  Max Frames: {unit.max_frames_per_slice}")
    finally:
        unit.dispose()


if __name__ == "__main__":
    main()
