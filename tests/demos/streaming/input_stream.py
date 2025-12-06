#!/usr/bin/env python3
"""Create audio input stream for capture.

Usage:
    python input_stream.py
"""

from coremusic.audio.streaming import AudioInputStream


def main():
    stream = AudioInputStream(channels=2, sample_rate=44100.0, buffer_size=512)

    print(f"Input Stream Configuration:")
    print(f"  Channels: {stream.channels}")
    print(f"  Sample Rate: {stream.sample_rate} Hz")
    print(f"  Buffer Size: {stream.buffer_size} frames")
    print(f"  Latency: {stream.latency * 1000:.2f} ms")


if __name__ == "__main__":
    main()
