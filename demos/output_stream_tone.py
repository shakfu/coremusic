#!/usr/bin/env python3
"""Play a generated sine tone through a real-time output stream.

Opens the default output device and feeds it a continuously generated sine wave
via a pull generator, using ``coremusic.audio.streaming.AudioOutputStream``. The
generator returns float32 interleaved samples as raw bytes, so this example has
no NumPy dependency.

Usage::

    python demos/output_stream_tone.py                 # 440 Hz for 3 s
    python demos/output_stream_tone.py --freq 220 --duration 5
    python demos/output_stream_tone.py --freq 330 --channels 1
"""

from __future__ import annotations

import argparse
import math
import struct
import time

from coremusic.audio.streaming import AudioOutputStream


def make_sine_generator(freq: float, sample_rate: float, channels: int, gain: float):
    """Return a generator(frame_count) -> bytes producing a continuous sine.

    A running phase kept across calls avoids clicks at buffer boundaries.
    """
    phase = 0.0
    phase_inc = 2.0 * math.pi * freq / sample_rate

    def generate(frame_count: int) -> bytes:
        nonlocal phase
        samples = []
        for _ in range(frame_count):
            value = math.sin(phase) * gain
            phase += phase_inc
            if phase >= 2.0 * math.pi:
                phase -= 2.0 * math.pi
            samples.extend([value] * channels)  # interleaved
        return struct.pack(f"<{len(samples)}f", *samples)

    return generate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--freq", type=float, default=440.0, help="Tone frequency in Hz"
    )
    parser.add_argument("--duration", type=float, default=3.0, help="Seconds to play")
    parser.add_argument("--channels", type=int, default=2, help="Output channels")
    parser.add_argument(
        "--sample-rate", type=float, default=44100.0, help="Sample rate"
    )
    parser.add_argument("--gain", type=float, default=0.3, help="Amplitude 0.0-1.0")
    parser.add_argument(
        "--buffer-size", type=int, default=512, help="Frames per buffer"
    )
    args = parser.parse_args()

    stream = AudioOutputStream(
        channels=args.channels,
        sample_rate=args.sample_rate,
        buffer_size=args.buffer_size,
    )
    stream.set_generator(
        make_sine_generator(args.freq, args.sample_rate, args.channels, args.gain)
    )

    print(
        f"Playing {args.freq:.0f} Hz for {args.duration:.1f} s "
        f"({args.channels} ch @ {args.sample_rate:.0f} Hz)... Ctrl-C to stop."
    )
    try:
        with stream:
            time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:  # noqa: BLE001 - report and exit for a CLI example
        print(f"Could not open the output stream: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
