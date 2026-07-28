#!/usr/bin/env python3
"""Show the round-trip latency of each real-time buffer size.

Buffer size is the one setting that trades latency against the risk of
dropouts: the smaller the buffer, the sooner audio comes out, and the less
time your callback has to produce it. This prints what each choice costs at a
given sample rate, which is the number to reason about before picking one.

Usage::

    python demos/stream_latency.py
    python demos/stream_latency.py --sample-rate 48000
    python demos/stream_latency.py --measure     # open each stream for real
"""

from __future__ import annotations

import argparse
import sys

# Buffer sizes worth considering, with what each is usually chosen for.
CONFIGS = [
    (64, "Ultra-low (guitar/vocals)"),
    (128, "Very low (live monitoring)"),
    (256, "Low (real-time effects)"),
    (512, "Balanced (general use)"),
    (1024, "Higher (less CPU)"),
    (2048, "High (background)"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sample-rate", type=float, default=44100.0, help="Sample rate in Hz"
    )
    parser.add_argument(
        "--channels", type=int, default=2, help="Channel count for the stream"
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Open a real loopback per size and read back its latency, rather "
        "than computing it (needs an input and an output device)",
    )
    args = parser.parse_args()

    print(f"{'Buffer':<8} {'Latency':<12} {'Use case'}")
    print("-" * 50)

    for buffer_size, use_case in CONFIGS:
        if args.measure:
            from coremusic.audio.streaming import create_loopback

            loopback = create_loopback(
                channels=args.channels,
                sample_rate=args.sample_rate,
                buffer_size=buffer_size,
            )
            latency_ms = loopback.latency * 1000
        else:
            # One buffer period, which is what DirectLoopback.latency reports
            # once it is open. A full input-process-output chain
            # (AudioProcessor) is three of these.
            latency_ms = (buffer_size / args.sample_rate) * 1000

        print(f"{buffer_size:<8} {latency_ms:<12.2f} {use_case}")

    print(f"\nOne-way, at {args.sample_rate:.0f} Hz. A full input -> process ->")
    print("output chain costs three of these. Under ~10ms round trip is")
    print("imperceptible for live monitoring; over ~20ms is audible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
