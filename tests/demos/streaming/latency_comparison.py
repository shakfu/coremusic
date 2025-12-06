#!/usr/bin/env python3
"""Compare latency for different buffer sizes.

Usage:
    python latency_comparison.py
"""

from coremusic.audio.streaming import create_loopback


def main():
    configs = [
        (64, "Ultra-low (guitar/vocals)"),
        (128, "Very low (live monitoring)"),
        (256, "Low (real-time effects)"),
        (512, "Balanced (general use)"),
        (1024, "Higher (less CPU)"),
        (2048, "High (background)"),
    ]

    print(f"{'Buffer':<8} {'Latency':<12} {'Use Case'}")
    print("-" * 50)

    for buffer_size, use_case in configs:
        processor = create_loopback(buffer_size=buffer_size)
        latency_ms = processor.latency * 1000
        print(f"{buffer_size:<8} {latency_ms:<12.2f} {use_case}")


if __name__ == "__main__":
    main()
