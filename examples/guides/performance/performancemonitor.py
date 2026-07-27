#!/usr/bin/env python3
"""A small timing and allocation monitor."""

# --8<-- [start:example]
import time
import tracemalloc

from coremusic.audio import AudioFile


class PerformanceMonitor:
    """Report elapsed time and allocated memory since construction."""

    def __init__(self):
        tracemalloc.start()
        self.start_time = time.perf_counter()
        self.start_memory, _ = tracemalloc.get_traced_memory()

    def report(self, label):
        elapsed = time.perf_counter() - self.start_time
        current, peak = tracemalloc.get_traced_memory()

        print(f"{label}:")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Memory: {current / 1024 / 1024:.1f} MB "
              f"(+{(current - self.start_memory) / 1024 / 1024:.1f} MB)")
        print(f"  Peak: {peak / 1024 / 1024:.1f} MB")


# Usage
monitor = PerformanceMonitor()

with AudioFile("audio.wav") as audio:
    data, count = audio.read_packets(0, audio.packet_count)

monitor.report("After reading audio")
# --8<-- [end:example]
