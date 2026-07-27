#!/usr/bin/env python3
"""Measure throughput at different buffer sizes."""

# --8<-- [start:example]
import time

from coremusic.audio import AudioFile


def benchmark_buffer_size(file_path, buffer_size):
    start = time.time()
    total_packets = 0

    with AudioFile(file_path) as audio:
        total = audio.packet_count
        while total_packets < total:
            to_read = min(buffer_size, total - total_packets)
            data, count = audio.read_packets(total_packets, to_read)
            if count == 0:
                break
            total_packets += count

    duration = time.time() - start
    return total_packets / duration / 1_000_000  # Million packets/sec


# Test different buffer sizes
for size in [512, 1024, 2048, 4096, 8192, 16384]:
    throughput = benchmark_buffer_size("audio.wav", size)
    print(f"Buffer {size}: {throughput:.2f} Mpackets/sec")
# --8<-- [end:example]
