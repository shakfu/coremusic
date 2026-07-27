#!/usr/bin/env python3
"""Measuring how much memory an operation holds."""

# --8<-- [start:example]
import tracemalloc

from coremusic.audio import AudioFile


def load_files(paths):
    """Hold several decoded files in memory at once."""
    loaded = []
    for path in paths:
        with AudioFile(path) as audio:
            data, count = audio.read_packets(0, audio.packet_count)
        loaded.append(data)
    return loaded


tracemalloc.start()

files = load_files(["audio.wav", "input.wav", "drums.wav"])

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Held: {current / 1024 / 1024:.1f} MB")
print(f"Peak: {peak / 1024 / 1024:.1f} MB")
# --8<-- [end:example]
