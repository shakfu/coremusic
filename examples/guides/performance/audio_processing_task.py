#!/usr/bin/env python3
"""Profiling a processing loop."""

# --8<-- [start:example]
import cProfile
import pstats

from coremusic.audio import AudioFile


def audio_processing_task():
    with AudioFile("audio.wav") as audio:
        total = audio.packet_count
        for offset in range(0, total, 4096):
            data, count = audio.read_packets(offset, min(4096, total - offset))
            # Process...


# Profile the code
profiler = cProfile.Profile()
profiler.enable()

audio_processing_task()

profiler.disable()
stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
# --8<-- [end:example]
