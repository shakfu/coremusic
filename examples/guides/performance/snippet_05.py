#!/usr/bin/env python3
"""Reusing buffers instead of allocating per block."""

# --8<-- [start:example]
import numpy as np

from coremusic.audio import AudioFile

buffer_size = 4096

with AudioFile("audio.wav") as audio:
    total = audio.packet_count
    for offset in range(0, total, buffer_size):
        data, count = audio.read_packets(offset, min(buffer_size, total - offset))
        if count == 0:
            break

        # Wrap the bytes rather than copying them
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)

        # Process in place to avoid further copies
        samples *= 0.5  # Example: reduce volume
# --8<-- [end:example]
