#!/usr/bin/env python3
"""Check Availability."""

# --8<-- [start:example]
from coremusic.audio import NUMPY_AVAILABLE, AudioFile

if NUMPY_AVAILABLE:
    import numpy as np

    with AudioFile("audio.wav") as audio:
        # Get NumPy dtype
        dtype = audio.format.to_numpy_dtype()

        # Read and convert
        data, count = audio.read_packets(0, 1024)
        samples = np.frombuffer(data, dtype=dtype)
# --8<-- [end:example]
