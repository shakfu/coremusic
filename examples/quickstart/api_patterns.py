#!/usr/bin/env python3
"""The three patterns that recur throughout the API."""

# --8<-- [start:context-manager]
from coremusic.audio import AudioFile

# Automatic resource cleanup
with AudioFile("audio.wav") as audio:
    data = audio.read_packets(0, 1000)
# File automatically closed
# --8<-- [end:context-manager]

# --8<-- [start:error-handling]
from coremusic.audio import AudioFile
from coremusic.exceptions import AudioFileError

try:
    with AudioFile("audio.wav") as audio:
        data = audio.read_packets(0, 1000)
except AudioFileError as e:
    print(f"Audio error: {e}")
except FileNotFoundError:
    print("File not found")
# --8<-- [end:error-handling]

# --8<-- [start:numpy]
from coremusic.audio import NUMPY_AVAILABLE, AudioFile

if NUMPY_AVAILABLE:
    import numpy as np

    with AudioFile("audio.wav") as audio:
        # Read as NumPy array
        data = audio.read_as_numpy()
        print(f"Shape: {data.shape}")
        print(f"Peak: {np.max(np.abs(data))}")
# --8<-- [end:numpy]
