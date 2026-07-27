#!/usr/bin/env python3
"""Memory-mapped reads."""

# --8<-- [start:example]
from coremusic.audio import MMapAudioFile

with MMapAudioFile("audio.wav") as mapped:
    # Fast random access without decoding the whole file
    chunk = mapped.read_frames(1000, 1000)  # 1000 frames from frame 1000

    # NumPy view over the mapped bytes
    audio_np = mapped.read_as_numpy(start_frame=0, num_frames=44100)
    print(audio_np.shape, mapped.frame_count)
# --8<-- [end:example]
