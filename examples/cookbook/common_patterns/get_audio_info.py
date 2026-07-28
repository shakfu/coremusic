#!/usr/bin/env python3
"""Simple Cache."""

# --8<-- [start:example]
from functools import lru_cache

from coremusic.audio import AudioFile


@lru_cache(maxsize=100)
def get_audio_info(filepath):
    """Get cached audio file information."""
    with AudioFile(filepath) as audio:
        return {
            'duration': audio.duration,
            'sample_rate': audio.format.sample_rate,
            'channels': audio.format.channels_per_frame,
            'frame_count': audio.packet_count
        }

# First call: reads file
info1 = get_audio_info("audio.wav")

# Second call: returns cached result
info2 = get_audio_info("audio.wav")
# --8<-- [end:example]
