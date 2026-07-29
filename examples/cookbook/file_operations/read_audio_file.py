#!/usr/bin/env python3
"""Read Entire File."""

# --8<-- [start:example]
from coremusic.audio import AudioFile


def read_audio_file(filepath):
    """Read entire audio file into memory."""
    with AudioFile(filepath) as audio:
        data, count = audio.read_packets(0, audio.packet_count)
        return data, audio.format


# Usage
audio_data, format_info = read_audio_file("audio.wav")
print(f"Read {len(audio_data)} bytes")
# --8<-- [end:example]
