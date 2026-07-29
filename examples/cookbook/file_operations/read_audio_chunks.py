#!/usr/bin/env python3
"""Read File in Chunks."""


def process_chunk(chunk):
    pass


# --8<-- [start:example]
from coremusic.audio import AudioFile


def read_audio_chunks(filepath, chunk_size=4096):
    """Generator that yields audio chunks."""
    with AudioFile(filepath) as audio:
        total_frames = audio.packet_count
        current = 0

        while current < total_frames:
            to_read = min(chunk_size, total_frames - current)
            data, count = audio.read_packets(current, to_read)
            yield data
            current += count


# Usage
for chunk in read_audio_chunks("audio.wav"):
    process_chunk(chunk)
# --8<-- [end:example]
