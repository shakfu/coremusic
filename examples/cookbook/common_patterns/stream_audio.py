#!/usr/bin/env python3
"""Generator-Based Streaming."""


def process_chunk(chunk):
    pass


# --8<-- [start:example]
from coremusic.audio import AudioFile


def stream_audio(filepath, chunk_size=4096):
    """Stream audio data as a generator."""
    with AudioFile(filepath) as audio:
        total_frames = audio.packet_count
        current = 0

        while current < total_frames:
            to_read = min(chunk_size, total_frames - current)
            data, count = audio.read_packets(current, to_read)

            if count == 0:
                break

            yield data
            current += count


# Usage
for chunk in stream_audio("audio.wav"):
    process_chunk(chunk)
# --8<-- [end:example]
