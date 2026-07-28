#!/usr/bin/env python3
"""Process Audio in Chunks."""

# --8<-- [start:example]
from coremusic.audio import AudioFile


def process_audio_chunks(filepath, chunk_size=1024):
    """Process audio file in chunks."""
    with AudioFile(filepath) as audio:
        total_frames = audio.packet_count
        current_frame = 0

        while current_frame < total_frames:
            # Calculate chunk size
            frames_to_read = min(chunk_size, total_frames - current_frame)

            # Read chunk
            data, count = audio.read_packets(current_frame, frames_to_read)

            # Process chunk
            process_audio_data(data)

            current_frame += count

def process_audio_data(data):
    """Process audio data chunk."""
    # Your processing logic here


process_audio_chunks("audio.wav")
# --8<-- [end:example]
