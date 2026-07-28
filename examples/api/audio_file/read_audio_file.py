#!/usr/bin/env python3
"""Read Entire Audio File."""

# --8<-- [start:example]
from coremusic.audio import AudioFile


def read_audio_file(filepath):
    """Read entire audio file into memory."""
    with AudioFile(filepath) as audio:
        # Get total frame count
        total_frames = audio.packet_count

        # Read all data
        data, count = audio.read_packets(0, total_frames)

        return {
            'data': data,
            'sample_rate': audio.format.sample_rate,
            'channels': audio.format.channels_per_frame,
            'format': audio.format.format_id
        }

# Use the function
audio_data = read_audio_file("audio.wav")
print(f"Loaded {len(audio_data['data'])} bytes")
# --8<-- [end:example]
