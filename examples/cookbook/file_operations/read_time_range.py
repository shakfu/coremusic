#!/usr/bin/env python3
"""Read Specific Section."""

# --8<-- [start:example]
from coremusic.audio import AudioFile

def read_time_range(filepath, start_seconds, duration_seconds):
    """Read specific time range from audio file."""
    with AudioFile(filepath) as audio:
        # Calculate frame positions
        sample_rate = audio.format.sample_rate
        start_frame = int(start_seconds * sample_rate)
        frame_count = int(duration_seconds * sample_rate)

        # Ensure we don't read past the end
        available = max(0, audio.packet_count - start_frame)
        frame_count = min(frame_count, available)
        if frame_count == 0:
            return b""

        # Read data
        data, count = audio.read_packets(start_frame, frame_count)
        return data

# Usage: read 1 second starting at 1 second
data = read_time_range("audio.wav", start_seconds=1.0, duration_seconds=1.0)
# --8<-- [end:example]
