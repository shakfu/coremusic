#!/usr/bin/env python3
"""Get File Metadata."""

# --8<-- [start:example]
from coremusic.audio import AudioFile

from pathlib import Path

def get_audio_metadata(filepath):
    """Extract comprehensive audio file metadata."""
    path = Path(filepath)

    with AudioFile(filepath) as audio:
        fmt = audio.format

        metadata = {
            'filename': path.name,
            'file_size': path.stat().st_size,
            'duration': audio.duration,
            'sample_rate': fmt.sample_rate,
            'channels': fmt.channels_per_frame,
            'bit_depth': fmt.bits_per_channel,
            'format_id': fmt.format_id,
            'frame_count': audio.packet_count,
            'bitrate': (fmt.sample_rate * fmt.bytes_per_frame * 8) / 1000
        }

        return metadata

# Usage
metadata = get_audio_metadata("audio.wav")
for key, value in metadata.items():
    print(f"{key}: {value}")
# --8<-- [end:example]
