#!/usr/bin/env python3
"""Format Human-Readable Info."""

# --8<-- [start:example]
from coremusic.audio import AudioFile


def format_duration(seconds):
    """Format duration as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def format_file_size(bytes):
    """Format file size as human-readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} TB"

def format_audio_info(filepath):
    """Format audio information for display."""
    from pathlib import Path

    path = Path(filepath)
    with AudioFile(filepath) as audio:
        fmt = audio.format

        info = f"""
File: {path.name}
Size: {format_file_size(path.stat().st_size)}
Duration: {format_duration(audio.duration)}
Format: {fmt.format_id}
Sample Rate: {fmt.sample_rate:,.0f} Hz
Channels: {fmt.channels_per_frame}
Bit Depth: {fmt.bits_per_channel}-bit
Bitrate: {(fmt.sample_rate * fmt.bytes_per_frame * 8 / 1000):,.0f} kbps
"""
        return info.strip()

# Usage
print(format_audio_info("audio.wav"))
# --8<-- [end:example]
