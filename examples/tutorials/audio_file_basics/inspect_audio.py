#!/usr/bin/env python3
"""A complete audio file inspector."""

import sys

sys.argv = [sys.argv[0], "audio.wav"]


# --8<-- [start:example]
import sys
from pathlib import Path

from coremusic.audio import AudioFile
from coremusic.exceptions import AudioFileError


def format_bytes(num_bytes):
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def inspect_audio_file(filepath):
    """Comprehensive audio file inspection."""
    # Check file exists
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}")
        return

    print(f"Inspecting: {filepath}")
    print(f"File size: {format_bytes(path.stat().st_size)}")
    print()

    try:
        with AudioFile(filepath) as audio:
            # Format information
            fmt = audio.format
            print("Format Information:")
            print(f"  Format ID: {fmt.format_id}")
            print(f"  Sample Rate: {fmt.sample_rate} Hz")
            print(f"  Channels: {fmt.channels_per_frame}")
            print(f"  Bit Depth: {fmt.bits_per_channel}")
            print(f"  Bytes/Frame: {fmt.bytes_per_frame}")
            print()

            # Duration information
            print("Duration Information:")
            print(f"  Total Packets: {audio.packet_count:,}")
            print(f"  Duration: {audio.duration:.2f} seconds")
            print(f"  Duration: {audio.duration / 60:.2f} minutes")
            print()

            # Quality classification
            print("Classification:")
            if fmt.sample_rate == 44100 and fmt.bits_per_channel == 16:
                quality = "CD Quality"
            elif fmt.sample_rate >= 96000:
                quality = "Hi-Res Audio"
            elif fmt.sample_rate >= 48000:
                quality = "Professional Audio"
            else:
                quality = "Standard Audio"
            print(f"  Quality: {quality}")

            channel_type = {
                1: "Mono",
                2: "Stereo",
                4: "Quadraphonic",
                6: "5.1 Surround",
                8: "7.1 Surround",
            }.get(fmt.channels_per_frame, f"{fmt.channels_per_frame}-channel")
            print(f"  Channel Type: {channel_type}")

            # Calculate bitrate
            bitrate = (fmt.sample_rate * fmt.bytes_per_frame * 8) / 1000
            print(f"  Bitrate: {bitrate:.0f} kbps")

    except AudioFileError as e:
        print(f"Error opening file: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_audio.py <audio_file>")
        sys.exit(1)

    inspect_audio_file(sys.argv[1])
# --8<-- [end:example]
