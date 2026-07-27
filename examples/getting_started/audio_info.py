#!/usr/bin/env python3
"""Print everything worth knowing about an audio file."""

import sys

sys.argv = [sys.argv[0], "audio.wav"]


# --8<-- [start:example]
import sys

from coremusic.audio import AudioFile


def display_audio_info(filepath):
    """Display comprehensive audio file information."""
    with AudioFile(filepath) as audio:
        fmt = audio.format

        print(f"File: {filepath}")
        print(f"Duration: {audio.duration:.2f} seconds")
        print(f"Sample Rate: {fmt.sample_rate} Hz")
        print(f"Channels: {fmt.channels_per_frame}")
        print(f"Bits per Channel: {fmt.bits_per_channel}")
        print(f"Format: {fmt.format_id}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_info.py <audio_file>")
        sys.exit(1)

    display_audio_info(sys.argv[1])
# --8<-- [end:example]
