#!/usr/bin/env python3
"""Play a file and report progress while it runs."""

import sys

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_output_devices():
    print("No audio output device available.")
    raise SystemExit(0)

sys.argv = [sys.argv[0], "audio.wav"]


# --8<-- [start:example]
import sys
import time

from coremusic.base import AudioPlayer


def play_audio(filepath):
    """Play an audio file."""
    # Create audio player
    player = AudioPlayer()

    # Load and setup
    player.load_file(filepath)
    player.setup_output()

    # Start playback
    print(f"Playing: {filepath}")
    player.play()

    # Wait for playback to complete
    while player.is_playing():
        progress = player.get_progress()
        print(f"Progress: {progress:.1%}", end="\r")
        time.sleep(0.1)

    print("\nPlayback complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python play_audio.py <audio_file>")
        sys.exit(1)

    play_audio(sys.argv[1])
# --8<-- [end:example]
