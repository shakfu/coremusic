#!/usr/bin/env python3
"""A small command-line music player."""

import sys

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_output_devices():
    print("No audio output device available.")
    raise SystemExit(0)

sys.argv = [sys.argv[0], "audio.wav"]


# --8<-- [start:example]
import sys
import time
from pathlib import Path

from coremusic.audio import AudioFile
from coremusic.base import AudioPlayer


class SimpleMusicPlayer:
    """Simple command-line music player."""

    def __init__(self):
        self.player = AudioPlayer()
        self.duration = 0.0

    def load(self, filepath):
        """Load audio file."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with AudioFile(filepath) as audio:
            self.duration = audio.duration

        self.player.load_file(filepath)
        self.player.setup_output()
        print(f"Loaded: {filepath}")
        print(f"Duration: {self.duration:.2f}s")

    def play(self):
        """Start playback."""
        self.player.play()
        print("Playing...")

    def stop(self):
        """Stop playback."""
        self.player.stop()
        print("Stopped")

    def rewind(self):
        """Return to the start of the file."""
        self.player.reset_playback()

    def get_status(self):
        """Get current playback status."""
        progress = self.player.get_progress()
        return {
            "playing": self.player.is_playing(),
            "progress": progress,
            "current_time": progress * self.duration,
            "duration": self.duration,
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python music_player.py <audio_file>")
        sys.exit(1)

    player = SimpleMusicPlayer()

    try:
        player.load(sys.argv[1])
        player.play()

        # Simple playback loop
        while player.player.is_playing():
            status = player.get_status()
            bar_width = 30
            filled = int(bar_width * status["progress"])
            bar = "=" * filled + "-" * (bar_width - filled)

            sys.stdout.write(
                f"\r[{bar}] {status['current_time']:.1f}s / {status['duration']:.1f}s"
            )
            sys.stdout.flush()
            time.sleep(0.1)

        print("\nPlayback complete!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        player.stop()


if __name__ == "__main__":
    main()
# --8<-- [end:example]
