#!/usr/bin/env python3
"""Play a file, drawing a progress bar as it goes."""

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_output_devices():
    print("No audio output device available.")
    raise SystemExit(0)

# --8<-- [start:example]
import sys
import time

from coremusic.audio import AudioFile
from coremusic.base import AudioPlayer


def play_with_progress(filepath):
    """Play audio file with progress display."""
    # AudioPlayer does not report duration, so read it from the file
    with AudioFile(filepath) as audio:
        duration = audio.duration

    player = AudioPlayer()
    player.load_file(filepath)
    player.setup_output()

    print(f"Playing: {filepath}")
    print(f"Duration: {duration:.2f}s")

    player.play()

    while player.is_playing():
        progress = player.get_progress()
        current_time = progress * duration

        # Display progress bar
        bar_width = 40
        filled = int(bar_width * progress)
        bar = '=' * filled + '-' * (bar_width - filled)

        sys.stdout.write(f'\r[{bar}] {current_time:.1f}s / {duration:.1f}s')
        sys.stdout.flush()
        time.sleep(0.1)

    print('\nDone!')


play_with_progress("audio.wav")
# --8<-- [end:example]
