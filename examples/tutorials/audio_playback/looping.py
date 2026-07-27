#!/usr/bin/env python3
"""Play a file more than once."""

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_output_devices():
    print("No audio output device available.")
    raise SystemExit(0)

# --8<-- [start:manual]
import time

from coremusic.base import AudioPlayer


def play_looped(filepath, num_loops=3):
    """Play audio file multiple times."""
    player = AudioPlayer()
    player.load_file(filepath)
    player.setup_output()

    for i in range(num_loops):
        print(f"Loop {i + 1}/{num_loops}")
        player.play()

        while player.is_playing():
            time.sleep(0.1)

        # Rewind for the next pass
        player.reset_playback()

    print("Looping complete!")


play_looped("audio.wav", num_loops=2)
# --8<-- [end:manual]

# --8<-- [start:builtin]
player = AudioPlayer()
player.load_file("audio.wav")
player.setup_output()

# Let the player loop by itself, and stop it when you have had enough
player.set_looping(True)
player.play()
time.sleep(1.0)
player.stop()
# --8<-- [end:builtin]
