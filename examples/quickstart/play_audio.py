#!/usr/bin/env python3
"""Play an audio file, with the one-call helper and with AudioPlayer."""

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_output_devices():
    print("No audio output device available.")
    raise SystemExit(0)

# --8<-- [start:shortcut]
from coremusic.shortcuts import play

play("audio.wav")  # blocks until the file finishes
# --8<-- [end:shortcut]

# --8<-- [start:player]
import time

from coremusic.base import AudioPlayer

player = AudioPlayer()
player.load_file("audio.wav")
player.setup_output()
player.play()

while player.is_playing():
    time.sleep(0.1)

player.stop()
# --8<-- [end:player]
