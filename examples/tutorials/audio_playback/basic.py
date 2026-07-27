#!/usr/bin/env python3
"""Play a file and wait for it to finish."""

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_output_devices():
    print("No audio output device available.")
    raise SystemExit(0)

# --8<-- [start:example]
import time

from coremusic.base import AudioPlayer

# Create player and load file
player = AudioPlayer()
player.load_file("audio.wav")
player.setup_output()

# Start playback
player.play()

# Wait for playback to complete
while player.is_playing():
    time.sleep(0.1)

print("Playback complete!")
# --8<-- [end:example]
