#!/usr/bin/env python3
"""Real-Time Beat Monitoring."""

# --8<-- [start:example]
from coremusic import link
from coremusic.base import AudioPlayer

import time

with link.LinkSession(bpm=120.0) as session:
    player = AudioPlayer(link_session=session)
    player.load_file("audio.wav")
    player.setup_output()
    player.play()
    player.start()

    # Monitor beats during playback
    while player.is_playing():
        timing = player.get_link_timing(quantum=4.0)
        beat = timing['beat']

        # Visual beat indicator
        indicator = "●" if int(beat) % 4 == 0 else "○"

        print(f"{indicator} Beat: {beat:7.2f} | "
              f"Tempo: {timing['tempo']:6.1f} BPM", end='\r')

        time.sleep(0.1)

    player.stop()
# --8<-- [end:example]
