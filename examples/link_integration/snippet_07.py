#!/usr/bin/env python3
"""AudioPlayer with Link."""

# --8<-- [start:example]
import time

from coremusic import link
from coremusic.base import AudioPlayer

# Create Link session
with link.LinkSession(bpm=120.0) as session:
    # Create AudioPlayer with Link integration
    player = AudioPlayer(link_session=session)

    # Load and setup audio
    player.load_file("audio.wav")
    player.setup_output()

    # Query Link timing before playback
    timing = player.get_link_timing(quantum=4.0)
    print(f"Starting at beat {timing['beat']:.2f}")
    print(f"Tempo: {timing['tempo']:.1f} BPM")

    # Start playback
    player.play()

    # Monitor playback with Link timing
    for _ in range(3):
        timing = player.get_link_timing(quantum=4.0)
        progress = player.get_progress()

        print(f"Beat: {timing['beat']:7.2f} | "
              f"Phase: {timing['phase']:4.2f} | "
              f"Progress: {progress*100:5.1f}%", end='\r')

        time.sleep(0.5)

    # Stop playback
    player.stop()
# --8<-- [end:example]
