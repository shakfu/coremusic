#!/usr/bin/env python3
"""Sync AudioPlayer to Link."""

# --8<-- [start:example]
from coremusic import link
from coremusic.base import AudioPlayer

import time

# Create Link session
with link.LinkSession(bpm=120.0) as session:
    # Create AudioPlayer with Link
    player = AudioPlayer(link_session=session)
    player.load_file("audio.wav")
    player.setup_output()

    # Query Link timing
    timing = player.get_link_timing(quantum=4.0)
    print(f"Beat: {timing['beat']:.2f}")
    print(f"Phase: {timing['phase']:.2f}")
    print(f"Tempo: {timing['tempo']:.1f} BPM")
    print(f"Playing: {timing['is_playing']}")

    # Start playback
    player.play()

    # Monitor sync while playing
    for _ in range(4):
        time.sleep(0.5)
        timing = player.get_link_timing(quantum=4.0)
        print(f"Beat: {timing['beat']:.2f}, Phase: {timing['phase']:.2f}")

    player.stop()
# --8<-- [end:example]
