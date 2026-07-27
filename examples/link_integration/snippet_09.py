#!/usr/bin/env python3
"""Quantized Playback Start."""

# --8<-- [start:example]
import time

from coremusic import link
from coremusic.base import AudioPlayer

with link.LinkSession(bpm=120.0) as session:
    player = AudioPlayer(link_session=session)
    player.load_file("audio.wav")
    player.setup_output()

    # Get current Link state
    state = session.capture_app_session_state()
    current_time = session.clock.micros()

    # Calculate next bar boundary (4 beats)
    current_beat = state.beat_at_time(current_time, quantum=4.0)
    next_bar = (int(current_beat / 4) + 1) * 4.0

    print(f"Current beat: {current_beat:.2f}")
    print(f"Waiting for beat {next_bar:.0f}...")

    # Wait for next bar
    while True:
        state = session.capture_app_session_state()
        current_time = session.clock.micros()
        beat = state.beat_at_time(current_time, quantum=4.0)

        if beat >= next_bar:
            break

        time.sleep(0.001)

    # Start playback exactly on the bar
    player.play()
    player.start()
    print("Started!")
# --8<-- [end:example]
