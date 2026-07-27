#!/usr/bin/env python3
"""Tempo-Synced Loops."""

# --8<-- [start:example]
from coremusic import link

import time

with link.LinkSession(bpm=120.0) as session:
    # 4-bar loop
    loop_length_beats = 16.0

    state = session.capture_app_session_state()
    state.set_is_playing(True, session.clock.micros())
    session.commit_app_session_state(state)

    # Monitor loop position
    for _ in range(4):
        time.sleep(0.1)
        state = session.capture_app_session_state()
        current_time = session.clock.micros()

        # Get beat position
        beat = state.beat_at_time(current_time, 4.0)

        # Calculate loop position
        loop_beat = beat % loop_length_beats
        bar = int(loop_beat / 4) + 1
        beat_in_bar = (loop_beat % 4) + 1

        print(f"Bar {bar}, Beat {beat_in_bar:.1f}")
# --8<-- [end:example]
