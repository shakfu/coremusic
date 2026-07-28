#!/usr/bin/env python3
"""Change Tempo."""

# --8<-- [start:example]
import time

from coremusic import link

with link.LinkSession(bpm=120.0) as session:
    state = session.capture_app_session_state()
    state.set_is_playing(True, session.clock.micros())
    session.commit_app_session_state(state)

    # Gradually increase tempo
    for bpm in range(120, 140, 2):
        state = session.capture_app_session_state()
        state.set_tempo(float(bpm), session.clock.micros())
        session.commit_app_session_state(state)
        time.sleep(0.2)
        print(f"Tempo: {bpm} BPM")
# --8<-- [end:example]
