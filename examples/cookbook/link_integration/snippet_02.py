#!/usr/bin/env python3
"""Query Tempo and Beat."""

# --8<-- [start:example]
import time

from coremusic import link

with link.LinkSession(bpm=120.0) as session:
    # Start transport
    state = session.capture_app_session_state()
    state.set_is_playing(True, session.clock.micros())
    session.commit_app_session_state(state)

    # Monitor beat position
    for _i in range(4):
        time.sleep(0.1)
        state = session.capture_app_session_state()
        current_time = session.clock.micros()
        beat = state.beat_at_time(current_time, 4.0)  # 4/4 time
        print(f"Beat: {beat:.2f}, Tempo: {state.tempo:.1f} BPM")
# --8<-- [end:example]
