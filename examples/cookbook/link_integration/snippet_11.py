#!/usr/bin/env python3
"""Request Beat Alignment."""

# --8<-- [start:example]
from coremusic import link

with link.LinkSession(bpm=120.0) as session:
    state = session.capture_app_session_state()
    current_time = session.clock.micros()

    # Request that beat 0 occurs now
    state.request_beat_at_time(0.0, current_time, 4.0)
    session.commit_app_session_state(state)
    print("Beat grid aligned to current time")

    # Or align to start of playback
    state = session.capture_app_session_state()
    state.request_beat_at_start_playing_time(0.0, 4.0)
    session.commit_app_session_state(state)
    print("Beat 0 will occur when transport starts")
# --8<-- [end:example]
