#!/usr/bin/env python3
"""State Capture and Commit."""

# --8<-- [start:example]
from coremusic import link

with link.LinkSession(bpm=120.0) as session:
    # Capture current state
    state = session.capture_app_session_state()

    # Modify state
    state.set_tempo(140.0, session.clock.micros())
    state.set_is_playing(True, session.clock.micros())

    # Commit changes
    session.commit_app_session_state(state)
# --8<-- [end:example]
