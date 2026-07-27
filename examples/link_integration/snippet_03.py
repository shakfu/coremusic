#!/usr/bin/env python3
"""Change Tempo."""

# --8<-- [start:example]
from coremusic import link

with link.LinkSession(bpm=120.0) as session:
    # Capture state
    state = session.capture_app_session_state()
    current_time = session.clock.micros()

    # Set new tempo
    state.set_tempo(140.0, current_time)
    session.commit_app_session_state(state)

    print("Tempo changed to 140 BPM")
# --8<-- [end:example]
