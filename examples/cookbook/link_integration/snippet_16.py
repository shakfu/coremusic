#!/usr/bin/env python3
"""Timing Precision."""

# --8<-- [start:example]
from coremusic import link

with link.LinkSession(bpm=120.0) as session:
    # Always use clock.micros() for current time
    current_time = session.clock.micros()

    state = session.capture_app_session_state()
    beat = state.beat_at_time(current_time, 4.0)

    # Don't use time.time() - it's not precise enough for audio
# --8<-- [end:example]
