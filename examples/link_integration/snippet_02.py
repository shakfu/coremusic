#!/usr/bin/env python3
"""Query Beat Position."""

# --8<-- [start:example]
from coremusic import link

with link.LinkSession(bpm=120.0) as session:
    clock = session.clock

    # Get current beat position
    state = session.capture_app_session_state()
    current_time = clock.micros()
    beat = state.beat_at_time(current_time, quantum=4.0)
    phase = state.phase_at_time(current_time, quantum=4.0)

    print(f"Beat: {beat:.2f}, Phase: {phase:.2f}/4")
# --8<-- [end:example]
