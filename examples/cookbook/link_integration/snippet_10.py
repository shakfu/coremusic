#!/usr/bin/env python3
"""Map Timeline to Beats."""

# --8<-- [start:example]
from coremusic import link

with link.LinkSession(bpm=120.0) as session:
    state = session.capture_app_session_state()
    current_time = session.clock.micros()

    # Get current beat
    beat = state.beat_at_time(current_time, 4.0)
    print(f"Current beat: {beat:.2f}")

    # Get phase within bar (0.0 - 4.0 for 4/4 time)
    phase = state.phase_at_time(current_time, 4.0)
    print(f"Phase: {phase:.2f}")

    # Calculate time for future beat
    future_beat = beat + 8.0  # 2 bars from now
    future_time = state.time_at_beat(future_beat, 4.0)
    wait_micros = future_time - current_time
    print(f"2 bars from now in {wait_micros / 1000000.0:.2f} seconds")
# --8<-- [end:example]
