#!/usr/bin/env python3
"""Transport Control."""

# --8<-- [start:example]
import time

from coremusic import link

with link.LinkSession(bpm=120.0) as session:
    # Enable start/stop sync
    session.start_stop_sync_enabled = True

    # Start transport
    state = session.capture_app_session_state()
    state.set_is_playing(True, session.clock.micros())
    session.commit_app_session_state(state)
    print("Transport started")

    time.sleep(0.5)

    # Stop transport
    state = session.capture_app_session_state()
    state.set_is_playing(False, session.clock.micros())
    session.commit_app_session_state(state)
    print("Transport stopped")
# --8<-- [end:example]
