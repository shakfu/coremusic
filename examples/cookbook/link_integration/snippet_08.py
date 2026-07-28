#!/usr/bin/env python3
"""Sync Multiple Applications."""

# --8<-- [start:example]
import time

from coremusic import link

# Create first Link session (e.g., for drums)
with link.LinkSession(bpm=120.0) as session1:
    session1.enabled = True

    # Wait for peer connections
    time.sleep(0.5)
    print(f"Session 1 - Peers: {session1.num_peers}")

    # Create second Link session (e.g., for bass)
    with link.LinkSession(bpm=120.0) as session2:
        session2.enabled = True

        time.sleep(1)
        print(f"Session 1 - Peers: {session1.num_peers}")
        print(f"Session 2 - Peers: {session2.num_peers}")

        # Both sessions are now synchronized
        state1 = session1.capture_app_session_state()
        state2 = session2.capture_app_session_state()

        current_time = session1.clock.micros()
        beat1 = state1.beat_at_time(current_time, 4.0)
        beat2 = state2.beat_at_time(current_time, 4.0)

        print(f"Session 1 beat: {beat1:.2f}")
        print(f"Session 2 beat: {beat2:.2f}")
        print(f"Synchronized: {abs(beat1 - beat2) < 0.01}")
# --8<-- [end:example]
