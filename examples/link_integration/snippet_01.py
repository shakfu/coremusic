#!/usr/bin/env python3
"""Basic Link Session."""

# --8<-- [start:example]
from coremusic import link

# Create Link session with context manager
with link.LinkSession(bpm=120.0) as session:
    print(f"Link enabled: {session.enabled}")
    print(f"Connected peers: {session.num_peers}")

    # Get current state
    state = session.capture_app_session_state()
    print(f"Tempo: {state.tempo:.1f} BPM")
    print(f"Playing: {state.is_playing}")
# --8<-- [end:example]
