#!/usr/bin/env python3
"""LinkSession Class."""

# --8<-- [start:example]
from coremusic import link

# Create session with initial tempo
session = link.LinkSession(bpm=120.0)

# Enable networking (discovers peers)
session.enabled = True

# Enable transport sync
session.start_stop_sync_enabled = True

# Check connections
print(f"Connected to {session.num_peers} peers")

# Access the clock
clock = session.clock

# Cleanup
session.enabled = False
# --8<-- [end:example]
