#!/usr/bin/env python3
"""Session Management."""

# --8<-- [start:example]
from coremusic import link

# Good: Automatic cleanup
with link.LinkSession(bpm=120.0) as session:
    # Use session
    pass

# Avoid: Manual management
session = link.LinkSession(bpm=120.0)
try:
    session.enabled = True
    # Use session
finally:
    session.enabled = False
# --8<-- [end:example]
