#!/usr/bin/env python3
"""Cleaning up MIDI sequencing objects."""

# --8<-- [start:example]
from coremusic.midi import MusicPlayer, MusicSequence

# Risky: a raised exception leaves both objects undisposed
player = MusicPlayer()
sequence = MusicSequence()
player.dispose()
sequence.dispose()

# Better: MusicPlayer is a context manager; dispose the sequence in a finally
sequence = MusicSequence()
try:
    with MusicPlayer() as player:
        player.sequence = sequence
finally:
    sequence.dispose()
# --8<-- [end:example]
