#!/usr/bin/env python3
"""Key Signatures."""

# --8<-- [start:example]
from coremusic.music.theory import CIRCLE_OF_FIFTHS, KEY_SIGNATURES

# Key signatures show sharps/flats, and whether the key is minor
print(KEY_SIGNATURES['G'])   # (['F#'], False) - 1 sharp, major
print(KEY_SIGNATURES['Bb'])  # (['Bb', 'Eb'], False) - 2 flats, major
print(KEY_SIGNATURES['Am'])  # ([], True) - no accidentals, minor

# Circle of fifths
print(CIRCLE_OF_FIFTHS)
# ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']
# --8<-- [end:example]
