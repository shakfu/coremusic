#!/usr/bin/env python3
"""Build notes, scales, and chords, and measure an interval."""

# --8<-- [start:example]
from coremusic.music.theory import Chord, ChordType, Note, Scale, ScaleType

# Create notes, scales, and chords
c4 = Note("C", 4)
c_major = Scale(c4, ScaleType.MAJOR)
cmaj7 = Chord(c4, ChordType.MAJOR_7)

# Get scale degrees and chord notes
print(f"C Major scale: {[str(n) for n in c_major.get_notes()]}")
print(f"CMaj7 chord: {[str(n) for n in cmaj7.get_notes()]}")

# Interval analysis
from coremusic.music.theory import Interval

interval = Interval.between(c4, Note("G", 4))
print(f"Interval: {interval.name}")  # Perfect Fifth
# --8<-- [end:example]
