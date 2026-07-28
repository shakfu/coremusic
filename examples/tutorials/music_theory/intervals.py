#!/usr/bin/env python3
"""Intervals."""

# --8<-- [start:example]
from coremusic.music.theory import Interval

# Standard intervals (semitone values)
print(Interval.UNISON)         # 0 semitones
print(Interval.MINOR_SECOND)   # 1 semitone
print(Interval.MAJOR_SECOND)   # 2 semitones
print(Interval.MINOR_THIRD)    # 3 semitones
print(Interval.MAJOR_THIRD)    # 4 semitones
print(Interval.PERFECT_FOURTH) # 5 semitones
print(Interval.TRITONE)        # 6 semitones
print(Interval.PERFECT_FIFTH)  # 7 semitones
print(Interval.OCTAVE)         # 12 semitones
# --8<-- [end:example]
