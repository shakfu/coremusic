#!/usr/bin/env python3
"""Pitch Transformers."""

from coremusic.midi.utilities import MIDISequence

sequence = MIDISequence.load("input.mid")

# --8<-- [start:transpose]
from coremusic.midi.transform import Transpose

# Transpose up an octave
up_octave = Transpose(12).transform(sequence)

# Transpose down a fifth
down_fifth = Transpose(-7).transform(sequence)

# Notes are clamped to valid MIDI range (0-127)
# --8<-- [end:transpose]

# --8<-- [start:invert]
from coremusic.midi.transform import Invert

# Invert around middle C (MIDI 60)
inverted = Invert(pivot=60).transform(sequence)

# Notes above pivot go below, and vice versa
# --8<-- [end:invert]

# --8<-- [start:harmonize]
from coremusic.midi.transform import Harmonize

# Add a major third above each note
thirds = Harmonize([4]).transform(sequence)

# Add third and fifth (triads)
triads = Harmonize([4, 7], velocity_scale=0.7).transform(sequence)

# Add power chord (fifth and octave)
power = Harmonize([7, 12]).transform(sequence)
# --8<-- [end:harmonize]
