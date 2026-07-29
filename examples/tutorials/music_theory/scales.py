#!/usr/bin/env python3
"""Scales."""

# --8<-- [start:example]
from coremusic.music.theory import Note, Scale, ScaleType

root = Note("C", 4)

# Diatonic scales
major = Scale(root, ScaleType.MAJOR)
natural_minor = Scale(root, ScaleType.NATURAL_MINOR)
harmonic_minor = Scale(root, ScaleType.HARMONIC_MINOR)
melodic_minor = Scale(root, ScaleType.MELODIC_MINOR)

# Modes
dorian = Scale(Note("D", 4), ScaleType.DORIAN)
phrygian = Scale(Note("E", 4), ScaleType.PHRYGIAN)
lydian = Scale(Note("F", 4), ScaleType.LYDIAN)
mixolydian = Scale(Note("G", 4), ScaleType.MIXOLYDIAN)
locrian = Scale(Note("B", 4), ScaleType.LOCRIAN)

# Pentatonic and blues
major_pent = Scale(root, ScaleType.MAJOR_PENTATONIC)
minor_pent = Scale(root, ScaleType.MINOR_PENTATONIC)
blues = Scale(root, ScaleType.BLUES)
major_blues = Scale(root, ScaleType.MAJOR_BLUES)

# Symmetric scales
whole_tone = Scale(root, ScaleType.WHOLE_TONE)
diminished = Scale(root, ScaleType.DIMINISHED)
chromatic = Scale(root, ScaleType.CHROMATIC)

# World scales
double_harmonic = Scale(root, ScaleType.DOUBLE_HARMONIC)
hungarian_minor = Scale(root, ScaleType.HUNGARIAN_MINOR)
persian = Scale(root, ScaleType.PERSIAN)
arabian = Scale(root, ScaleType.ARABIAN)

# Exotic scales
hirajoshi = Scale(root, ScaleType.HIRAJOSHI)
in_sen = Scale(root, ScaleType.IN_SEN)
iwato = Scale(root, ScaleType.IWATO)
balinese = Scale(root, ScaleType.BALINESE)

# Get scale notes
notes = major.get_notes()
print([n.name for n in notes])
# ['C', 'D', 'E', 'F', 'G', 'A', 'B']

# Get MIDI note numbers
midi_notes = major.get_midi_notes()
print(midi_notes)  # [60, 62, 64, 65, 67, 69, 71]
# --8<-- [end:example]
