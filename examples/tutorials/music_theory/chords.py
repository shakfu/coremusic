#!/usr/bin/env python3
"""Chords."""

# --8<-- [start:example]
from coremusic.music.theory import Chord, ChordType, Note

root = Note("C", 4)

# Triads
major = Chord(root, ChordType.MAJOR)  # C, E, G
minor = Chord(root, ChordType.MINOR)  # C, Eb, G
diminished = Chord(root, ChordType.DIMINISHED)  # C, Eb, Gb
augmented = Chord(root, ChordType.AUGMENTED)  # C, E, G#
sus2 = Chord(root, ChordType.SUS2)  # C, D, G
sus4 = Chord(root, ChordType.SUS4)  # C, F, G

# Seventh chords
dom7 = Chord(root, ChordType.DOMINANT_7)  # C, E, G, Bb
maj7 = Chord(root, ChordType.MAJOR_7)  # C, E, G, B
min7 = Chord(root, ChordType.MINOR_7)  # C, Eb, G, Bb
dim7 = Chord(root, ChordType.DIMINISHED_7)  # C, Eb, Gb, Bbb
half_dim7 = Chord(root, ChordType.HALF_DIMINISHED_7)  # C, Eb, Gb, Bb
min_maj7 = Chord(root, ChordType.MINOR_MAJOR_7)  # C, Eb, G, B

# Extended chords
dom9 = Chord(root, ChordType.DOMINANT_9)  # C, E, G, Bb, D
maj9 = Chord(root, ChordType.MAJOR_9)  # C, E, G, B, D
dom11 = Chord(root, ChordType.DOMINANT_11)  # C, E, G, Bb, D, F
dom13 = Chord(root, ChordType.DOMINANT_13)  # C, E, G, Bb, D, F, A

# Altered chords
dom7b5 = Chord(root, ChordType.DOMINANT_7_FLAT_5)
dom7b9 = Chord(root, ChordType.DOMINANT_7_FLAT_9)
dom7s9 = Chord(root, ChordType.DOMINANT_7_SHARP_9)

# Added tone chords
add9 = Chord(root, ChordType.ADD9)  # C, E, G, D
six = Chord(root, ChordType.MAJOR_6)  # C, E, G, A

# Get chord notes
notes = maj7.get_notes()
print([n.name for n in notes])
# ['C', 'E', 'G', 'B']

# Get MIDI note numbers
midi_notes = maj7.get_midi_notes()
print(midi_notes)  # [60, 64, 67, 71]
# --8<-- [end:example]
