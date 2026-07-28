#!/usr/bin/env python3
"""Notes."""

# --8<-- [start:example]
from coremusic.music.theory import Note

# Create notes from name and octave
c4 = Note('C', 4)  # Middle C
fs3 = Note('F#', 3)
bb5 = Note('Bb', 5)

# Note properties
print(f"MIDI: {c4.midi}")           # 60
print(f"Name: {c4.name}")           # C
print(f"Octave: {c4.octave}")       # 4
print(f"Frequency: {c4.frequency:.2f} Hz")  # 261.63 Hz

# Create from MIDI number
a4 = Note.from_midi(69)  # A440

# Transposition
e4 = c4.transpose(4)   # Up 4 semitones -> E4
g3 = c4.transpose(-5)  # Down 5 semitones -> G3
# --8<-- [end:example]
