#!/usr/bin/env python3
"""Note Name Conversion."""

# --8<-- [start:example]
from coremusic.music.theory import midi_to_note_name, note_name_to_midi

# Name to MIDI
midi = note_name_to_midi("C", 4)  # 60
midi = note_name_to_midi("A", 4)  # 69

# MIDI to name, as a single string with the octave
name = midi_to_note_name(60)  # 'C4'
name = midi_to_note_name(69)  # 'A4'

# Note carries the two apart
from coremusic.music.theory import Note

note = Note.from_midi(60)
print(note.name, note.octave)  # C 4
# --8<-- [end:example]
