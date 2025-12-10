Music Theory Primitives
=======================

This tutorial covers the music theory primitives available in the
``coremusic.music`` module for working with notes, intervals, scales,
and chords.

Overview
--------

The music module provides:

- **Note**: Musical note with MIDI number, name, octave, and frequency
- **Interval**: Distance between two notes with quality (major, minor, perfect)
- **Scale**: 25+ scale types including modes and exotic scales
- **Chord**: 35+ chord types from triads to extended/altered chords

Music Theory Basics
-------------------

Notes
^^^^^

The ``Note`` class represents a musical note:

.. code-block:: python

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

Intervals
^^^^^^^^^

The ``Interval`` class represents the distance between two notes:

.. code-block:: python

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

Scales
^^^^^^

The ``Scale`` class provides 25+ scale types:

.. code-block:: python

   from coremusic.music.theory import Note, Scale, ScaleType

   root = Note('C', 4)

   # Diatonic scales
   major = Scale(root, ScaleType.MAJOR)
   natural_minor = Scale(root, ScaleType.NATURAL_MINOR)
   harmonic_minor = Scale(root, ScaleType.HARMONIC_MINOR)
   melodic_minor = Scale(root, ScaleType.MELODIC_MINOR)

   # Modes
   dorian = Scale(Note('D', 4), ScaleType.DORIAN)
   phrygian = Scale(Note('E', 4), ScaleType.PHRYGIAN)
   lydian = Scale(Note('F', 4), ScaleType.LYDIAN)
   mixolydian = Scale(Note('G', 4), ScaleType.MIXOLYDIAN)
   locrian = Scale(Note('B', 4), ScaleType.LOCRIAN)

   # Pentatonic and blues
   major_pent = Scale(root, ScaleType.MAJOR_PENTATONIC)
   minor_pent = Scale(root, ScaleType.MINOR_PENTATONIC)
   blues = Scale(root, ScaleType.BLUES_MINOR)

   # Jazz scales
   bebop = Scale(root, ScaleType.BEBOP_DOMINANT)
   whole_tone = Scale(root, ScaleType.WHOLE_TONE)
   diminished = Scale(root, ScaleType.DIMINISHED)

   # World scales
   harmonic_major = Scale(root, ScaleType.HARMONIC_MAJOR)
   double_harmonic = Scale(root, ScaleType.DOUBLE_HARMONIC)
   hungarian_minor = Scale(root, ScaleType.HUNGARIAN_MINOR)

   # Exotic scales
   hirajoshi = Scale(root, ScaleType.HIRAJOSHI)
   in_sen = Scale(root, ScaleType.IN_SEN)

   # Get scale notes
   notes = major.get_notes()
   print([n.name for n in notes])
   # ['C', 'D', 'E', 'F', 'G', 'A', 'B']

   # Get MIDI note numbers
   midi_notes = major.get_midi_notes()
   print(midi_notes)  # [60, 62, 64, 65, 67, 69, 71]

Chords
^^^^^^

The ``Chord`` class provides 35+ chord types:

.. code-block:: python

   from coremusic.music.theory import Note, Chord, ChordType

   root = Note('C', 4)

   # Triads
   major = Chord(root, ChordType.MAJOR)           # C, E, G
   minor = Chord(root, ChordType.MINOR)           # C, Eb, G
   diminished = Chord(root, ChordType.DIMINISHED) # C, Eb, Gb
   augmented = Chord(root, ChordType.AUGMENTED)   # C, E, G#
   sus2 = Chord(root, ChordType.SUS2)             # C, D, G
   sus4 = Chord(root, ChordType.SUS4)             # C, F, G

   # Seventh chords
   dom7 = Chord(root, ChordType.DOMINANT_7)       # C, E, G, Bb
   maj7 = Chord(root, ChordType.MAJOR_7)          # C, E, G, B
   min7 = Chord(root, ChordType.MINOR_7)          # C, Eb, G, Bb
   dim7 = Chord(root, ChordType.DIMINISHED_7)     # C, Eb, Gb, Bbb
   half_dim7 = Chord(root, ChordType.HALF_DIMINISHED_7)  # C, Eb, Gb, Bb
   min_maj7 = Chord(root, ChordType.MINOR_MAJOR_7)       # C, Eb, G, B

   # Extended chords
   dom9 = Chord(root, ChordType.DOMINANT_9)       # C, E, G, Bb, D
   maj9 = Chord(root, ChordType.MAJOR_9)          # C, E, G, B, D
   dom11 = Chord(root, ChordType.DOMINANT_11)     # C, E, G, Bb, D, F
   dom13 = Chord(root, ChordType.DOMINANT_13)     # C, E, G, Bb, D, F, A

   # Altered chords
   dom7b5 = Chord(root, ChordType.DOMINANT_7_FLAT_5)
   dom7s5 = Chord(root, ChordType.DOMINANT_7_SHARP_5)
   dom7b9 = Chord(root, ChordType.DOMINANT_7_FLAT_9)
   dom7s9 = Chord(root, ChordType.DOMINANT_7_SHARP_9)

   # Added tone chords
   add9 = Chord(root, ChordType.ADD_9)            # C, E, G, D
   six = Chord(root, ChordType.SIXTH)             # C, E, G, A

   # Get chord notes
   notes = maj7.get_notes()
   print([n.name for n in notes])
   # ['C', 'E', 'G', 'B']

   # Get MIDI note numbers
   midi_notes = maj7.get_midi_notes()
   print(midi_notes)  # [60, 64, 67, 71]

Key Signatures
^^^^^^^^^^^^^^

Work with key signatures and the circle of fifths:

.. code-block:: python

   from coremusic.music.theory import KEY_SIGNATURES, CIRCLE_OF_FIFTHS

   # Key signatures show sharps/flats
   print(KEY_SIGNATURES['G'])   # (['F#'], False) - 1 sharp, major
   print(KEY_SIGNATURES['Bb'])  # (['Bb', 'Eb'], False) - 2 flats, major
   print(KEY_SIGNATURES['Am'])  # ([], True) - no accidentals, minor

   # Circle of fifths
   print(CIRCLE_OF_FIFTHS)
   # ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']

Utility Functions
^^^^^^^^^^^^^^^^^

Convert between note names and MIDI numbers:

.. code-block:: python

   from coremusic.music.theory import note_name_to_midi, midi_to_note_name

   # Name to MIDI
   midi = note_name_to_midi('C', 4)  # 60
   midi = note_name_to_midi('A', 4)  # 69

   # MIDI to name
   name, octave = midi_to_note_name(60)  # ('C', 4)
   name, octave = midi_to_note_name(69)  # ('A', 4)

See Also
--------

- :doc:`async_audio` - Asynchronous audio processing
- :doc:`audio_file_basics` - Working with audio files
