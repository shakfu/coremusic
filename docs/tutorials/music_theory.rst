Music Theory and Generative Algorithms
======================================

This tutorial covers the music theory primitives and generative algorithms
available in the ``coremusic.music`` module for algorithmic composition
and MIDI generation.

Overview
--------

The music module provides:

- **Music Theory Primitives**: Note, Interval, Scale, Chord, ChordProgression
- **Generative Algorithms**: Arpeggiator, Euclidean, Markov, Probabilistic, Sequence, Melody, Polyrhythm
- **MIDI Integration**: Direct output to MIDITrack and MIDISequence

Music Theory Basics
-------------------

Notes
^^^^^

The ``Note`` class represents a musical note with MIDI number, name, and frequency:

.. code-block:: python

   from coremusic.music.theory import Note

   # Create notes from MIDI numbers
   c4 = Note(60)  # Middle C
   a4 = Note(69)  # A440

   # Create from note names
   c4 = Note.from_name("C4")
   fs3 = Note.from_name("F#3")
   bb5 = Note.from_name("Bb5")

   # Note properties
   print(f"MIDI: {c4.midi}")           # 60
   print(f"Name: {c4.name}")           # C4
   print(f"Frequency: {c4.frequency:.2f} Hz")  # 261.63 Hz

   # Transposition
   e4 = c4.transpose(4)  # Up 4 semitones
   g3 = c4.transpose(-5) # Down 5 semitones

Intervals
^^^^^^^^^

The ``Interval`` class represents the distance between two notes:

.. code-block:: python

   from coremusic.music.theory import Interval

   # Standard intervals
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

   root = Note.from_name("C4")

   # Diatonic scales
   major = Scale(root, ScaleType.MAJOR)
   natural_minor = Scale(root, ScaleType.NATURAL_MINOR)
   harmonic_minor = Scale(root, ScaleType.HARMONIC_MINOR)
   melodic_minor = Scale(root, ScaleType.MELODIC_MINOR)

   # Modes
   dorian = Scale(Note.from_name("D4"), ScaleType.DORIAN)
   phrygian = Scale(Note.from_name("E4"), ScaleType.PHRYGIAN)
   lydian = Scale(Note.from_name("F4"), ScaleType.LYDIAN)
   mixolydian = Scale(Note.from_name("G4"), ScaleType.MIXOLYDIAN)
   locrian = Scale(Note.from_name("B4"), ScaleType.LOCRIAN)

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
   print([str(n) for n in major.notes])
   # ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4']

   # Check if a note is in the scale
   e4 = Note.from_name("E4")
   print(e4 in major)  # True

Chords
^^^^^^

The ``Chord`` class provides 35+ chord types:

.. code-block:: python

   from coremusic.music.theory import Note, Chord, ChordType

   root = Note.from_name("C4")

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
   print([str(n) for n in maj7.notes])
   # ['C4', 'E4', 'G4', 'B4']

Chord Progressions
^^^^^^^^^^^^^^^^^^

The ``ChordProgression`` class supports Roman numeral notation:

.. code-block:: python

   from coremusic.music.theory import ChordProgression

   # Common progressions
   pop = ChordProgression.from_roman("C", ["I", "V", "vi", "IV"])
   jazz_251 = ChordProgression.from_roman("C", ["ii7", "V7", "Imaj7"])
   blues = ChordProgression.from_roman("A", ["I7", "I7", "I7", "I7",
                                              "IV7", "IV7", "I7", "I7",
                                              "V7", "IV7", "I7", "V7"])

   # Iterate through chords
   for chord in pop.chords:
       print(f"{chord.root.name}: {[str(n) for n in chord.notes]}")

Generative Algorithms
---------------------

Arpeggiator
^^^^^^^^^^^

The ``Arpeggiator`` generates arpeggiated patterns from chord notes:

.. code-block:: python

   from coremusic.music.theory import Note, Chord, ChordType
   from coremusic.music.generative import Arpeggiator, ArpPattern

   # Create a chord
   chord = Chord(Note.from_name("C4"), ChordType.MAJOR_7)

   # Arpeggiator patterns
   # UP, DOWN, UP_DOWN, DOWN_UP, RANDOM, RANDOM_WALK,
   # OUTSIDE_IN, INSIDE_OUT, CHORD, AS_PLAYED

   # Basic up pattern
   arp = Arpeggiator(
       notes=chord.notes,
       pattern=ArpPattern.UP,
       note_duration=0.25,
       velocity=100
   )
   events = arp.generate(num_notes=16)

   # Up-down with swing
   arp_swing = Arpeggiator(
       notes=chord.notes,
       pattern=ArpPattern.UP_DOWN,
       note_duration=0.25,
       swing=0.3  # 0.0 = no swing, 1.0 = max swing
   )

   # Random with seed for reproducibility
   arp_random = Arpeggiator(
       notes=chord.notes,
       pattern=ArpPattern.RANDOM,
       note_duration=0.125,
       seed=42
   )

Euclidean Rhythms
^^^^^^^^^^^^^^^^^

The ``EuclideanGenerator`` creates rhythmic patterns using Bjorklund's algorithm:

.. code-block:: python

   from coremusic.music.theory import Note
   from coremusic.music.generative import EuclideanGenerator

   # Classic Euclidean patterns
   # Tresillo (3,8) - Cuban rhythm
   tresillo = EuclideanGenerator(
       pulses=3, steps=8,
       note=Note.from_name("C4"),
       step_duration=0.125
   )

   # Cinquillo (5,8) - Habanera rhythm
   cinquillo = EuclideanGenerator(
       pulses=5, steps=8,
       note=Note.from_name("C4"),
       step_duration=0.125
   )

   # Rumba (7,12)
   rumba = EuclideanGenerator(
       pulses=7, steps=12,
       note=Note.from_name("C4"),
       step_duration=0.125
   )

   events = cinquillo.generate(num_cycles=4)

Markov Chain Generator
^^^^^^^^^^^^^^^^^^^^^^

The ``MarkovGenerator`` creates probabilistic note sequences:

.. code-block:: python

   from coremusic.music.theory import Note, Scale, ScaleType
   from coremusic.music.generative import MarkovGenerator

   # Create and train Markov generator
   markov = MarkovGenerator(order=2)

   # Train on a melody (MIDI note numbers)
   melody = [60, 62, 64, 65, 67, 65, 64, 62, 60]
   markov.train(melody)

   # Optionally constrain to a scale
   scale = Scale(Note.from_name("C4"), ScaleType.MAJOR)
   markov.set_scale_constraint(scale)

   # Generate new melody
   events = markov.generate(
       num_notes=16,
       start_note=Note.from_name("C4"),
       note_duration=0.25
   )

Probabilistic Generator
^^^^^^^^^^^^^^^^^^^^^^^

The ``ProbabilisticGenerator`` uses weighted random note selection:

.. code-block:: python

   from coremusic.music.theory import Note, Scale, ScaleType
   from coremusic.music.generative import ProbabilisticGenerator

   # Create from scale
   scale = Scale(Note.from_name("D4"), ScaleType.DORIAN)
   prob = ProbabilisticGenerator(
       notes=scale.notes,
       note_duration=0.25
   )

   # Custom weights (emphasize root and fifth)
   weights = {0: 3.0, 4: 2.0}  # Index 0 and 4 weighted higher
   prob.set_weights(weights)

   # Add rest probability
   prob.set_rest_probability(0.1)  # 10% chance of rest

   events = prob.generate(num_notes=32)

Step Sequencer
^^^^^^^^^^^^^^

The ``SequenceGenerator`` provides a classic step sequencer:

.. code-block:: python

   from coremusic.music.theory import Note
   from coremusic.music.generative import SequenceGenerator

   # Create 16-step sequencer
   seq = SequenceGenerator(num_steps=16, step_duration=0.125)

   # Program a drum pattern
   kick = Note(36)   # MIDI note for kick drum
   snare = Note(38)  # MIDI note for snare
   hihat = Note(42)  # MIDI note for hi-hat

   # Four-on-the-floor kick pattern
   for i in [0, 4, 8, 12]:
       seq.set_step(i, kick, velocity=100)

   # Snare on 2 and 4
   for i in [4, 12]:
       seq.set_step(i, snare, velocity=90)

   # Hi-hats on every step
   for i in range(16):
       seq.set_step(i, hihat, velocity=60, probability=0.8)

   events = seq.generate(num_cycles=4)

Melody Generator
^^^^^^^^^^^^^^^^

The ``MelodyGenerator`` creates rule-based melodic phrases:

.. code-block:: python

   from coremusic.music.theory import Note, Scale, ScaleType
   from coremusic.music.generative import MelodyGenerator

   scale = Scale(Note.from_name("C4"), ScaleType.MAJOR)

   melody = MelodyGenerator(
       scale=scale,
       note_duration=0.25,
       max_jump=3,         # Maximum interval jump in scale degrees
       rest_probability=0.1
   )

   events = melody.generate(
       num_notes=16,
       start_note=Note.from_name("C4")
   )

Polyrhythm Generator
^^^^^^^^^^^^^^^^^^^^

The ``PolyrhythmGenerator`` creates layered polyrhythmic patterns:

.. code-block:: python

   from coremusic.music.theory import Note
   from coremusic.music.generative import PolyrhythmGenerator

   # Create 3-against-4 polyrhythm
   poly = PolyrhythmGenerator(cycle_duration=1.0)

   # Layer 1: 3 notes per cycle
   poly.add_layer(
       divisions=3,
       note=Note.from_name("C4"),
       velocity=100
   )

   # Layer 2: 4 notes per cycle
   poly.add_layer(
       divisions=4,
       note=Note.from_name("E4"),
       velocity=80
   )

   events = poly.generate(num_cycles=4)

   # Complex polyrhythm: 5:4:3
   complex_poly = PolyrhythmGenerator(cycle_duration=2.0)
   complex_poly.add_layer(5, Note.from_name("C4"), velocity=100)
   complex_poly.add_layer(4, Note.from_name("E4"), velocity=90)
   complex_poly.add_layer(3, Note.from_name("G4"), velocity=80)

MIDI Export
-----------

All generators output ``MIDIEvent`` objects compatible with the MIDI utilities:

.. code-block:: python

   from coremusic.music.theory import Note, Chord, ChordType
   from coremusic.music.generative import Arpeggiator, ArpPattern
   from coremusic.midi.utilities import MIDISequence, MIDITrack, MIDIStatus

   # Generate arpeggio events
   chord = Chord(Note.from_name("C4"), ChordType.MAJOR)
   arp = Arpeggiator(chord.notes, pattern=ArpPattern.UP_DOWN)
   events = arp.generate(num_notes=16)

   # Create MIDI sequence
   sequence = MIDISequence(ticks_per_beat=480)
   track = sequence.create_track("Arpeggio")

   # Convert events to notes
   note_ons = {}
   for event in sorted(events, key=lambda e: e.time):
       if event.status == MIDIStatus.NOTE_ON and event.data2 > 0:
           key = (event.data1, event.channel)
           note_ons[key] = event
       elif event.status == MIDIStatus.NOTE_OFF or \
            (event.status == MIDIStatus.NOTE_ON and event.data2 == 0):
           key = (event.data1, event.channel)
           if key in note_ons:
               on_event = note_ons.pop(key)
               duration = event.time - on_event.time
               track.add_note(
                   on_event.time,
                   on_event.data1,
                   on_event.data2,
                   duration
               )

   # Save to file
   sequence.save("arpeggio.mid")

Combining Generators
--------------------

Use the ``combine_generators`` utility to layer multiple patterns:

.. code-block:: python

   from coremusic.music.theory import Note, Chord, ChordType, Scale, ScaleType
   from coremusic.music.generative import (
       Arpeggiator, ArpPattern, EuclideanGenerator,
       MelodyGenerator, combine_generators
   )

   # Create multiple generators
   chord = Chord(Note.from_name("C4"), ChordType.MAJOR)
   arp = Arpeggiator(chord.notes, ArpPattern.UP, note_duration=0.25)

   kick = EuclideanGenerator(3, 8, Note(36), step_duration=0.125)
   hihat = EuclideanGenerator(5, 8, Note(42), step_duration=0.125)

   scale = Scale(Note.from_name("C5"), ScaleType.MAJOR)
   melody = MelodyGenerator(scale, note_duration=0.5)

   # Combine and generate
   all_events = combine_generators([
       (arp, 16),      # 16 arp notes
       (kick, 4),      # 4 kick cycles
       (hihat, 4),     # 4 hihat cycles
       (melody, 8),    # 8 melody notes
   ])

See Also
--------

- :doc:`async_audio` - Asynchronous audio processing
- :doc:`audio_file_basics` - Working with audio files
- ``coremusic.midi.utilities`` - MIDI file I/O
