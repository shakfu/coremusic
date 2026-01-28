#!/usr/bin/env python3
"""Tutorial: Music Theory

This module demonstrates music theory operations with coremusic.
All examples are executable doctests.

Run with: pytest tests/tutorials/test_music_theory.py --doctest-modules -v
"""
from __future__ import annotations


def create_note_from_midi():
    """Create a Note from MIDI note number.

    >>> from coremusic.music.theory import Note
    >>> note = Note.from_midi(60)
    >>> assert note.midi == 60  # Middle C
    >>> note = Note.from_midi(69)
    >>> assert note.midi == 69  # A440
    """
    pass


def note_properties():
    """Access Note properties.

    >>> from coremusic.music.theory import Note
    >>> c4 = Note.from_midi(60)
    >>> # MIDI number
    >>> assert c4.midi == 60
    >>> # Pitch class (0-11, C=0)
    >>> assert c4.pitch_class == 0
    >>> # Octave
    >>> assert c4.octave == 4
    >>> # Frequency (Hz) - C4 is approximately 261.63 Hz
    >>> assert 260 < c4.frequency < 263
    """
    pass


def note_transposition():
    """Transpose notes by semitones.

    >>> from coremusic.music.theory import Note
    >>> c4 = Note.from_midi(60)
    >>> # Transpose up a perfect fifth (7 semitones)
    >>> g4 = c4.transpose(7)
    >>> assert g4.midi == 67
    >>> # Transpose down an octave (-12 semitones)
    >>> c3 = c4.transpose(-12)
    >>> assert c3.midi == 48
    """
    pass


def create_interval():
    """Create and use Interval objects.

    >>> from coremusic.music.theory import Note
    >>> c4 = Note.from_midi(60)
    >>> g4 = Note.from_midi(67)
    >>> # Calculate interval between notes
    >>> interval = c4.interval_to(g4)
    >>> assert interval.semitones == 7
    """
    pass


def interval_semitones():
    """Reference of interval semitones.

    >>> INTERVALS = {
    ...     'unison': 0,
    ...     'minor_2nd': 1,
    ...     'major_2nd': 2,
    ...     'minor_3rd': 3,
    ...     'major_3rd': 4,
    ...     'perfect_4th': 5,
    ...     'tritone': 6,
    ...     'perfect_5th': 7,
    ...     'minor_6th': 8,
    ...     'major_6th': 9,
    ...     'minor_7th': 10,
    ...     'major_7th': 11,
    ...     'octave': 12,
    ... }
    >>> assert INTERVALS['perfect_5th'] == 7
    >>> assert INTERVALS['major_3rd'] == 4
    >>> assert INTERVALS['octave'] == 12
    """
    pass


def create_scale():
    """Create a Scale object.

    >>> from coremusic.music.theory import Note, Scale, ScaleType
    >>> c4 = Note.from_midi(60)
    >>> c_major = Scale(c4, ScaleType.MAJOR)
    >>> assert c_major is not None
    >>> # Get scale notes by iterating
    >>> notes = list(c_major)
    >>> assert len(notes) == 7  # 7 notes in major scale
    >>> # First note should be the root
    >>> assert notes[0].midi == 60
    """
    pass


def scale_types():
    """Reference of scale types.

    >>> from coremusic.music.theory import ScaleType
    >>> # Major and minor scales
    >>> _ = ScaleType.MAJOR
    >>> _ = ScaleType.NATURAL_MINOR
    >>> _ = ScaleType.HARMONIC_MINOR
    >>> _ = ScaleType.MELODIC_MINOR
    >>> # Modes
    >>> _ = ScaleType.DORIAN
    >>> _ = ScaleType.PHRYGIAN
    >>> _ = ScaleType.LYDIAN
    >>> _ = ScaleType.MIXOLYDIAN
    >>> _ = ScaleType.LOCRIAN
    >>> # Pentatonic
    >>> _ = ScaleType.MAJOR_PENTATONIC
    >>> _ = ScaleType.MINOR_PENTATONIC
    >>> # Blues
    >>> _ = ScaleType.BLUES
    """
    pass


def scale_intervals():
    """Scale intervals in semitones.

    >>> SCALE_INTERVALS = {
    ...     'major': [0, 2, 4, 5, 7, 9, 11],
    ...     'natural_minor': [0, 2, 3, 5, 7, 8, 10],
    ...     'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    ...     'dorian': [0, 2, 3, 5, 7, 9, 10],
    ...     'phrygian': [0, 1, 3, 5, 7, 8, 10],
    ...     'lydian': [0, 2, 4, 6, 7, 9, 11],
    ...     'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    ...     'locrian': [0, 1, 3, 5, 6, 8, 10],
    ...     'pentatonic_major': [0, 2, 4, 7, 9],
    ...     'pentatonic_minor': [0, 3, 5, 7, 10],
    ...     'blues': [0, 3, 5, 6, 7, 10],
    ... }
    >>> assert SCALE_INTERVALS['major'] == [0, 2, 4, 5, 7, 9, 11]
    >>> assert len(SCALE_INTERVALS['pentatonic_major']) == 5
    >>> assert len(SCALE_INTERVALS['blues']) == 6
    """
    pass


def create_chord():
    """Create a Chord object.

    >>> from coremusic.music.theory import Note, Chord, ChordType
    >>> c4 = Note.from_midi(60)
    >>> c_major = Chord(c4, ChordType.MAJOR)
    >>> assert c_major is not None
    >>> # Get chord notes by iterating
    >>> notes = list(c_major)
    >>> assert len(notes) >= 3  # At least a triad
    >>> # Root should be first
    >>> assert notes[0].midi == 60
    """
    pass


def chord_types():
    """Reference of chord types.

    >>> from coremusic.music.theory import ChordType
    >>> # Triads
    >>> _ = ChordType.MAJOR
    >>> _ = ChordType.MINOR
    >>> _ = ChordType.DIMINISHED
    >>> _ = ChordType.AUGMENTED
    >>> # Seventh chords
    >>> _ = ChordType.MAJOR_7
    >>> _ = ChordType.MINOR_7
    >>> _ = ChordType.DOMINANT_7
    >>> _ = ChordType.DIMINISHED_7
    >>> _ = ChordType.HALF_DIMINISHED_7
    """
    pass


def chord_intervals():
    """Chord intervals from root.

    >>> CHORD_INTERVALS = {
    ...     'major': [0, 4, 7],              # Root, major 3rd, perfect 5th
    ...     'minor': [0, 3, 7],              # Root, minor 3rd, perfect 5th
    ...     'diminished': [0, 3, 6],         # Root, minor 3rd, diminished 5th
    ...     'augmented': [0, 4, 8],          # Root, major 3rd, augmented 5th
    ...     'major_7': [0, 4, 7, 11],        # Major triad + major 7th
    ...     'minor_7': [0, 3, 7, 10],        # Minor triad + minor 7th
    ...     'dominant_7': [0, 4, 7, 10],     # Major triad + minor 7th
    ...     'diminished_7': [0, 3, 6, 9],    # Diminished triad + diminished 7th
    ...     'half_diminished_7': [0, 3, 6, 10],  # Diminished triad + minor 7th
    ... }
    >>> assert CHORD_INTERVALS['major'] == [0, 4, 7]
    >>> assert CHORD_INTERVALS['minor'] == [0, 3, 7]
    >>> assert len(CHORD_INTERVALS['major_7']) == 4
    """
    pass


def chord_inversions():
    """Calculate chord inversions.

    >>> def get_inversion(chord_notes, inversion):
    ...     '''Get chord inversion (0=root, 1=first, 2=second, etc.)'''
    ...     n = len(chord_notes)
    ...     inv = inversion % n
    ...     return chord_notes[inv:] + [note + 12 for note in chord_notes[:inv]]

    >>> # C major chord (MIDI notes)
    >>> c_major = [60, 64, 67]  # C E G
    >>> # Root position
    >>> get_inversion(c_major, 0)
    [60, 64, 67]
    >>> # First inversion (E G C)
    >>> get_inversion(c_major, 1)
    [64, 67, 72]
    >>> # Second inversion (G C E)
    >>> get_inversion(c_major, 2)
    [67, 72, 76]
    """
    pass


def frequency_from_midi():
    """Calculate frequency from MIDI note number.

    The standard formula: f = 440 * 2^((n-69)/12)
    where n is the MIDI note number and 440 Hz is A4 (MIDI 69).

    >>> import math
    >>> def midi_to_freq(midi_note):
    ...     return 440.0 * (2 ** ((midi_note - 69) / 12.0))

    >>> round(midi_to_freq(69), 2)  # A4 = 440 Hz
    440.0
    >>> round(midi_to_freq(60), 2)  # C4 (Middle C)
    261.63
    >>> round(midi_to_freq(81), 2)  # A5 = 880 Hz
    880.0
    """
    pass


def midi_from_frequency():
    """Calculate MIDI note number from frequency.

    The inverse formula: n = 69 + 12 * log2(f/440)

    >>> import math
    >>> def freq_to_midi(freq):
    ...     return 69 + 12 * math.log2(freq / 440.0)

    >>> round(freq_to_midi(440.0))  # A4
    69
    >>> round(freq_to_midi(261.63))  # C4
    60
    >>> round(freq_to_midi(880.0))  # A5
    81
    """
    pass


def circle_of_fifths():
    """Navigate the circle of fifths.

    >>> # Circle of fifths (starting from C)
    >>> CIRCLE_OF_FIFTHS = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#',
    ...                      'G#', 'D#', 'A#', 'F']
    >>> # Enharmonic equivalents for flats
    >>> FLAT_NAMES = ['C', 'G', 'D', 'A', 'E', 'B', 'Gb', 'Db',
    ...               'Ab', 'Eb', 'Bb', 'F']

    >>> # Each step is a perfect 5th (7 semitones)
    >>> def next_fifth(note_class):
    ...     return (note_class + 7) % 12
    >>> next_fifth(0)  # C -> G
    7
    >>> next_fifth(7)  # G -> D
    2

    >>> # Number of sharps in major key (by position in circle)
    >>> SHARPS_IN_KEY = {
    ...     'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6, 'C#': 7
    ... }
    >>> SHARPS_IN_KEY['G']
    1
    >>> SHARPS_IN_KEY['D']
    2
    """
    pass


def relative_minor():
    """Find relative minor of a major key.

    The relative minor is 3 semitones below the major key root.

    >>> def get_relative_minor(major_root_midi):
    ...     return major_root_midi - 3

    >>> # C major -> A minor
    >>> get_relative_minor(60) == 57  # C4 -> A3
    True

    >>> RELATIVE_MINORS = {
    ...     'C': 'Am', 'G': 'Em', 'D': 'Bm', 'A': 'F#m',
    ...     'E': 'C#m', 'B': 'G#m', 'F#': 'D#m', 'C#': 'A#m',
    ...     'F': 'Dm', 'Bb': 'Gm', 'Eb': 'Cm', 'Ab': 'Fm'
    ... }
    >>> RELATIVE_MINORS['C']
    'Am'
    >>> RELATIVE_MINORS['G']
    'Em'
    """
    pass


def diatonic_chords():
    """Generate diatonic chords from a scale.

    >>> MAJOR_SCALE_CHORDS = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii0']
    >>> MAJOR_CHORD_TYPES = ['major', 'minor', 'minor', 'major', 'major', 'minor', 'diminished']

    >>> # C major diatonic chords
    >>> C_MAJOR_DIATONIC = {
    ...     'I': 'C',      # C major
    ...     'ii': 'Dm',    # D minor
    ...     'iii': 'Em',   # E minor
    ...     'IV': 'F',     # F major
    ...     'V': 'G',      # G major
    ...     'vi': 'Am',    # A minor
    ...     'vii0': 'Bdim' # B diminished
    ... }
    >>> C_MAJOR_DIATONIC['I']
    'C'
    >>> C_MAJOR_DIATONIC['V']
    'G'
    >>> C_MAJOR_DIATONIC['vi']
    'Am'
    """
    pass


# Test runner
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
