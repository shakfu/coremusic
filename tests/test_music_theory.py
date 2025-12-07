#!/usr/bin/env python3
"""Tests for music theory module."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from coremusic.music.theory import (
    Note,
    Interval,
    IntervalQuality,
    Scale,
    ScaleType,
    Chord,
    ChordType,
    ChordProgression,
    Mode,
    KEY_SIGNATURES,
    CIRCLE_OF_FIFTHS,
    note_name_to_midi,
    midi_to_note_name,
)


class TestNoteNameConversion:
    """Tests for note name conversion utilities."""

    def test_note_name_to_midi_c4(self):
        """Test C4 (middle C) conversion."""
        assert note_name_to_midi('C', 4) == 60

    def test_note_name_to_midi_a4(self):
        """Test A4 (440Hz reference) conversion."""
        assert note_name_to_midi('A', 4) == 69

    def test_note_name_to_midi_with_sharp(self):
        """Test sharp note conversion."""
        assert note_name_to_midi('C#', 4) == 61
        assert note_name_to_midi('F#', 4) == 66

    def test_note_name_to_midi_with_flat(self):
        """Test flat note conversion (enharmonic)."""
        assert note_name_to_midi('Db', 4) == 61
        assert note_name_to_midi('Bb', 4) == 70

    def test_note_name_to_midi_with_octave_in_name(self):
        """Test parsing octave from name string."""
        assert note_name_to_midi('C4') == 60
        assert note_name_to_midi('A3') == 57
        assert note_name_to_midi('F#5') == 78

    def test_note_name_to_midi_extreme_octaves(self):
        """Test extreme octave values."""
        assert note_name_to_midi('C', -1) == 0
        assert note_name_to_midi('G', 9) == 127

    def test_note_name_to_midi_invalid(self):
        """Test invalid note names."""
        with pytest.raises(ValueError):
            note_name_to_midi('X', 4)
        with pytest.raises(ValueError):
            note_name_to_midi('', 4)

    def test_note_name_to_midi_out_of_range(self):
        """Test out of MIDI range."""
        with pytest.raises(ValueError):
            note_name_to_midi('C', -2)
        with pytest.raises(ValueError):
            note_name_to_midi('A', 10)

    def test_midi_to_note_name_c4(self):
        """Test MIDI 60 to C4."""
        assert midi_to_note_name(60) == 'C4'

    def test_midi_to_note_name_sharps(self):
        """Test MIDI to sharp names."""
        assert midi_to_note_name(61) == 'C#4'
        assert midi_to_note_name(66) == 'F#4'

    def test_midi_to_note_name_flats(self):
        """Test MIDI to flat names."""
        assert midi_to_note_name(61, use_flats=True) == 'Db4'
        assert midi_to_note_name(70, use_flats=True) == 'Bb4'

    def test_midi_to_note_name_roundtrip(self):
        """Test conversion roundtrip."""
        for midi in range(0, 128):
            name = midi_to_note_name(midi)
            back = note_name_to_midi(name)
            assert back == midi


class TestNote:
    """Tests for Note class."""

    def test_create_note_basic(self):
        """Test basic note creation."""
        note = Note('C', 4)
        assert note.name == 'C'
        assert note.octave == 4
        assert note.velocity == 100

    def test_create_note_with_velocity(self):
        """Test note creation with velocity."""
        note = Note('E', 5, velocity=80)
        assert note.velocity == 80

    def test_note_midi_property(self):
        """Test MIDI number property."""
        assert Note('C', 4).midi == 60
        assert Note('A', 4).midi == 69
        assert Note('C', -1).midi == 0

    def test_note_pitch_class(self):
        """Test pitch class property."""
        assert Note('C', 4).pitch_class == 0
        assert Note('D', 3).pitch_class == 2
        assert Note('B', 5).pitch_class == 11

    def test_note_frequency(self):
        """Test frequency calculation."""
        a4 = Note('A', 4)
        assert a4.frequency == pytest.approx(440.0)

        c4 = Note('C', 4)
        assert c4.frequency == pytest.approx(261.63, rel=0.01)

    def test_note_from_midi(self):
        """Test creating Note from MIDI number."""
        note = Note.from_midi(60)
        assert note.name == 'C'
        assert note.octave == 4

        note = Note.from_midi(69)
        assert note.name == 'A'
        assert note.octave == 4

    def test_note_from_midi_with_flats(self):
        """Test creating Note from MIDI with flat names."""
        note = Note.from_midi(61, use_flats=True)
        # Flats are normalized to sharps internally for consistency
        assert note.midi == 61
        assert note.name == 'C#'  # Normalized from Db

    def test_note_transpose_semitones(self):
        """Test transposing by semitones."""
        c4 = Note('C', 4)
        g4 = c4.transpose(7)
        assert g4.name == 'G'
        assert g4.octave == 4

    def test_note_transpose_interval(self):
        """Test transposing by interval."""
        c4 = Note('C', 4)
        g4 = c4.transpose(Interval.PERFECT_FIFTH)
        assert g4.midi == 67

    def test_note_transpose_negative(self):
        """Test transposing down."""
        c4 = Note('C', 4)
        f3 = c4.transpose(-7)
        assert f3.name == 'F'
        assert f3.octave == 3

    def test_note_transpose_out_of_range(self):
        """Test transpose out of range raises error."""
        c4 = Note('C', 4)
        with pytest.raises(ValueError):
            c4.transpose(100)

    def test_note_interval_to(self):
        """Test getting interval between notes."""
        c4 = Note('C', 4)
        g4 = Note('G', 4)
        interval = c4.interval_to(g4)
        assert interval.semitones == 7

    def test_note_equality(self):
        """Test note equality based on MIDI."""
        assert Note('C', 4) == Note('C', 4)
        assert Note('C#', 4) == Note.from_midi(61)
        assert Note('C', 4) != Note('D', 4)

    def test_note_comparison(self):
        """Test note ordering."""
        assert Note('C', 4) < Note('D', 4)
        assert Note('C', 5) > Note('C', 4)

    def test_note_hash(self):
        """Test note hashing for sets/dicts."""
        notes = {Note('C', 4), Note('E', 4), Note('G', 4)}
        assert len(notes) == 3
        assert Note('C', 4) in notes

    def test_note_invalid_name(self):
        """Test invalid note name raises error."""
        with pytest.raises(ValueError):
            Note('X', 4)

    def test_note_invalid_octave(self):
        """Test invalid octave raises error."""
        with pytest.raises(ValueError):
            Note('C', 10)

    def test_note_invalid_velocity(self):
        """Test invalid velocity raises error."""
        with pytest.raises(ValueError):
            Note('C', 4, velocity=200)


class TestInterval:
    """Tests for Interval class."""

    def test_standard_intervals(self):
        """Test standard interval values."""
        assert Interval.UNISON.semitones == 0
        assert Interval.MINOR_SECOND.semitones == 1
        assert Interval.MAJOR_SECOND.semitones == 2
        assert Interval.MINOR_THIRD.semitones == 3
        assert Interval.MAJOR_THIRD.semitones == 4
        assert Interval.PERFECT_FOURTH.semitones == 5
        assert Interval.TRITONE.semitones == 6
        assert Interval.PERFECT_FIFTH.semitones == 7
        assert Interval.MINOR_SIXTH.semitones == 8
        assert Interval.MAJOR_SIXTH.semitones == 9
        assert Interval.MINOR_SEVENTH.semitones == 10
        assert Interval.MAJOR_SEVENTH.semitones == 11
        assert Interval.OCTAVE.semitones == 12

    def test_interval_from_semitones(self):
        """Test creating interval from semitones."""
        assert Interval.from_semitones(0) == Interval.UNISON
        assert Interval.from_semitones(7) == Interval.PERFECT_FIFTH
        assert Interval.from_semitones(12).semitones == 0  # Wraps to unison

    def test_interval_between_notes(self):
        """Test calculating interval between notes."""
        c4 = Note('C', 4)
        e4 = Note('E', 4)
        interval = Interval.between(c4, e4)
        assert interval.semitones == 4

    def test_interval_quality(self):
        """Test interval qualities."""
        assert Interval.PERFECT_FIFTH.quality == IntervalQuality.PERFECT
        assert Interval.MAJOR_THIRD.quality == IntervalQuality.MAJOR
        assert Interval.MINOR_THIRD.quality == IntervalQuality.MINOR


class TestScale:
    """Tests for Scale class."""

    def test_create_major_scale(self):
        """Test creating major scale."""
        scale = Scale(Note('C', 4), ScaleType.MAJOR)
        notes = scale.get_notes()

        assert len(notes) == 7
        assert [n.name for n in notes] == ['C', 'D', 'E', 'F', 'G', 'A', 'B']

    def test_create_natural_minor_scale(self):
        """Test creating natural minor scale."""
        scale = Scale(Note('A', 4), ScaleType.NATURAL_MINOR)
        notes = scale.get_notes()

        assert len(notes) == 7
        assert [n.name for n in notes] == ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    def test_scale_multiple_octaves(self):
        """Test scale across multiple octaves."""
        scale = Scale(Note('C', 4), ScaleType.MAJOR, octaves=2)
        notes = scale.get_notes()

        assert len(notes) == 14

    def test_scale_degree(self):
        """Test getting scale degree."""
        scale = Scale(Note('C', 4), ScaleType.MAJOR)

        assert scale.degree(1).name == 'C'
        assert scale.degree(3).name == 'E'
        assert scale.degree(5).name == 'G'

    def test_scale_degree_extended(self):
        """Test getting degrees beyond one octave."""
        scale = Scale(Note('C', 4), ScaleType.MAJOR)

        assert scale.degree(8).midi == Note('C', 5).midi
        assert scale.degree(9).midi == Note('D', 5).midi

    def test_scale_contains(self):
        """Test checking if note is in scale."""
        c_major = Scale(Note('C', 4), ScaleType.MAJOR)

        assert c_major.contains(Note('C', 4)) is True
        assert c_major.contains(Note('E', 5)) is True  # Different octave
        assert c_major.contains(Note('C#', 4)) is False
        assert c_major.contains(Note('F#', 4)) is False

    def test_scale_get_midi_notes(self):
        """Test getting MIDI note numbers."""
        scale = Scale(Note('C', 4), ScaleType.MAJOR)
        midi_notes = scale.get_midi_notes()

        assert midi_notes == [60, 62, 64, 65, 67, 69, 71]

    def test_scale_harmonize(self):
        """Test harmonizing scale degree."""
        c_major = Scale(Note('C', 4), ScaleType.MAJOR)

        # I chord = C major
        chord1 = c_major.harmonize(1)
        assert chord1.root.name == 'C'
        assert chord1.chord_type == ChordType.MAJOR

        # ii chord = D minor
        chord2 = c_major.harmonize(2)
        assert chord2.root.name == 'D'
        assert chord2.chord_type == ChordType.MINOR

        # V chord = G major
        chord5 = c_major.harmonize(5)
        assert chord5.root.name == 'G'
        assert chord5.chord_type == ChordType.MAJOR

        # vii chord = B diminished
        chord7 = c_major.harmonize(7)
        assert chord7.root.name == 'B'
        assert chord7.chord_type == ChordType.DIMINISHED

    def test_scale_parallel(self):
        """Test getting parallel scale."""
        c_major = Scale(Note('C', 4), ScaleType.MAJOR)
        c_minor = c_major.parallel(ScaleType.NATURAL_MINOR)

        assert c_minor.root.name == 'C'
        assert c_minor.scale_type == ScaleType.NATURAL_MINOR

    def test_scale_relative_minor(self):
        """Test getting relative minor."""
        c_major = Scale(Note('C', 4), ScaleType.MAJOR)
        a_minor = c_major.relative_minor()

        assert a_minor.root.name == 'A'
        assert a_minor.scale_type == ScaleType.NATURAL_MINOR

    def test_scale_relative_major(self):
        """Test getting relative major."""
        a_minor = Scale(Note('A', 4), ScaleType.NATURAL_MINOR)
        c_major = a_minor.relative_major()

        assert c_major.root.name == 'C'
        assert c_major.scale_type == ScaleType.MAJOR

    def test_pentatonic_scale(self):
        """Test pentatonic scale."""
        scale = Scale(Note('C', 4), ScaleType.MAJOR_PENTATONIC)
        notes = scale.get_notes()

        assert len(notes) == 5
        assert [n.name for n in notes] == ['C', 'D', 'E', 'G', 'A']

    def test_blues_scale(self):
        """Test blues scale."""
        scale = Scale(Note('C', 4), ScaleType.BLUES)
        notes = scale.get_notes()

        assert len(notes) == 6
        # C, Eb, F, F#, G, Bb
        midi_notes = [n.midi for n in notes]
        assert midi_notes == [60, 63, 65, 66, 67, 70]

    def test_scale_iteration(self):
        """Test iterating over scale."""
        scale = Scale(Note('C', 4), ScaleType.MAJOR)
        notes_iter = list(scale)

        assert len(notes_iter) == 7

    def test_scale_len(self):
        """Test scale length."""
        assert len(Scale(Note('C', 4), ScaleType.MAJOR)) == 7
        assert len(Scale(Note('C', 4), ScaleType.MAJOR_PENTATONIC)) == 5
        assert len(Scale(Note('C', 4), ScaleType.CHROMATIC)) == 12


class TestChord:
    """Tests for Chord class."""

    def test_create_major_chord(self):
        """Test creating major chord."""
        chord = Chord(Note('C', 4), ChordType.MAJOR)
        notes = chord.get_notes()

        assert len(notes) == 3
        assert [n.name for n in notes] == ['C', 'E', 'G']

    def test_create_minor_chord(self):
        """Test creating minor chord."""
        chord = Chord(Note('A', 4), ChordType.MINOR)
        notes = chord.get_notes()

        assert [n.name for n in notes] == ['A', 'C', 'E']

    def test_chord_get_midi_notes(self):
        """Test getting MIDI note numbers."""
        chord = Chord(Note('C', 4), ChordType.MAJOR)
        midi_notes = chord.get_midi_notes()

        assert midi_notes == [60, 64, 67]

    def test_seventh_chord(self):
        """Test seventh chord."""
        chord = Chord(Note('C', 4), ChordType.MAJOR_7)
        midi_notes = chord.get_midi_notes()

        assert midi_notes == [60, 64, 67, 71]

    def test_dominant_seventh(self):
        """Test dominant seventh chord."""
        chord = Chord(Note('G', 4), ChordType.DOMINANT_7)
        midi_notes = chord.get_midi_notes()

        # G, B, D, F
        assert midi_notes == [67, 71, 74, 77]

    def test_diminished_chord(self):
        """Test diminished chord."""
        chord = Chord(Note('B', 4), ChordType.DIMINISHED)
        notes = chord.get_notes()

        # B, D, F
        intervals = [n.midi - chord.root.midi for n in notes]
        assert intervals == [0, 3, 6]

    def test_augmented_chord(self):
        """Test augmented chord."""
        chord = Chord(Note('C', 4), ChordType.AUGMENTED)
        notes = chord.get_notes()

        # C, E, G#
        intervals = [n.midi - chord.root.midi for n in notes]
        assert intervals == [0, 4, 8]

    def test_sus2_chord(self):
        """Test sus2 chord."""
        chord = Chord(Note('C', 4), ChordType.SUS2)
        notes = chord.get_notes()

        # C, D, G
        assert [n.name for n in notes] == ['C', 'D', 'G']

    def test_sus4_chord(self):
        """Test sus4 chord."""
        chord = Chord(Note('C', 4), ChordType.SUS4)
        notes = chord.get_notes()

        # C, F, G
        assert [n.name for n in notes] == ['C', 'F', 'G']

    def test_chord_inversion_first(self):
        """Test first inversion."""
        chord = Chord(Note('C', 4), ChordType.MAJOR)
        inv1 = chord.inversion(1)

        # E is now the bass
        assert inv1.root.name == 'E'

    def test_chord_inversion_second(self):
        """Test second inversion."""
        chord = Chord(Note('C', 4), ChordType.MAJOR)
        inv2 = chord.inversion(2)

        # G is now the bass
        assert inv2.root.name == 'G'

    def test_chord_transpose(self):
        """Test transposing chord."""
        c_major = Chord(Note('C', 4), ChordType.MAJOR)
        g_major = c_major.transpose(7)

        assert g_major.root.name == 'G'
        assert g_major.chord_type == ChordType.MAJOR

    def test_chord_from_symbol_major(self):
        """Test parsing major chord symbols."""
        chord = Chord.from_symbol('C')
        assert chord.root.name == 'C'
        assert chord.chord_type == ChordType.MAJOR

        chord = Chord.from_symbol('Cmaj')
        assert chord.chord_type == ChordType.MAJOR

    def test_chord_from_symbol_minor(self):
        """Test parsing minor chord symbols."""
        chord = Chord.from_symbol('Am')
        assert chord.root.name == 'A'
        assert chord.chord_type == ChordType.MINOR

        chord = Chord.from_symbol('F#m')
        assert chord.root.name == 'F#'
        assert chord.chord_type == ChordType.MINOR

    def test_chord_from_symbol_seventh(self):
        """Test parsing seventh chord symbols."""
        chord = Chord.from_symbol('Cmaj7')
        assert chord.chord_type == ChordType.MAJOR_7

        chord = Chord.from_symbol('Dm7')
        assert chord.chord_type == ChordType.MINOR_7

        chord = Chord.from_symbol('G7')
        assert chord.chord_type == ChordType.DOMINANT_7

    def test_chord_from_symbol_with_flat(self):
        """Test parsing chord with flat root."""
        chord = Chord.from_symbol('Bb7')
        # Bb is normalized to A# internally for consistency
        assert chord.root.name == 'A#'
        assert chord.chord_type == ChordType.DOMINANT_7

    def test_chord_symbol_property(self):
        """Test chord symbol generation."""
        chord = Chord(Note('C', 4), ChordType.MAJOR_7)
        assert 'C' in chord.symbol
        assert 'maj7' in chord.symbol or 'M7' in chord.symbol

    def test_chord_iteration(self):
        """Test iterating over chord notes."""
        chord = Chord(Note('C', 4), ChordType.MAJOR)
        notes = list(chord)

        assert len(notes) == 3

    def test_chord_len(self):
        """Test chord length."""
        assert len(Chord(Note('C', 4), ChordType.MAJOR)) == 3
        assert len(Chord(Note('C', 4), ChordType.MAJOR_7)) == 4
        assert len(Chord(Note('C', 4), ChordType.MAJOR_9)) == 5

    def test_extended_chords(self):
        """Test extended chords (9, 11, 13)."""
        chord9 = Chord(Note('C', 4), ChordType.MAJOR_9)
        assert len(chord9) == 5

        chord11 = Chord(Note('C', 4), ChordType.MAJOR_11)
        assert len(chord11) == 6

        chord13 = Chord(Note('C', 4), ChordType.MAJOR_13)
        assert len(chord13) == 7


class TestChordProgression:
    """Tests for ChordProgression class."""

    def test_from_numerals_basic(self):
        """Test creating progression from Roman numerals."""
        prog = ChordProgression.from_numerals('C', ['I', 'IV', 'V', 'I'])

        assert len(prog.chords) == 4
        assert prog.chords[0].root.name == 'C'
        assert prog.chords[1].root.name == 'F'
        assert prog.chords[2].root.name == 'G'
        assert prog.chords[3].root.name == 'C'

    def test_from_numerals_with_minor(self):
        """Test progression with minor chords."""
        prog = ChordProgression.from_numerals('C', ['I', 'V', 'vi', 'IV'])

        assert prog.chords[0].chord_type == ChordType.MAJOR
        assert prog.chords[2].chord_type == ChordType.MINOR
        assert prog.chords[2].root.name == 'A'

    def test_from_numerals_with_sevenths(self):
        """Test progression with seventh chords."""
        prog = ChordProgression.from_numerals('C', ['I7', 'IV7', 'V7', 'I7'])

        assert prog.chords[0].chord_type == ChordType.MAJOR_7
        assert prog.chords[2].chord_type == ChordType.DOMINANT_7

    def test_from_symbols(self):
        """Test creating progression from chord symbols."""
        prog = ChordProgression.from_symbols(['Cmaj7', 'Dm7', 'G7', 'Cmaj7'])

        assert len(prog.chords) == 4
        assert prog.chords[0].chord_type == ChordType.MAJOR_7
        assert prog.chords[1].chord_type == ChordType.MINOR_7

    def test_progression_transpose(self):
        """Test transposing progression."""
        prog = ChordProgression.from_numerals('C', ['I', 'IV', 'V'])
        transposed = prog.transpose(5)  # Up to F

        assert transposed.chords[0].root.name == 'F'
        assert transposed.chords[1].root.name == 'A#'
        assert transposed.chords[2].root.name == 'C'

    def test_progression_iteration(self):
        """Test iterating over progression."""
        prog = ChordProgression.from_numerals('C', ['I', 'IV', 'V', 'I'])
        chords = list(prog)

        assert len(chords) == 4


class TestKeySignaturesAndCircleOfFifths:
    """Tests for key signatures and circle of fifths."""

    def test_c_major_key_signature(self):
        """Test C major has no sharps/flats."""
        sharps, is_minor = KEY_SIGNATURES['C']
        assert sharps == []
        assert is_minor is False

    def test_g_major_key_signature(self):
        """Test G major has one sharp."""
        sharps, _ = KEY_SIGNATURES['G']
        assert 'F#' in sharps
        assert len(sharps) == 1

    def test_f_major_key_signature(self):
        """Test F major has one flat."""
        flats, _ = KEY_SIGNATURES['F']
        assert 'Bb' in flats
        assert len(flats) == 1

    def test_relative_minor_keys(self):
        """Test relative minor key signatures."""
        sharps_c, is_minor_c = KEY_SIGNATURES['C']
        sharps_am, is_minor_am = KEY_SIGNATURES['Am']

        assert sharps_c == sharps_am
        assert is_minor_c is False
        assert is_minor_am is True

    def test_circle_of_fifths(self):
        """Test circle of fifths."""
        assert CIRCLE_OF_FIFTHS[0] == 'C'
        assert 'G' in CIRCLE_OF_FIFTHS
        assert 'F' in CIRCLE_OF_FIFTHS
        assert len(CIRCLE_OF_FIFTHS) == 12
