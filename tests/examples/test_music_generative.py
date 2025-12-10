#!/usr/bin/env python3
"""Tests for generative music module."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from coremusic.music.theory import Note, Chord, ChordType, Scale, ScaleType
from coremusic.music.generative import (
    Generator,
    GeneratorConfig,
    Arpeggiator,
    ArpPattern,
    ArpConfig,
    EuclideanGenerator,
    EuclideanConfig,
    MarkovGenerator,
    MarkovConfig,
    ProbabilisticGenerator,
    ProbabilisticConfig,
    SequenceGenerator,
    SequenceConfig,
    Step,
    MelodyGenerator,
    MelodyConfig,
    PolyrhythmGenerator,
    PolyrhythmConfig,
    RhythmLayer,
    BitShiftRegister,
    BitShiftRegisterGenerator,
    BitShiftRegisterConfig,
    create_arp_from_progression,
    combine_generators,
)
from coremusic.midi.utilities import MIDIStatus


class TestGeneratorConfig:
    """Tests for GeneratorConfig base class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GeneratorConfig()

        assert config.tempo == 120.0
        assert config.channel == 0
        assert config.velocity == 100
        assert config.swing == 0.0
        assert config.humanize_timing == 0.0
        assert config.humanize_velocity == 0

    def test_config_validation_channel(self):
        """Test channel validation."""
        with pytest.raises(ValueError, match="Channel must be 0-15"):
            GeneratorConfig(channel=16)

    def test_config_validation_velocity(self):
        """Test velocity validation."""
        with pytest.raises(ValueError, match="Velocity must be 0-127"):
            GeneratorConfig(velocity=128)

    def test_config_validation_swing(self):
        """Test swing validation."""
        with pytest.raises(ValueError, match="Swing must be 0.0-1.0"):
            GeneratorConfig(swing=1.5)


class TestArpeggiator:
    """Tests for Arpeggiator class."""

    def test_create_arpeggiator(self):
        """Test creating arpeggiator."""
        chord = Chord(Note('C', 4), ChordType.MAJOR)
        arp = Arpeggiator(chord, ArpPattern.UP)

        assert arp.chord == chord
        assert arp.pattern == ArpPattern.UP

    def test_arp_pattern_up(self):
        """Test ascending arpeggio pattern."""
        chord = Chord(Note('C', 4), ChordType.MAJOR)
        arp = Arpeggiator(chord, ArpPattern.UP)
        events = arp.generate(num_cycles=1)

        # Extract note on events
        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]
        pitches = [e.data1 for e in note_ons]

        # C4, E4, G4 = 60, 64, 67
        assert pitches == [60, 64, 67]

    def test_arp_pattern_down(self):
        """Test descending arpeggio pattern."""
        chord = Chord(Note('C', 4), ChordType.MAJOR)
        arp = Arpeggiator(chord, ArpPattern.DOWN)
        events = arp.generate(num_cycles=1)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]
        pitches = [e.data1 for e in note_ons]

        # G4, E4, C4 = 67, 64, 60
        assert pitches == [67, 64, 60]

    def test_arp_pattern_up_down(self):
        """Test up-down arpeggio pattern."""
        chord = Chord(Note('C', 4), ChordType.MAJOR)
        arp = Arpeggiator(chord, ArpPattern.UP_DOWN)
        events = arp.generate(num_cycles=1)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]
        pitches = [e.data1 for e in note_ons]

        # C4, E4, G4, E4 (endpoints not repeated in down)
        assert pitches == [60, 64, 67, 64]

    def test_arp_pattern_up_down_inclusive(self):
        """Test up-down inclusive pattern."""
        chord = Chord(Note('C', 4), ChordType.MAJOR)
        arp = Arpeggiator(chord, ArpPattern.UP_DOWN_INCLUSIVE)
        events = arp.generate(num_cycles=1)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]
        pitches = [e.data1 for e in note_ons]

        # C4, E4, G4, G4, E4, C4 (endpoints repeated)
        assert pitches == [60, 64, 67, 67, 64, 60]

    def test_arp_multiple_octaves(self):
        """Test arpeggio across multiple octaves."""
        chord = Chord(Note('C', 4), ChordType.MAJOR)
        config = ArpConfig(octave_range=2)
        arp = Arpeggiator(chord, ArpPattern.UP, config)
        events = arp.generate(num_cycles=1)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]
        pitches = [e.data1 for e in note_ons]

        # Should include notes across 2 octaves
        assert 60 in pitches
        assert 72 in pitches  # C5

    def test_arp_generate_with_duration(self):
        """Test generating for specific duration."""
        chord = Chord(Note('C', 4), ChordType.MAJOR)
        config = ArpConfig(tempo=120.0, rate=0.25)
        arp = Arpeggiator(chord, ArpPattern.UP, config)

        # Generate for 2 seconds at 120 BPM
        events = arp.generate(duration=2.0)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]
        # At 120 BPM, 16th notes = 8 notes/sec, so ~16 notes in 2 sec
        assert len(note_ons) >= 10

    def test_arp_set_chord(self):
        """Test changing chord dynamically."""
        arp = Arpeggiator(pattern=ArpPattern.UP)
        chord1 = Chord(Note('C', 4), ChordType.MAJOR)
        chord2 = Chord(Note('A', 4), ChordType.MINOR)

        arp.set_chord(chord1)
        events1 = arp.generate(num_cycles=1)
        note_ons1 = [e.data1 for e in events1 if e.status == MIDIStatus.NOTE_ON]

        arp.set_chord(chord2)
        events2 = arp.generate(num_cycles=1)
        note_ons2 = [e.data1 for e in events2 if e.status == MIDIStatus.NOTE_ON]

        assert note_ons1 != note_ons2

    def test_arp_velocity_pattern(self):
        """Test velocity pattern."""
        chord = Chord(Note('C', 4), ChordType.MAJOR)
        config = ArpConfig(velocity_pattern=[100, 80, 60])
        arp = Arpeggiator(chord, ArpPattern.UP, config)
        events = arp.generate(num_cycles=1)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]
        velocities = [e.data2 for e in note_ons]

        assert velocities == [100, 80, 60]

    def test_arp_random_pattern(self):
        """Test random pattern produces valid notes."""
        chord = Chord(Note('C', 4), ChordType.MAJOR)
        config = ArpConfig(seed=42)  # Fixed seed for reproducibility
        arp = Arpeggiator(chord, ArpPattern.RANDOM, config)
        events = arp.generate(num_cycles=1)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]
        pitches = set(e.data1 for e in note_ons)

        # All pitches should be chord tones
        chord_tones = set(chord.get_midi_notes())
        assert pitches.issubset(chord_tones)

    def test_arp_no_chord_returns_empty(self):
        """Test generating without chord returns empty."""
        arp = Arpeggiator(pattern=ArpPattern.UP)
        events = arp.generate(num_cycles=1)

        assert events == []


class TestEuclideanGenerator:
    """Tests for EuclideanGenerator class."""

    def test_euclidean_basic_pattern(self):
        """Test basic Euclidean pattern."""
        euclid = EuclideanGenerator(pulses=3, steps=8)
        pattern = euclid.get_pattern()

        assert len(pattern) == 8
        assert sum(pattern) == 3

    def test_euclidean_known_patterns(self):
        """Test known Euclidean patterns."""
        # Tresillo (Cuban rhythm) - E(3,8)
        euclid = EuclideanGenerator(pulses=3, steps=8)
        pattern = euclid.get_pattern()
        assert sum(pattern) == 3
        assert pattern == [1, 0, 0, 1, 0, 0, 1, 0]

        # E(5,8) - evenly distributed 5 pulses over 8 steps
        euclid = EuclideanGenerator(pulses=5, steps=8)
        pattern = euclid.get_pattern()
        assert sum(pattern) == 5
        # The algorithm produces this specific pattern
        assert pattern == [1, 0, 1, 0, 1, 0, 1, 1]

    def test_euclidean_all_pulses(self):
        """Test all steps are pulses."""
        euclid = EuclideanGenerator(pulses=4, steps=4)
        assert euclid.get_pattern() == [1, 1, 1, 1]

    def test_euclidean_no_pulses(self):
        """Test zero pulses."""
        euclid = EuclideanGenerator(pulses=0, steps=4)
        assert euclid.get_pattern() == [0, 0, 0, 0]

    def test_euclidean_rotation(self):
        """Test pattern rotation."""
        config = EuclideanConfig(rotation=1)
        euclid = EuclideanGenerator(pulses=3, steps=8, config=config)
        pattern = euclid.get_pattern()

        # Original: [1, 0, 0, 1, 0, 0, 1, 0]
        # Rotated 1: [0, 0, 1, 0, 0, 1, 0, 1]
        assert pattern[0] == 0
        assert pattern[-1] == 1

    def test_euclidean_generate_events(self):
        """Test generating MIDI events."""
        euclid = EuclideanGenerator(pulses=4, steps=8, pitch=36)
        events = euclid.generate(cycles=1)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]

        assert len(note_ons) == 4
        assert all(e.data1 == 36 for e in note_ons)

    def test_euclidean_generate_multiple_cycles(self):
        """Test generating multiple cycles."""
        euclid = EuclideanGenerator(pulses=3, steps=8, pitch=36)
        events = euclid.generate(cycles=4)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]

        assert len(note_ons) == 12  # 3 pulses * 4 cycles

    def test_euclidean_with_note(self):
        """Test creating with Note object."""
        note = Note('C', 2)
        euclid = EuclideanGenerator(pulses=4, steps=8, pitch=note)

        assert euclid.pitch == 36  # C2

    def test_euclidean_pulses_exceed_steps_raises(self):
        """Test pulses > steps raises error."""
        with pytest.raises(ValueError):
            EuclideanGenerator(pulses=10, steps=8)

    def test_euclidean_set_parameters(self):
        """Test changing parameters."""
        euclid = EuclideanGenerator(pulses=3, steps=8)
        euclid.set_parameters(pulses=5, steps=8)

        assert sum(euclid.get_pattern()) == 5


class TestMarkovGenerator:
    """Tests for MarkovGenerator class."""

    def test_markov_train_simple(self):
        """Test training from simple sequence."""
        markov = MarkovGenerator()
        markov.train([60, 62, 64, 62, 60])

        assert 60 in markov.transitions
        assert 62 in markov.transitions

    def test_markov_set_transitions(self):
        """Test setting transitions manually."""
        markov = MarkovGenerator()
        markov.set_transitions({
            60: {62: 0.5, 64: 0.5},
            62: {60: 1.0},
            64: {60: 1.0},
        })

        assert 60 in markov.transitions
        assert sum(markov.transitions[60].values()) == pytest.approx(1.0)

    def test_markov_generate(self):
        """Test generating melody."""
        markov = MarkovGenerator()
        markov.set_transitions({
            60: {62: 0.5, 64: 0.5},
            62: {60: 0.5, 64: 0.5},
            64: {60: 0.5, 62: 0.5},
        })

        config = MarkovConfig(seed=42)
        markov.config = config
        events = markov.generate(num_notes=10, start_note=60)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]
        assert len(note_ons) == 10

    def test_markov_with_scale_constraint(self):
        """Test constraining to scale."""
        scale = Scale(Note('C', 4), ScaleType.MAJOR)
        markov = MarkovGenerator(scale=scale)

        # Train with chromatic sequence
        markov.train([60, 61, 62, 63, 64, 65])

        config = MarkovConfig(seed=42)
        markov.config = config
        events = markov.generate(num_notes=10, start_note=60)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]

        # All notes should be in scale
        for event in note_ons:
            assert scale.contains(Note.from_midi(event.data1))

    def test_markov_empty_transitions(self):
        """Test generating with no transitions returns empty."""
        markov = MarkovGenerator()
        events = markov.generate(num_notes=10)

        assert events == []

    def test_markov_with_note_objects(self):
        """Test training with Note objects."""
        markov = MarkovGenerator()
        notes = [Note('C', 4), Note('D', 4), Note('E', 4), Note('D', 4)]
        markov.train(notes)

        assert 60 in markov.transitions


class TestProbabilisticGenerator:
    """Tests for ProbabilisticGenerator class."""

    def test_probabilistic_with_scale(self):
        """Test generating from scale."""
        scale = Scale(Note('C', 4), ScaleType.MAJOR)
        prob = ProbabilisticGenerator(scale)
        events = prob.generate(num_notes=10)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]

        assert len(note_ons) == 10
        for event in note_ons:
            assert scale.contains(Note.from_midi(event.data1))

    def test_probabilistic_with_weights(self):
        """Test generating with custom weights."""
        prob = ProbabilisticGenerator()
        prob.set_weights({60: 10, 64: 1})  # Heavily favor C over E

        config = ProbabilisticConfig(seed=42)
        prob.config = config
        events = prob.generate(num_notes=100)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]
        c_count = sum(1 for e in note_ons if e.data1 == 60)
        e_count = sum(1 for e in note_ons if e.data1 == 64)

        # C should appear much more often
        assert c_count > e_count * 3

    def test_probabilistic_with_rests(self):
        """Test generating with rest probability."""
        scale = Scale(Note('C', 4), ScaleType.MAJOR)
        config = ProbabilisticConfig(rest_probability=0.5, seed=42)
        prob = ProbabilisticGenerator(scale, config=config)

        events = prob.generate(num_notes=100)
        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]

        # Should have significantly fewer than 100 notes due to rests
        assert len(note_ons) < 80

    def test_probabilistic_default_notes(self):
        """Test default note range when no scale/weights."""
        prob = ProbabilisticGenerator()
        events = prob.generate(num_notes=10)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]

        # Default is C4-B4 (60-71)
        for event in note_ons:
            assert 60 <= event.data1 <= 71


class TestSequenceGenerator:
    """Tests for SequenceGenerator class."""

    def test_sequence_create(self):
        """Test creating sequence generator."""
        seq = SequenceGenerator(steps=16)
        assert seq.num_steps == 16

    def test_sequence_set_step(self):
        """Test setting a step."""
        seq = SequenceGenerator(steps=8)
        seq.set_step(0, Step(pitch=60, velocity=100))
        seq.set_step(4, Step(pitch=64))

        events = seq.generate(cycles=1)
        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]

        assert len(note_ons) == 2
        pitches = [e.data1 for e in note_ons]
        assert 60 in pitches
        assert 64 in pitches

    def test_sequence_set_pattern(self):
        """Test setting pattern from list."""
        seq = SequenceGenerator(steps=8)
        seq.set_pattern([60, None, 64, None, 67, None, 64, None])

        events = seq.generate(cycles=1)
        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]

        assert len(note_ons) == 4
        pitches = [e.data1 for e in note_ons]
        assert pitches == [60, 64, 67, 64]

    def test_sequence_clear_step(self):
        """Test clearing a step."""
        seq = SequenceGenerator(steps=8)
        seq.set_step(0, Step(pitch=60))
        seq.set_step(1, Step(pitch=64))
        seq.clear_step(1)

        events = seq.generate(cycles=1)
        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]

        assert len(note_ons) == 1

    def test_sequence_step_probability(self):
        """Test step probability."""
        seq = SequenceGenerator(steps=8)
        config = SequenceConfig(seed=42)
        seq.config = config

        # 50% probability
        seq.set_step(0, Step(pitch=60, probability=0.5))

        # Generate many cycles and check hit rate
        total_notes = 0
        for _ in range(100):
            events = seq.generate(cycles=1)
            note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]
            total_notes += len(note_ons)

        # Should be roughly 50% (allow some variance)
        assert 30 <= total_notes <= 70

    def test_sequence_step_gate(self):
        """Test per-step gate time."""
        seq = SequenceGenerator(steps=4)
        config = SequenceConfig(tempo=120.0, step_duration=0.5, gate=0.5)
        seq.config = config

        seq.set_step(0, Step(pitch=60, gate=1.0))  # Full gate
        seq.set_step(1, Step(pitch=64, gate=0.25))  # Short gate

        events = seq.generate(cycles=1)

        # Find note durations by pairing on/off events
        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]
        note_offs = [e for e in events if e.status == MIDIStatus.NOTE_OFF]

        # First note should be longer than second
        dur1 = note_offs[0].time - note_ons[0].time
        dur2 = note_offs[1].time - note_ons[1].time
        assert dur1 > dur2

    def test_sequence_multiple_cycles(self):
        """Test generating multiple cycles."""
        seq = SequenceGenerator(steps=4)
        seq.set_pattern([60, 64, 67, 72])

        events = seq.generate(cycles=4)
        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]

        assert len(note_ons) == 16


class TestMelodyGenerator:
    """Tests for MelodyGenerator class."""

    def test_melody_create(self):
        """Test creating melody generator."""
        scale = Scale(Note('C', 4), ScaleType.MAJOR)
        melody = MelodyGenerator(scale)

        assert melody.scale == scale

    def test_melody_generate_basic(self):
        """Test basic melody generation."""
        scale = Scale(Note('C', 4), ScaleType.MAJOR)
        melody = MelodyGenerator(scale)
        events = melody.generate(num_notes=16)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]

        # Should generate notes
        assert len(note_ons) > 0

        # All should be in scale
        for event in note_ons:
            assert scale.contains(Note.from_midi(event.data1))

    def test_melody_start_note(self):
        """Test specifying start note."""
        scale = Scale(Note('C', 4), ScaleType.MAJOR)
        melody = MelodyGenerator(scale)
        events = melody.generate(num_notes=5, start_note=Note('G', 4))

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]

        # First note should be G4
        assert note_ons[0].data1 == 67

    def test_melody_max_jump(self):
        """Test max interval jump."""
        scale = Scale(Note('C', 4), ScaleType.MAJOR)
        config = MelodyConfig(max_jump=2, seed=42)  # Only seconds
        melody = MelodyGenerator(scale, config)
        events = melody.generate(num_notes=20)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]

        # Check consecutive intervals
        for i in range(1, len(note_ons)):
            interval = abs(note_ons[i].data1 - note_ons[i-1].data1)
            # Allow small intervals (may be 0 if same note)
            assert interval <= 5  # Account for scale steps, not semitones

    def test_melody_with_rests(self):
        """Test melody with phrase rests."""
        scale = Scale(Note('C', 4), ScaleType.MAJOR)
        config = MelodyConfig(
            phrase_length=4,
            rest_probability=1.0,  # Always rest between phrases
            seed=42,
        )
        melody = MelodyGenerator(scale, config)
        events = melody.generate(num_notes=20)

        # Should have gaps in the note sequence
        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]
        assert len(note_ons) < 20


class TestPolyrhythmGenerator:
    """Tests for PolyrhythmGenerator class."""

    def test_polyrhythm_create(self):
        """Test creating polyrhythm generator."""
        poly = PolyrhythmGenerator(cycle_beats=4)
        assert poly.cycle_beats == 4

    def test_polyrhythm_add_layer(self):
        """Test adding rhythm layers."""
        poly = PolyrhythmGenerator(cycle_beats=4)
        poly.add_layer(RhythmLayer(pulses=3, pitch=36))
        poly.add_layer(RhythmLayer(pulses=4, pitch=38))

        assert len(poly.layers) == 2

    def test_polyrhythm_generate_3_4(self):
        """Test generating 3:4 polyrhythm."""
        poly = PolyrhythmGenerator(cycle_beats=4)
        poly.add_layer(RhythmLayer(pulses=3, pitch=36))
        poly.add_layer(RhythmLayer(pulses=4, pitch=38))

        events = poly.generate(cycles=1)

        kick_notes = [e for e in events if e.status == MIDIStatus.NOTE_ON and e.data1 == 36]
        snare_notes = [e for e in events if e.status == MIDIStatus.NOTE_ON and e.data1 == 38]

        assert len(kick_notes) == 3
        assert len(snare_notes) == 4

    def test_polyrhythm_multiple_cycles(self):
        """Test generating multiple cycles."""
        poly = PolyrhythmGenerator(cycle_beats=4)
        poly.add_layer(RhythmLayer(pulses=3, pitch=36))

        events = poly.generate(cycles=4)
        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]

        assert len(note_ons) == 12

    def test_polyrhythm_with_offset(self):
        """Test layer offset."""
        poly = PolyrhythmGenerator(cycle_beats=4)
        poly.add_layer(RhythmLayer(pulses=4, pitch=36, offset=0.0))
        poly.add_layer(RhythmLayer(pulses=4, pitch=38, offset=0.5))  # Half beat offset

        events = poly.generate(cycles=1)

        kicks = sorted([e for e in events if e.status == MIDIStatus.NOTE_ON and e.data1 == 36],
                       key=lambda e: e.time)
        snares = sorted([e for e in events if e.status == MIDIStatus.NOTE_ON and e.data1 == 38],
                        key=lambda e: e.time)

        # Snares should be offset from kicks
        assert snares[0].time > kicks[0].time

    def test_polyrhythm_clear_layers(self):
        """Test clearing layers."""
        poly = PolyrhythmGenerator(cycle_beats=4)
        poly.add_layer(RhythmLayer(pulses=3, pitch=36))
        poly.clear_layers()

        assert len(poly.layers) == 0


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_arp_from_progression(self):
        """Test creating arpeggiated progression."""
        progression = [
            Chord(Note('C', 4), ChordType.MAJOR),
            Chord(Note('F', 4), ChordType.MAJOR),
            Chord(Note('G', 4), ChordType.MAJOR),
        ]

        events = create_arp_from_progression(progression, ArpPattern.UP, beats_per_chord=2)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]

        # Should have notes from all three chords
        assert len(note_ons) > 0

        # Check that notes span all chords
        pitches = set(e.data1 for e in note_ons)
        assert 60 in pitches  # C from C chord
        assert 65 in pitches  # F from F chord
        assert 67 in pitches  # G from G chord

    def test_combine_generators(self):
        """Test combining multiple generators."""
        chord = Chord(Note('C', 4), ChordType.MAJOR)
        arp = Arpeggiator(chord, ArpPattern.UP)

        euclid = EuclideanGenerator(pulses=4, steps=8, pitch=36)

        generators = [
            (arp, {'num_cycles': 2}),
            (euclid, {'cycles': 2}),
        ]

        events = combine_generators(generators)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]

        # Should have events from both generators
        arp_notes = [e for e in note_ons if e.data1 in [60, 64, 67]]
        euclid_notes = [e for e in note_ons if e.data1 == 36]

        assert len(arp_notes) > 0
        assert len(euclid_notes) > 0


class TestSwingAndHumanization:
    """Tests for swing and humanization features."""

    def test_swing_application(self):
        """Test that swing delays odd-numbered steps."""
        chord = Chord(Note('C', 4), ChordType.MAJOR)
        config = ArpConfig(tempo=120.0, swing=0.5, rate=0.5)  # 8th notes
        arp = Arpeggiator(chord, ArpPattern.UP, config)

        events = arp.generate(num_cycles=2)
        note_ons = sorted([e for e in events if e.status == MIDIStatus.NOTE_ON],
                          key=lambda e: e.time)

        # With swing, intervals between even->odd should be longer than odd->even
        if len(note_ons) >= 3:
            interval_0_1 = note_ons[1].time - note_ons[0].time
            interval_1_2 = note_ons[2].time - note_ons[1].time

            # First interval (even to odd) should be longer due to swing
            assert interval_0_1 > interval_1_2

    def test_humanize_timing(self):
        """Test timing humanization adds variation."""
        scale = Scale(Note('C', 4), ScaleType.MAJOR)
        config = ProbabilisticConfig(
            humanize_timing=0.01,  # 10ms variation
            seed=42,
        )
        prob = ProbabilisticGenerator(scale, config=config)

        # Generate twice with same seed but different instances
        events1 = prob.generate(num_notes=8)
        prob2 = ProbabilisticGenerator(scale, config=config)
        events2 = prob2.generate(num_notes=8)

        times1 = [e.time for e in events1 if e.status == MIDIStatus.NOTE_ON]
        times2 = [e.time for e in events2 if e.status == MIDIStatus.NOTE_ON]

        # With same seed, times should be identical
        assert times1 == times2

    def test_humanize_velocity(self):
        """Test velocity humanization adds variation."""
        scale = Scale(Note('C', 4), ScaleType.MAJOR)
        config = ProbabilisticConfig(
            velocity=100,
            humanize_velocity=10,
            seed=42,
        )
        prob = ProbabilisticGenerator(scale, config=config)
        events = prob.generate(num_notes=20)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]
        velocities = [e.data2 for e in note_ons]

        # Should have some variation
        assert min(velocities) != max(velocities)
        # All should be within range
        for v in velocities:
            assert 90 <= v <= 110


class TestReproducibility:
    """Tests for random seed reproducibility."""

    def test_arpeggiator_reproducibility(self):
        """Test arpeggiator produces same output with same seed."""
        chord = Chord(Note('C', 4), ChordType.MAJOR)
        config = ArpConfig(seed=12345, humanize_timing=0.01)

        arp1 = Arpeggiator(chord, ArpPattern.RANDOM, config)
        events1 = arp1.generate(num_cycles=4)

        arp2 = Arpeggiator(chord, ArpPattern.RANDOM, ArpConfig(seed=12345, humanize_timing=0.01))
        events2 = arp2.generate(num_cycles=4)

        pitches1 = [e.data1 for e in events1 if e.status == MIDIStatus.NOTE_ON]
        pitches2 = [e.data1 for e in events2 if e.status == MIDIStatus.NOTE_ON]

        assert pitches1 == pitches2

    def test_markov_reproducibility(self):
        """Test Markov produces same output with same seed."""
        config = MarkovConfig(seed=12345)

        markov1 = MarkovGenerator(config=config)
        markov1.set_transitions({60: {62: 0.5, 64: 0.5}, 62: {60: 0.5, 64: 0.5}, 64: {60: 0.5, 62: 0.5}})
        events1 = markov1.generate(num_notes=20, start_note=60)

        markov2 = MarkovGenerator(config=MarkovConfig(seed=12345))
        markov2.set_transitions({60: {62: 0.5, 64: 0.5}, 62: {60: 0.5, 64: 0.5}, 64: {60: 0.5, 62: 0.5}})
        events2 = markov2.generate(num_notes=20, start_note=60)

        pitches1 = [e.data1 for e in events1 if e.status == MIDIStatus.NOTE_ON]
        pitches2 = [e.data1 for e in events2 if e.status == MIDIStatus.NOTE_ON]

        assert pitches1 == pitches2


# ============================================================================
# MIDI File Generation Tests
# ============================================================================

class TestMIDIFileGeneration:
    """Tests that generate actual MIDI files demonstrating each algorithm.

    Generated files are saved to build/midi_files/<generator_type>/ for audition.
    """

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Create base output directory for MIDI files."""
        self.base_output_dir = Path(__file__).parent.parent / "build" / "midi_files"
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def _get_output_dir(self, generator_type: str) -> Path:
        """Get output directory for a specific generator type."""
        output_dir = self.base_output_dir / generator_type
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _events_to_track(self, events, track):
        """Helper to convert generator events to MIDITrack."""
        from coremusic.midi.utilities import MIDIStatus as MS
        # Group note on/off pairs
        note_ons = {}
        for event in sorted(events, key=lambda e: e.time):
            if event.status == MS.NOTE_ON and event.data2 > 0:
                key = (event.data1, event.channel)
                note_ons[key] = event
            elif event.status == MS.NOTE_OFF or (event.status == MS.NOTE_ON and event.data2 == 0):
                key = (event.data1, event.channel)
                if key in note_ons:
                    on_event = note_ons.pop(key)
                    duration = event.time - on_event.time
                    track.add_note(on_event.time, on_event.data1, on_event.data2,
                                   duration, on_event.channel)

    def test_generate_arpeggiator_up(self):
        """Generate MIDI file with ascending arpeggio."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=120.0)
        track = seq.add_track("Arpeggiator Up")

        chord = Chord(Note('C', 4), ChordType.MAJOR_7)
        config = ArpConfig(tempo=120.0, octave_range=2, rate=0.25)
        arp = Arpeggiator(chord, ArpPattern.UP, config)
        events = arp.generate(num_cycles=4)

        self._events_to_track(events, track)

        output_path = self._get_output_dir("arpeggiator") / "arpeggiator_up.mid"
        seq.save(str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_generate_arpeggiator_up_down(self):
        """Generate MIDI file with up-down arpeggio."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=130.0)
        track = seq.add_track("Arpeggiator Up-Down")

        chord = Chord(Note('A', 3), ChordType.MINOR_7)
        config = ArpConfig(tempo=130.0, octave_range=2, rate=0.125, swing=0.3)
        arp = Arpeggiator(chord, ArpPattern.UP_DOWN, config)
        events = arp.generate(num_cycles=4)

        self._events_to_track(events, track)

        output_path = self._get_output_dir("arpeggiator") / "arpeggiator_up_down.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_arpeggiator_random(self):
        """Generate MIDI file with random arpeggio pattern."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=110.0)
        track = seq.add_track("Arpeggiator Random")

        chord = Chord(Note('D', 4), ChordType.DOMINANT_9)
        config = ArpConfig(tempo=110.0, octave_range=2, rate=0.25, seed=42)
        arp = Arpeggiator(chord, ArpPattern.RANDOM, config)
        events = arp.generate(num_cycles=8)

        self._events_to_track(events, track)

        output_path = self._get_output_dir("arpeggiator") / "arpeggiator_random.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_euclidean_tresillo(self):
        """Generate MIDI file with Euclidean tresillo rhythm (3,8)."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=100.0)
        track = seq.add_track("Euclidean Tresillo E(3,8)")

        config = EuclideanConfig(tempo=100.0, note_duration=0.25)
        euclid = EuclideanGenerator(pulses=3, steps=8, pitch=Note('C', 2), config=config)
        events = euclid.generate(cycles=8)

        self._events_to_track(events, track)

        output_path = self._get_output_dir("euclidean") / "euclidean_tresillo_3_8.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_euclidean_cinquillo(self):
        """Generate MIDI file with Euclidean cinquillo rhythm (5,8)."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=100.0)
        track = seq.add_track("Euclidean Cinquillo E(5,8)")

        config = EuclideanConfig(tempo=100.0, note_duration=0.25)
        euclid = EuclideanGenerator(pulses=5, steps=8, pitch=Note('D', 2), config=config)
        events = euclid.generate(cycles=8)

        self._events_to_track(events, track)

        output_path = self._get_output_dir("euclidean") / "euclidean_cinquillo_5_8.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_euclidean_7_12(self):
        """Generate MIDI file with Euclidean rhythm E(7,12)."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=120.0)
        track = seq.add_track("Euclidean E(7,12)")

        config = EuclideanConfig(tempo=120.0, note_duration=0.25)
        euclid = EuclideanGenerator(pulses=7, steps=12, pitch=Note('E', 2), config=config)
        events = euclid.generate(cycles=8)

        self._events_to_track(events, track)

        output_path = self._get_output_dir("euclidean") / "euclidean_7_12.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_markov_melody(self):
        """Generate MIDI file with Markov chain melody."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=90.0)
        track = seq.add_track("Markov Melody")

        # Train on a simple melodic pattern
        training_melody = [
            60, 62, 64, 65, 67, 65, 64, 62,  # C major ascending/descending
            60, 64, 67, 72, 67, 64, 60,      # Arpeggiated
            60, 62, 64, 62, 60, 59, 57, 55,  # Descending phrase
            60, 65, 64, 62, 60,              # Simple phrase
        ]

        config = MarkovConfig(tempo=90.0, note_duration=0.5, seed=42)
        markov = MarkovGenerator(config=config)
        markov.train(training_melody)
        events = markov.generate(num_notes=64, start_note=60)

        self._events_to_track(events, track)

        output_path = self._get_output_dir("markov") / "markov_melody.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_markov_constrained(self):
        """Generate MIDI file with scale-constrained Markov melody."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=100.0)
        track = seq.add_track("Markov Pentatonic")

        scale = Scale(Note('A', 4), ScaleType.MINOR_PENTATONIC)

        config = MarkovConfig(tempo=100.0, note_duration=0.375, seed=123)
        markov = MarkovGenerator(scale=scale, config=config)

        # Train with pentatonic intervals
        markov.set_transitions({
            69: {72: 0.4, 67: 0.3, 64: 0.2, 60: 0.1},  # A4
            72: {69: 0.3, 74: 0.4, 67: 0.3},          # C5
            74: {72: 0.5, 76: 0.3, 69: 0.2},          # D5
            76: {74: 0.4, 79: 0.3, 72: 0.3},          # E5
            79: {76: 0.5, 74: 0.3, 72: 0.2},          # G5
            67: {69: 0.5, 64: 0.3, 72: 0.2},          # G4
            64: {67: 0.4, 69: 0.4, 60: 0.2},          # E4
            60: {64: 0.5, 67: 0.3, 69: 0.2},          # C4
        })

        events = markov.generate(num_notes=48, start_note=69)
        self._events_to_track(events, track)

        output_path = self._get_output_dir("markov") / "markov_pentatonic.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_probabilistic_scale(self):
        """Generate MIDI file with probabilistic scale-based melody."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=120.0)
        track = seq.add_track("Probabilistic Scale")

        scale = Scale(Note('C', 4), ScaleType.DORIAN)
        config = ProbabilisticConfig(tempo=120.0, note_duration=0.25, seed=42)
        prob = ProbabilisticGenerator(scale, config=config)
        events = prob.generate(num_notes=64)

        self._events_to_track(events, track)

        output_path = self._get_output_dir("probabilistic") / "probabilistic_dorian.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_probabilistic_weighted(self):
        """Generate MIDI file with weighted probabilistic notes."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=100.0)
        track = seq.add_track("Probabilistic Weighted")

        # Weight toward chord tones with passing tones
        weights = {
            60: 5,   # C - root (high weight)
            62: 1,   # D - passing
            64: 4,   # E - third (high weight)
            65: 1,   # F - passing
            67: 4,   # G - fifth (high weight)
            69: 1,   # A - passing
            71: 3,   # B - seventh
            72: 3,   # C octave
        }

        config = ProbabilisticConfig(tempo=100.0, note_duration=0.375,
                                      rest_probability=0.1, seed=42)
        prob = ProbabilisticGenerator(weights=weights, config=config)
        events = prob.generate(num_notes=48)

        self._events_to_track(events, track)

        output_path = self._get_output_dir("probabilistic") / "probabilistic_weighted.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_sequence_pattern(self):
        """Generate MIDI file with step sequencer pattern."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=120.0)
        track = seq.add_track("Step Sequencer")

        config = SequenceConfig(tempo=120.0, step_duration=0.25, gate=0.8)
        sequencer = SequenceGenerator(steps=16, config=config)

        # Classic synth bass pattern
        pattern = [
            Step(pitch=36, velocity=120),   # Kick
            None,
            Step(pitch=36, velocity=80),
            None,
            Step(pitch=38, velocity=110),   # Snare
            None,
            Step(pitch=36, velocity=90),
            Step(pitch=36, velocity=70),
            Step(pitch=36, velocity=120),
            None,
            Step(pitch=36, velocity=80),
            Step(pitch=42, velocity=60),    # Hi-hat
            Step(pitch=38, velocity=110),
            Step(pitch=42, velocity=50),
            Step(pitch=36, velocity=90),
            Step(pitch=42, velocity=70),
        ]

        for i, step in enumerate(pattern):
            if step:
                sequencer.set_step(i, step)

        events = sequencer.generate(cycles=8)
        self._events_to_track(events, track)

        output_path = self._get_output_dir("sequence") / "sequence_drums.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_sequence_melodic(self):
        """Generate MIDI file with melodic step sequence."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=140.0)
        track = seq.add_track("Melodic Sequence")

        config = SequenceConfig(tempo=140.0, step_duration=0.125, gate=0.9)
        sequencer = SequenceGenerator(steps=16, config=config)

        # Synth lead pattern
        pitches = [60, 63, 67, 72, 70, 67, 63, 60,
                   58, 60, 63, 67, 70, 72, 75, 72]
        velocities = [100, 90, 110, 120, 100, 90, 80, 70,
                      80, 90, 100, 110, 120, 110, 127, 100]

        for i, (pitch, vel) in enumerate(zip(pitches, velocities)):
            sequencer.set_step(i, Step(pitch=pitch, velocity=vel))

        events = sequencer.generate(cycles=8)
        self._events_to_track(events, track)

        output_path = self._get_output_dir("sequence") / "sequence_melodic.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_melody_major(self):
        """Generate MIDI file with rule-based major melody."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=100.0)
        track = seq.add_track("Melody Major")

        scale = Scale(Note('C', 4), ScaleType.MAJOR)
        config = MelodyConfig(tempo=100.0, note_duration=0.5, max_jump=5,
                              phrase_length=8, rest_probability=0.15, seed=42)
        melody = MelodyGenerator(scale, config)
        events = melody.generate(num_notes=48, start_note=Note('C', 4))

        self._events_to_track(events, track)

        output_path = self._get_output_dir("melody") / "melody_major.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_melody_blues(self):
        """Generate MIDI file with blues scale melody."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=80.0)
        track = seq.add_track("Melody Blues")

        scale = Scale(Note('A', 3), ScaleType.BLUES)
        config = MelodyConfig(tempo=80.0, note_duration=0.375, max_jump=4,
                              phrase_length=6, rest_probability=0.2, seed=123,
                              humanize_timing=0.02, humanize_velocity=15)
        melody = MelodyGenerator(scale, config)
        events = melody.generate(num_notes=64, start_note=Note('A', 3))

        self._events_to_track(events, track)

        output_path = self._get_output_dir("melody") / "melody_blues.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_polyrhythm_3_4(self):
        """Generate MIDI file with 3:4 polyrhythm."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=120.0)
        track = seq.add_track("Polyrhythm 3:4")

        config = PolyrhythmConfig(tempo=120.0, note_duration=0.25)
        poly = PolyrhythmGenerator(cycle_beats=4, config=config)
        poly.add_layer(RhythmLayer(pulses=3, pitch=60, velocity=100))  # C4
        poly.add_layer(RhythmLayer(pulses=4, pitch=67, velocity=90))   # G4

        events = poly.generate(cycles=8)
        self._events_to_track(events, track)

        output_path = self._get_output_dir("polyrhythm") / "polyrhythm_3_4.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_polyrhythm_5_4_3(self):
        """Generate MIDI file with 5:4:3 polyrhythm."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=100.0)
        track = seq.add_track("Polyrhythm 5:4:3")

        config = PolyrhythmConfig(tempo=100.0, note_duration=0.2)
        poly = PolyrhythmGenerator(cycle_beats=4, config=config)
        poly.add_layer(RhythmLayer(pulses=5, pitch=36, velocity=120))  # Kick
        poly.add_layer(RhythmLayer(pulses=4, pitch=42, velocity=80))   # Hi-hat
        poly.add_layer(RhythmLayer(pulses=3, pitch=38, velocity=100))  # Snare

        events = poly.generate(cycles=8)
        self._events_to_track(events, track)

        output_path = self._get_output_dir("polyrhythm") / "polyrhythm_5_4_3.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_combined_arp_and_drums(self):
        """Generate MIDI file combining arpeggiator and Euclidean drums."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=120.0)

        # Arpeggio track
        arp_track = seq.add_track("Arpeggio")
        chord = Chord(Note('A', 3), ChordType.MINOR_7)
        arp_config = ArpConfig(tempo=120.0, octave_range=2, rate=0.25, channel=0)
        arp = Arpeggiator(chord, ArpPattern.UP_DOWN, arp_config)
        arp_events = arp.generate(num_cycles=8)
        self._events_to_track(arp_events, arp_track)

        # Drum track with Euclidean patterns
        drum_track = seq.add_track("Drums")

        # Kick E(4,16)
        kick_config = EuclideanConfig(tempo=120.0, note_duration=0.25, channel=9)
        kick = EuclideanGenerator(pulses=4, steps=16, pitch=36, config=kick_config)
        kick_events = kick.generate(cycles=4)
        self._events_to_track(kick_events, drum_track)

        # Snare E(4,16) with rotation
        snare_config = EuclideanConfig(tempo=120.0, note_duration=0.25, channel=9, rotation=4)
        snare = EuclideanGenerator(pulses=4, steps=16, pitch=38, config=snare_config)
        snare_events = snare.generate(cycles=4)
        self._events_to_track(snare_events, drum_track)

        # Hi-hat E(7,16)
        hh_config = EuclideanConfig(tempo=120.0, note_duration=0.125, channel=9)
        hh = EuclideanGenerator(pulses=7, steps=16, pitch=42, config=hh_config)
        hh_events = hh.generate(cycles=4)
        self._events_to_track(hh_events, drum_track)

        output_path = self._get_output_dir("combined") / "combined_arp_drums.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_chord_progression_arp(self):
        """Generate MIDI file with arpeggiated chord progression."""
        from coremusic.midi.utilities import MIDISequence
        from coremusic.music.theory import ChordProgression

        seq = MIDISequence(tempo=90.0)
        track = seq.add_track("Chord Progression")

        # ii-V-I-VI progression in C major
        prog = ChordProgression.from_numerals('C', ['ii', 'V', 'I', 'vi'])

        arp_config = ArpConfig(tempo=90.0, octave_range=1, rate=0.25)
        events = create_arp_from_progression(prog.chords, ArpPattern.UP,
                                              beats_per_chord=4, config=arp_config)
        self._events_to_track(events, track)

        output_path = self._get_output_dir("progression") / "progression_ii_V_I_vi.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_full_composition(self):
        """Generate MIDI file with multiple layers - a mini composition."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=110.0)

        # Bass track - Euclidean pattern
        bass_track = seq.add_track("Bass")
        bass_config = EuclideanConfig(tempo=110.0, note_duration=0.4, channel=1)
        bass = EuclideanGenerator(pulses=5, steps=16, pitch=Note('A', 1), config=bass_config)
        bass_events = bass.generate(cycles=8)
        self._events_to_track(bass_events, bass_track)

        # Pad track - Chord progression
        pad_track = seq.add_track("Pad")
        chords = [
            Chord(Note('A', 3), ChordType.MINOR_7),
            Chord(Note('D', 4), ChordType.MINOR_7),
            Chord(Note('G', 3), ChordType.MAJOR_7),
            Chord(Note('C', 4), ChordType.MAJOR_7),
        ]
        beat_duration = 60.0 / 110.0
        for i, chord in enumerate(chords * 2):  # 8 bars
            start_time = i * 4 * beat_duration
            for note in chord.get_notes():
                pad_track.add_note(start_time, note.midi, 70, 3.8 * beat_duration, channel=2)

        # Lead track - Markov melody
        lead_track = seq.add_track("Lead")
        scale = Scale(Note('A', 4), ScaleType.NATURAL_MINOR)
        markov_config = MarkovConfig(tempo=110.0, note_duration=0.25, seed=42, channel=3)
        markov = MarkovGenerator(scale=scale, config=markov_config)
        markov.train([69, 71, 72, 74, 76, 74, 72, 71, 69, 67, 65, 64, 65, 67, 69])
        lead_events = markov.generate(num_notes=64, start_note=69)
        self._events_to_track(lead_events, lead_track)

        # Arpeggio track
        arp_track = seq.add_track("Arpeggio")
        arp_chord = Chord(Note('A', 4), ChordType.MINOR)
        arp_config = ArpConfig(tempo=110.0, octave_range=2, rate=0.125,
                               swing=0.2, channel=4)
        arp = Arpeggiator(arp_chord, ArpPattern.RANDOM_WALK, arp_config)
        arp_events = arp.generate(duration=16 * beat_duration * 4)
        self._events_to_track(arp_events, arp_track)

        output_path = self._get_output_dir("composition") / "full_composition.mid"
        seq.save(str(output_path))

        assert output_path.exists()
        # Verify file has reasonable size
        assert output_path.stat().st_size > 500


# ============================================================================
# Bit Shift Register Tests
# ============================================================================


class TestBitShiftRegister:
    """Tests for BitShiftRegister class."""

    def test_create_register(self):
        """Test creating a shift register."""
        sr = BitShiftRegister(size=4)

        assert sr.size == 4
        assert len(sr) == 4
        assert sr.bits == [0, 0, 0, 0]

    def test_create_with_initial_state(self):
        """Test creating with initial state."""
        sr = BitShiftRegister(size=4, initial_state=[1, 0, 1, 0])

        assert sr.bits == [1, 0, 1, 0]

    def test_initial_state_wrong_size_raises(self):
        """Test that wrong-size initial state raises error."""
        with pytest.raises(ValueError, match="Initial state length"):
            BitShiftRegister(size=4, initial_state=[1, 0, 1])

    def test_size_must_be_positive(self):
        """Test that size must be at least 1."""
        with pytest.raises(ValueError, match="Size must be at least 1"):
            BitShiftRegister(size=0)

    def test_clock_basic(self):
        """Test basic clock operation."""
        sr = BitShiftRegister(size=4)

        # All zeros initially, output should be 0
        out1 = sr.clock(1)
        assert out1 == 0
        assert sr.bits == [1, 0, 0, 0]

        out2 = sr.clock(0)
        assert out2 == 0
        assert sr.bits == [0, 1, 0, 0]

        out3 = sr.clock(1)
        assert out3 == 0
        assert sr.bits == [1, 0, 1, 0]

        out4 = sr.clock(1)
        assert out4 == 0
        assert sr.bits == [1, 1, 0, 1]

        # Now the first 1 exits
        out5 = sr.clock(0)
        assert out5 == 1
        assert sr.bits == [0, 1, 1, 0]

    def test_clock_truthy_values(self):
        """Test that truthy values become 1."""
        sr = BitShiftRegister(size=2)

        sr.clock(True)
        assert sr.bits[0] == 1

        sr.clock(42)  # Truthy
        assert sr.bits[0] == 1

        sr.clock(None)  # Falsy
        assert sr.bits[0] == 0

    def test_peek(self):
        """Test peeking at bits."""
        sr = BitShiftRegister(size=4, initial_state=[1, 0, 1, 1])

        assert sr.peek(0) == 1
        assert sr.peek(1) == 0
        assert sr.peek(-1) == 1  # Last bit
        assert sr.peek(-2) == 1

    def test_reset(self):
        """Test reset to all zeros."""
        sr = BitShiftRegister(size=4, initial_state=[1, 1, 1, 1])
        sr.reset()

        assert sr.bits == [0, 0, 0, 0]

    def test_reset_with_pattern(self):
        """Test reset with new pattern."""
        sr = BitShiftRegister(size=4)
        sr.reset([1, 0, 1, 0])

        assert sr.bits == [1, 0, 1, 0]

    def test_reset_wrong_pattern_size_raises(self):
        """Test reset with wrong pattern size raises error."""
        sr = BitShiftRegister(size=4)

        with pytest.raises(ValueError, match="Pattern length"):
            sr.reset([1, 0, 1])

    def test_get_state(self):
        """Test getting state copy."""
        sr = BitShiftRegister(size=4, initial_state=[1, 0, 1, 0])
        state = sr.get_state()

        assert state == [1, 0, 1, 0]
        # Modifying returned state shouldn't affect register
        state[0] = 0
        assert sr.bits[0] == 1

    def test_set_state(self):
        """Test setting state directly."""
        sr = BitShiftRegister(size=4)
        sr.set_state([1, 1, 0, 0])

        assert sr.bits == [1, 1, 0, 0]

    def test_set_state_wrong_size_raises(self):
        """Test set_state with wrong size raises error."""
        sr = BitShiftRegister(size=4)

        with pytest.raises(ValueError, match="State length"):
            sr.set_state([1, 0])

    def test_repr_and_str(self):
        """Test string representations."""
        sr = BitShiftRegister(size=4, initial_state=[1, 0, 1, 1])

        assert repr(sr) == "1011"
        assert str(sr) == "1011"

    def test_example_from_spec(self):
        """Test the example from the original specification."""
        notes = [60, 62, 64, 67]
        gate_inputs = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0]

        sr = BitShiftRegister(size=4)
        results = []

        for step, gate_in in enumerate(gate_inputs):
            out_gate = sr.clock(gate_in)
            note_index = step % len(notes)
            note = notes[note_index]

            if out_gate == 1:
                results.append((step + 1, note, 'play'))
            else:
                results.append((step + 1, note, 'rest'))

        # First 4 steps should all be rests (register filling up)
        assert all(r[2] == 'rest' for r in results[:4])

        # After step 4, some plays should occur
        plays = [r for r in results[4:] if r[2] == 'play']
        assert len(plays) > 0


class TestBitShiftRegisterConfig:
    """Tests for BitShiftRegisterConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BitShiftRegisterConfig()

        assert config.step_duration == 0.25
        assert config.gate == 0.8
        assert config.velocity_mode == 'fixed'
        assert config.velocity == 100
        assert config.velocity_min == 64
        assert config.velocity_max == 127
        assert config.duration_mode == 'fixed'

    def test_velocity_mode_validation(self):
        """Test velocity_mode validation."""
        with pytest.raises(ValueError, match="velocity_mode"):
            BitShiftRegisterConfig(velocity_mode='invalid')

    def test_duration_mode_validation(self):
        """Test duration_mode validation."""
        with pytest.raises(ValueError, match="duration_mode"):
            BitShiftRegisterConfig(duration_mode='invalid')

    def test_velocity_range_validation(self):
        """Test velocity range validation."""
        with pytest.raises(ValueError, match="velocity_min"):
            BitShiftRegisterConfig(velocity_min=-1)

        with pytest.raises(ValueError, match="velocity_max"):
            BitShiftRegisterConfig(velocity_max=200)


class TestBitShiftRegisterGenerator:
    """Tests for BitShiftRegisterGenerator class."""

    def test_create_generator(self):
        """Test creating a generator."""
        gen = BitShiftRegisterGenerator(size=4)

        assert gen.register.size == 4
        assert gen.pitches == [60, 62, 64, 67]  # Default pitches

    def test_create_with_custom_pitches(self):
        """Test creating with custom pitches."""
        gen = BitShiftRegisterGenerator(size=4, pitches=[36, 38, 42, 46])

        assert gen.pitches == [36, 38, 42, 46]

    def test_create_with_note_objects(self):
        """Test creating with Note objects."""
        gen = BitShiftRegisterGenerator(
            size=4,
            pitches=[Note('C', 4), Note('E', 4), Note('G', 4)]
        )

        assert gen.pitches == [60, 64, 67]

    def test_create_with_initial_state(self):
        """Test creating with initial register state."""
        gen = BitShiftRegisterGenerator(size=4, initial_state=[1, 0, 1, 0])

        assert gen.register.get_state() == [1, 0, 1, 0]

    def test_empty_pitches_raises(self):
        """Test that empty pitches raises error."""
        with pytest.raises(ValueError, match="Must provide at least one pitch"):
            BitShiftRegisterGenerator(size=4, pitches=[])

    def test_set_pitches(self):
        """Test setting new pitches."""
        gen = BitShiftRegisterGenerator(size=4)
        gen.set_pitches([48, 50, 52])

        assert gen.pitches == [48, 50, 52]

    def test_set_pitches_empty_raises(self):
        """Test setting empty pitches raises error."""
        gen = BitShiftRegisterGenerator(size=4)

        with pytest.raises(ValueError, match="Must provide at least one pitch"):
            gen.set_pitches([])

    def test_reset(self):
        """Test reset method."""
        gen = BitShiftRegisterGenerator(size=4)

        # Clock a few times
        gen.clock_step(1, 0.0)
        gen.clock_step(1, 0.1)

        # Reset
        gen.reset()

        assert gen.register.get_state() == [0, 0, 0, 0]
        assert gen._step_counter == 0

    def test_reset_with_pattern(self):
        """Test reset with custom pattern."""
        gen = BitShiftRegisterGenerator(size=4)
        gen.reset([1, 1, 0, 0])

        assert gen.register.get_state() == [1, 1, 0, 0]

    def test_generate_with_gate_inputs(self):
        """Test generating with explicit gate inputs."""
        gen = BitShiftRegisterGenerator(size=4, pitches=[60, 62, 64, 67])
        gate_inputs = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0]

        events = gen.generate(gate_inputs=gate_inputs)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]

        # Should have some notes after register fills up
        assert len(note_ons) > 0

        # All notes should be from our pitch set
        for event in note_ons:
            assert event.data1 in [60, 62, 64, 67]

    def test_generate_with_num_steps(self):
        """Test generating with random gates."""
        config = BitShiftRegisterConfig(seed=42)
        gen = BitShiftRegisterGenerator(size=4, config=config)

        events = gen.generate(num_steps=20, gate_probability=0.5)

        note_ons = [e for e in events if e.status == MIDIStatus.NOTE_ON]

        # With random gates, should have some notes
        assert len(note_ons) > 0

    def test_generate_requires_gates_or_steps(self):
        """Test that generate requires either gate_inputs or num_steps."""
        gen = BitShiftRegisterGenerator(size=4)

        with pytest.raises(ValueError, match="Must provide either"):
            gen.generate()

    def test_clock_step(self):
        """Test single clock step processing."""
        gen = BitShiftRegisterGenerator(size=4, pitches=[60])
        gen.reset([0, 0, 0, 1])  # Put a 1 at the output position

        events, out_gate = gen.clock_step(0, 0.0)

        assert out_gate == 1
        assert events is not None
        assert len(events) == 2  # Note on and note off
        assert events[0].data1 == 60  # Pitch

    def test_clock_step_rest(self):
        """Test clock step that produces rest."""
        gen = BitShiftRegisterGenerator(size=4)
        gen.reset([0, 0, 0, 0])  # All zeros

        events, out_gate = gen.clock_step(1, 0.0)

        assert out_gate == 0
        assert events is None

    def test_velocity_mode_fixed(self):
        """Test fixed velocity mode."""
        config = BitShiftRegisterConfig(velocity=100, velocity_mode='fixed')
        gen = BitShiftRegisterGenerator(size=1, pitches=[60], config=config)

        gen.reset([1])  # Immediate output
        events, _ = gen.clock_step(0, 0.0)

        note_on = [e for e in events if e.is_note_on][0]
        assert note_on.data2 == 100

    def test_velocity_mode_random(self):
        """Test random velocity mode."""
        config = BitShiftRegisterConfig(
            velocity_mode='random',
            velocity_min=80,
            velocity_max=120,
            seed=42
        )
        gen = BitShiftRegisterGenerator(
            size=1,
            pitches=[60],
            initial_state=[1],
            config=config
        )

        velocities = []
        for i in range(10):
            gen.reset([1])
            events, _ = gen.clock_step(0, 0.0)
            note_on = [e for e in events if e.is_note_on][0]
            velocities.append(note_on.data2)

        # Check all velocities are in range
        for v in velocities:
            assert 80 <= v <= 120

    def test_velocity_mode_pattern(self):
        """Test pattern velocity mode."""
        config = BitShiftRegisterConfig(
            velocity_mode='pattern',
            velocity_pattern=[100, 80, 60, 120]
        )
        gen = BitShiftRegisterGenerator(
            size=1,
            pitches=[60],
            initial_state=[1],
            config=config
        )

        velocities = []
        for i in range(8):
            gen.register.set_state([1])
            events, _ = gen.clock_step(0, 0.0)
            note_on = [e for e in events if e.is_note_on][0]
            velocities.append(note_on.data2)

        # Should cycle through pattern
        assert velocities == [100, 80, 60, 120, 100, 80, 60, 120]

    def test_duration_mode_pattern(self):
        """Test pattern duration mode."""
        config = BitShiftRegisterConfig(
            duration_mode='pattern',
            duration_pattern=[1.0, 0.5, 0.25],
            step_duration=0.5,
            gate=1.0,  # Full gate for easier testing
            tempo=120.0
        )
        gen = BitShiftRegisterGenerator(
            size=1,
            pitches=[60],
            initial_state=[1],
            config=config
        )

        # Generate three notes
        durations = []
        beat_duration = 60.0 / 120.0  # 0.5 seconds per beat
        base_duration = 0.5 * beat_duration * 1.0  # step_duration * beat_duration * gate

        for i in range(3):
            gen.register.set_state([1])
            events, _ = gen.clock_step(0, 0.0)
            note_on = [e for e in events if e.is_note_on][0]
            note_off = [e for e in events if e.is_note_off][0]
            durations.append(note_off.time - note_on.time)

        # First should be full duration, second half, third quarter
        assert durations[0] == pytest.approx(base_duration * 1.0)
        assert durations[1] == pytest.approx(base_duration * 0.5)
        assert durations[2] == pytest.approx(base_duration * 0.25)

    def test_generate_with_trace(self):
        """Test generate_with_trace method."""
        gen = BitShiftRegisterGenerator(size=4, pitches=[60, 62, 64, 67])
        gate_inputs = [1, 0, 1, 1, 0, 1, 0, 0]

        events, trace = gen.generate_with_trace(gate_inputs)

        assert len(trace) == 8

        # Check trace structure
        for entry in trace:
            assert 'step' in entry
            assert 'input_gate' in entry
            assert 'register_state' in entry
            assert 'output_gate' in entry
            assert 'pitch' in entry
            assert 'velocity' in entry
            assert 'action' in entry

        # First 4 steps should be rests (register filling up)
        assert all(t['action'] == 'rest' for t in trace[:4])

    def test_pitches_cycle(self):
        """Test that pitches cycle through the sequence."""
        gen = BitShiftRegisterGenerator(
            size=1,
            pitches=[60, 62, 64],
            initial_state=[1]
        )

        pitches_played = []
        for i in range(6):
            gen.register.set_state([1])
            events, _ = gen.clock_step(0, 0.0)
            note_on = [e for e in events if e.is_note_on][0]
            pitches_played.append(note_on.data1)

        assert pitches_played == [60, 62, 64, 60, 62, 64]

    def test_repr(self):
        """Test string representation."""
        gen = BitShiftRegisterGenerator(size=4, pitches=[60, 62, 64])

        assert "size=4" in repr(gen)
        assert "pitches=3" in repr(gen)

    def test_swing_applied(self):
        """Test that swing is applied."""
        config = BitShiftRegisterConfig(
            tempo=120.0,
            swing=0.5,
            step_duration=0.5
        )
        gen = BitShiftRegisterGenerator(
            size=1,
            pitches=[60],
            initial_state=[1],
            config=config
        )

        # Collect times for a few steps
        times = []
        beat_duration = 60.0 / 120.0
        step_duration = 0.5 * beat_duration

        for i in range(4):
            gen.register.set_state([1])
            events, _ = gen.clock_step(0, i * step_duration)
            note_on = [e for e in events if e.is_note_on][0]
            times.append(note_on.time)

        # With swing, odd steps (index 1, 3) should be delayed
        # Step 1 should be later than expected
        expected_step1 = step_duration
        assert times[1] > expected_step1

    def test_reproducibility(self):
        """Test reproducibility with seed."""
        config = BitShiftRegisterConfig(seed=42)

        gen1 = BitShiftRegisterGenerator(size=4, config=config)
        events1 = gen1.generate(num_steps=20, gate_probability=0.5)
        pitches1 = [e.data1 for e in events1 if e.is_note_on]

        gen2 = BitShiftRegisterGenerator(size=4, config=BitShiftRegisterConfig(seed=42))
        events2 = gen2.generate(num_steps=20, gate_probability=0.5)
        pitches2 = [e.data1 for e in events2 if e.is_note_on]

        assert pitches1 == pitches2


class TestBitShiftRegisterMIDIFileGeneration:
    """Tests that generate MIDI files demonstrating bit shift register patterns."""

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Create output directory for MIDI files."""
        self.output_dir = Path(__file__).parent.parent / "build" / "midi_files" / "shift_register"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _events_to_track(self, events, track):
        """Helper to convert generator events to MIDITrack."""
        note_ons = {}
        for event in sorted(events, key=lambda e: e.time):
            if event.status == MIDIStatus.NOTE_ON and event.data2 > 0:
                key = (event.data1, event.channel)
                note_ons[key] = event
            elif event.status == MIDIStatus.NOTE_OFF or (event.status == MIDIStatus.NOTE_ON and event.data2 == 0):
                key = (event.data1, event.channel)
                if key in note_ons:
                    on_event = note_ons.pop(key)
                    duration = event.time - on_event.time
                    track.add_note(on_event.time, on_event.data1, on_event.data2,
                                   duration, on_event.channel)

    def test_generate_shift_register_basic(self):
        """Generate MIDI file with basic shift register pattern."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=120.0)
        track = seq.add_track("Shift Register Basic")

        config = BitShiftRegisterConfig(tempo=120.0, step_duration=0.25)
        gen = BitShiftRegisterGenerator(
            size=4,
            pitches=[60, 62, 64, 67],  # C, D, E, G
            config=config
        )

        # Use a repeating gate pattern
        gate_pattern = [1, 0, 1, 1, 0, 1, 0, 0] * 8  # 64 steps
        events = gen.generate(gate_inputs=gate_pattern)

        self._events_to_track(events, track)

        output_path = self.output_dir / "shift_register_basic.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_shift_register_variable_velocity(self):
        """Generate MIDI file with variable velocity."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=100.0)
        track = seq.add_track("Shift Register Variable Velocity")

        config = BitShiftRegisterConfig(
            tempo=100.0,
            step_duration=0.25,
            velocity_mode='pattern',
            velocity_pattern=[120, 80, 100, 60, 110, 70, 90, 50]
        )
        gen = BitShiftRegisterGenerator(
            size=4,
            pitches=[48, 52, 55, 60],  # C3, E3, G3, C4
            config=config
        )

        gate_pattern = [1, 1, 0, 1, 0, 1, 1, 0] * 8
        events = gen.generate(gate_inputs=gate_pattern)

        self._events_to_track(events, track)

        output_path = self.output_dir / "shift_register_velocity.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_shift_register_variable_duration(self):
        """Generate MIDI file with variable note duration."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=110.0)
        track = seq.add_track("Shift Register Variable Duration")

        config = BitShiftRegisterConfig(
            tempo=110.0,
            step_duration=0.25,
            duration_mode='pattern',
            duration_pattern=[1.0, 0.5, 0.75, 0.25, 1.0, 0.5]
        )
        gen = BitShiftRegisterGenerator(
            size=6,
            pitches=[36, 38, 42, 46, 49, 51],  # Drum-like pattern
            config=config
        )

        gate_pattern = [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1] * 6
        events = gen.generate(gate_inputs=gate_pattern)

        self._events_to_track(events, track)

        output_path = self.output_dir / "shift_register_duration.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_shift_register_random_gates(self):
        """Generate MIDI file with random gates."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=130.0)
        track = seq.add_track("Shift Register Random Gates")

        config = BitShiftRegisterConfig(
            tempo=130.0,
            step_duration=0.125,
            velocity_mode='random',
            velocity_min=70,
            velocity_max=127,
            seed=42
        )
        gen = BitShiftRegisterGenerator(
            size=8,
            pitches=[60, 63, 67, 70, 72, 75, 79, 82],  # Cm7 extended
            config=config
        )

        events = gen.generate(num_steps=128, gate_probability=0.6)

        self._events_to_track(events, track)

        output_path = self.output_dir / "shift_register_random.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_shift_register_with_swing(self):
        """Generate MIDI file with swing."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=95.0)
        track = seq.add_track("Shift Register Swing")

        config = BitShiftRegisterConfig(
            tempo=95.0,
            step_duration=0.25,
            swing=0.4,
            velocity_mode='pattern',
            velocity_pattern=[110, 70, 90, 60]
        )
        gen = BitShiftRegisterGenerator(
            size=4,
            pitches=[60, 64, 67, 72],  # C major arpeggio
            config=config
        )

        gate_pattern = [1, 1, 1, 1, 0, 1, 0, 1] * 8
        events = gen.generate(gate_inputs=gate_pattern)

        self._events_to_track(events, track)

        output_path = self.output_dir / "shift_register_swing.mid"
        seq.save(str(output_path))

        assert output_path.exists()

    def test_generate_shift_register_composition(self):
        """Generate MIDI file combining shift register with other generators."""
        from coremusic.midi.utilities import MIDISequence

        seq = MIDISequence(tempo=120.0)

        # Bass track - Euclidean pattern
        bass_track = seq.add_track("Bass")
        bass_config = EuclideanConfig(tempo=120.0, note_duration=0.3, channel=1)
        bass = EuclideanGenerator(pulses=5, steps=16, pitch=36, config=bass_config)
        bass_events = bass.generate(cycles=8)
        self._events_to_track(bass_events, bass_track)

        # Lead track - Shift Register
        lead_track = seq.add_track("Lead")
        lead_config = BitShiftRegisterConfig(
            tempo=120.0,
            step_duration=0.125,
            velocity_mode='pattern',
            velocity_pattern=[100, 80, 90, 70, 110, 75, 95, 65],
            channel=0
        )
        lead = BitShiftRegisterGenerator(
            size=8,
            pitches=[60, 62, 64, 65, 67, 69, 71, 72],  # C major scale
            config=lead_config
        )
        gate_pattern = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0] * 8
        lead_events = lead.generate(gate_inputs=gate_pattern)
        self._events_to_track(lead_events, lead_track)

        # Arp track
        arp_track = seq.add_track("Arp")
        arp_config = ArpConfig(tempo=120.0, rate=0.25, channel=2)
        chord = Chord(Note('C', 4), ChordType.MAJOR_7)
        arp = Arpeggiator(chord, ArpPattern.UP_DOWN, arp_config)
        arp_events = arp.generate(num_cycles=16)
        self._events_to_track(arp_events, arp_track)

        output_path = self.output_dir / "shift_register_composition.mid"
        seq.save(str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 500
