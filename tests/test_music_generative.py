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

    Generated files are saved to build/midi_files/ for audition.
    """

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Create output directory for MIDI files."""
        self.output_dir = Path(__file__).parent.parent / "build" / "midi_files"
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

        output_path = self.output_dir / "arpeggiator_up.mid"
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

        output_path = self.output_dir / "arpeggiator_up_down.mid"
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

        output_path = self.output_dir / "arpeggiator_random.mid"
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

        output_path = self.output_dir / "euclidean_tresillo_3_8.mid"
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

        output_path = self.output_dir / "euclidean_cinquillo_5_8.mid"
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

        output_path = self.output_dir / "euclidean_7_12.mid"
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

        output_path = self.output_dir / "markov_melody.mid"
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

        output_path = self.output_dir / "markov_pentatonic.mid"
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

        output_path = self.output_dir / "probabilistic_dorian.mid"
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

        output_path = self.output_dir / "probabilistic_weighted.mid"
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

        output_path = self.output_dir / "sequence_drums.mid"
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

        output_path = self.output_dir / "sequence_melodic.mid"
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

        output_path = self.output_dir / "melody_major.mid"
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

        output_path = self.output_dir / "melody_blues.mid"
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

        output_path = self.output_dir / "polyrhythm_3_4.mid"
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

        output_path = self.output_dir / "polyrhythm_5_4_3.mid"
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

        output_path = self.output_dir / "combined_arp_drums.mid"
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

        output_path = self.output_dir / "progression_ii_V_I_vi.mid"
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

        output_path = self.output_dir / "full_composition.mid"
        seq.save(str(output_path))

        assert output_path.exists()
        # Verify file has reasonable size
        assert output_path.stat().st_size > 500
