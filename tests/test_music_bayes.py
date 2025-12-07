#!/usr/bin/env python3
"""Tests for Bayesian network MIDI analysis and generation.

Tests cover all components of the bayes module including:
- NetworkConfig validation and serialization
- CPT (Conditional Probability Table) operations
- BayesianNetwork structure and training
- MIDIBayesAnalyzer for MIDI file analysis
- MIDIBayesGenerator for variant generation
- Utility functions
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from coremusic.music.bayes import (
    # Enums
    NetworkMode,
    StructureMode,
    # Data classes
    NoteObservation,
    NetworkConfig,
    # Core classes
    CPT,
    BayesianNetwork,
    # MIDI classes
    MIDIBayesAnalyzer,
    MIDIBayesGenerator,
    # Utility functions
    analyze_and_generate,
    merge_networks,
    network_statistics,
    # Variable names
    VAR_PITCH,
    VAR_DURATION,
    VAR_VELOCITY,
    VAR_IOI,
    var_pitch_lag,
    var_duration_lag,
)
from coremusic.midi.utilities import MIDISequence


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_observations():
    """Simple sequence of note observations."""
    return [
        NoteObservation(pitch=60, duration=0.5, velocity=100, ioi=0.0, time=0.0),
        NoteObservation(pitch=62, duration=0.5, velocity=90, ioi=0.5, time=0.5),
        NoteObservation(pitch=64, duration=0.5, velocity=80, ioi=0.5, time=1.0),
        NoteObservation(pitch=62, duration=0.5, velocity=90, ioi=0.5, time=1.5),
        NoteObservation(pitch=60, duration=0.5, velocity=100, ioi=0.5, time=2.0),
        NoteObservation(pitch=64, duration=0.5, velocity=80, ioi=0.5, time=2.5),
        NoteObservation(pitch=67, duration=0.5, velocity=70, ioi=0.5, time=3.0),
        NoteObservation(pitch=64, duration=0.5, velocity=80, ioi=0.5, time=3.5),
    ]


@pytest.fixture
def test_midi_file(tmp_path):
    """Create a test MIDI file."""
    seq = MIDISequence(tempo=120.0)
    track = seq.add_track("Test Track")

    # Add a simple melody
    notes = [60, 62, 64, 65, 67, 65, 64, 62, 60]
    for i, note in enumerate(notes):
        track.add_note(i * 0.5, note, 100, 0.4)

    filepath = tmp_path / "test.mid"
    seq.save(str(filepath))
    return filepath


# ============================================================================
# NoteObservation Tests
# ============================================================================


class TestNoteObservation:
    """Tests for NoteObservation dataclass."""

    def test_create_observation(self):
        """Create a basic observation."""
        obs = NoteObservation(pitch=60, duration=0.5, velocity=100)
        assert obs.pitch == 60
        assert obs.duration == 0.5
        assert obs.velocity == 100
        assert obs.ioi == 0.0
        assert obs.time == 0.0

    def test_observation_with_all_fields(self):
        """Create observation with all fields."""
        obs = NoteObservation(
            pitch=64,
            duration=0.25,
            velocity=80,
            ioi=0.5,
            time=1.0,
        )
        assert obs.pitch == 64
        assert obs.ioi == 0.5
        assert obs.time == 1.0

    def test_observation_serialization(self):
        """Test to_dict and from_dict."""
        obs = NoteObservation(pitch=60, duration=0.5, velocity=100, ioi=0.25, time=1.0)
        data = obs.to_dict()

        assert data['pitch'] == 60
        assert data['duration'] == 0.5
        assert data['velocity'] == 100
        assert data['ioi'] == 0.25
        assert data['time'] == 1.0

        restored = NoteObservation.from_dict(data)
        assert restored.pitch == obs.pitch
        assert restored.duration == obs.duration
        assert restored.velocity == obs.velocity


# ============================================================================
# NetworkConfig Tests
# ============================================================================


class TestNetworkConfig:
    """Tests for NetworkConfig dataclass."""

    def test_default_config(self):
        """Create config with defaults."""
        config = NetworkConfig()
        assert config.mode == NetworkMode.PITCH_DURATION_VELOCITY
        assert config.structure_mode == StructureMode.FIXED
        assert config.temporal_order == 1
        assert config.smoothing_alpha == 1.0

    def test_custom_config(self):
        """Create custom config."""
        config = NetworkConfig(
            mode=NetworkMode.FULL,
            structure_mode=StructureMode.LEARNED,
            temporal_order=2,
            smoothing_alpha=0.5,
        )
        assert config.mode == NetworkMode.FULL
        assert config.temporal_order == 2

    def test_invalid_temporal_order(self):
        """Reject invalid temporal order."""
        with pytest.raises(ValueError):
            NetworkConfig(temporal_order=0)

    def test_invalid_smoothing(self):
        """Reject negative smoothing."""
        with pytest.raises(ValueError):
            NetworkConfig(smoothing_alpha=-1.0)

    def test_config_serialization(self):
        """Test to_dict and from_dict."""
        config = NetworkConfig(
            mode=NetworkMode.FULL,
            temporal_order=2,
            pitch_bins=12,
        )
        data = config.to_dict()
        restored = NetworkConfig.from_dict(data)

        assert restored.mode == config.mode
        assert restored.temporal_order == config.temporal_order
        assert restored.pitch_bins == config.pitch_bins


# ============================================================================
# CPT Tests
# ============================================================================


class TestCPT:
    """Tests for Conditional Probability Table."""

    def test_create_cpt(self):
        """Create a basic CPT."""
        cpt = CPT("pitch", parents=("prev_pitch",), smoothing_alpha=1.0)
        assert cpt.variable == "pitch"
        assert cpt.parents == ("prev_pitch",)

    def test_observe_and_probability(self):
        """Test observation and probability calculation."""
        cpt = CPT("pitch", smoothing_alpha=0.0)

        # Observe values
        cpt.observe(60, ())
        cpt.observe(60, ())
        cpt.observe(62, ())

        # Check probabilities
        assert cpt.get_probability(60, ()) == pytest.approx(2/3)
        assert cpt.get_probability(62, ()) == pytest.approx(1/3)

    def test_observe_with_parents(self):
        """Test observations with parent conditioning."""
        cpt = CPT("pitch", parents=("prev_pitch",), smoothing_alpha=0.0)

        # Given prev_pitch=60, observe pitch=62 twice, pitch=64 once
        cpt.observe(62, (60,))
        cpt.observe(62, (60,))
        cpt.observe(64, (60,))

        # Given prev_pitch=62, observe pitch=64 twice
        cpt.observe(64, (62,))
        cpt.observe(64, (62,))

        # Check conditional probabilities
        assert cpt.get_probability(62, (60,)) == pytest.approx(2/3)
        assert cpt.get_probability(64, (60,)) == pytest.approx(1/3)
        assert cpt.get_probability(64, (62,)) == pytest.approx(1.0)

    def test_laplace_smoothing(self):
        """Test Laplace smoothing effect."""
        cpt = CPT("pitch", smoothing_alpha=1.0)

        cpt.observe(60, ())
        cpt.observe(62, ())

        # With smoothing, probabilities are adjusted
        prob_60 = cpt.get_probability(60, ())
        prob_62 = cpt.get_probability(62, ())

        assert prob_60 > 0
        assert prob_62 > 0
        assert prob_60 == prob_62  # Equal observations, equal smoothed probs

    def test_sample(self):
        """Test sampling from CPT."""
        cpt = CPT("pitch", smoothing_alpha=0.0)

        # Only one value observed
        cpt.observe(60, ())
        cpt.observe(60, ())

        # Should always sample 60
        for _ in range(10):
            assert cpt.sample(()) == 60

    def test_get_distribution(self):
        """Test getting full distribution."""
        cpt = CPT("pitch", smoothing_alpha=0.0)

        cpt.observe(60, ())
        cpt.observe(62, ())
        cpt.observe(64, ())

        dist = cpt.get_distribution(())
        assert len(dist) == 3
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_entropy(self):
        """Test entropy calculation."""
        cpt = CPT("pitch", smoothing_alpha=0.0)

        # Uniform distribution should have higher entropy
        cpt.observe(60, ())
        cpt.observe(62, ())
        cpt.observe(64, ())

        uniform_entropy = cpt.get_entropy(())

        # Now add more observations to make distribution less uniform
        cpt2 = CPT("pitch", smoothing_alpha=0.0)
        cpt2.observe(60, ())
        cpt2.observe(60, ())
        cpt2.observe(60, ())
        cpt2.observe(62, ())

        skewed_entropy = cpt2.get_entropy(())

        assert uniform_entropy > skewed_entropy

    def test_cpt_serialization(self):
        """Test CPT serialization."""
        cpt = CPT("pitch", parents=("prev_pitch",), smoothing_alpha=0.5)
        cpt.observe(62, (60,))
        cpt.observe(64, (60,))
        cpt.observe(64, (62,))

        data = cpt.to_dict()
        restored = CPT.from_dict(data)

        assert restored.variable == cpt.variable
        assert restored.parents == cpt.parents
        assert restored.get_probability(62, (60,)) == cpt.get_probability(62, (60,))


# ============================================================================
# BayesianNetwork Tests
# ============================================================================


class TestBayesianNetwork:
    """Tests for BayesianNetwork."""

    def test_create_network(self):
        """Create a basic network."""
        network = BayesianNetwork()
        assert network.config.mode == NetworkMode.PITCH_DURATION_VELOCITY
        assert len(network.get_variables()) == 0

    def test_add_variable(self):
        """Add variables to network."""
        network = BayesianNetwork()
        network.add_variable("pitch")
        network.add_variable("duration")

        assert "pitch" in network.get_variables()
        assert "duration" in network.get_variables()

    def test_add_edge(self):
        """Add edges between variables."""
        network = BayesianNetwork()
        network.add_variable("prev_pitch")
        network.add_variable("pitch")
        network.add_edge("prev_pitch", "pitch")

        assert ("prev_pitch", "pitch") in network.get_edges()
        assert network.get_parents("pitch") == ["prev_pitch"]
        assert network.get_children("prev_pitch") == ["pitch"]

    def test_edge_auto_adds_variables(self):
        """Adding edge automatically adds variables."""
        network = BayesianNetwork()
        network.add_edge("prev_pitch", "pitch")

        assert "prev_pitch" in network.get_variables()
        assert "pitch" in network.get_variables()

    def test_cycle_detection(self):
        """Detect and reject cycles."""
        network = BayesianNetwork()
        network.add_edge("A", "B")
        network.add_edge("B", "C")

        with pytest.raises(ValueError, match="cycle"):
            network.add_edge("C", "A")

    def test_remove_edge(self):
        """Remove an edge."""
        network = BayesianNetwork()
        network.add_edge("prev_pitch", "pitch")
        network.remove_edge("prev_pitch", "pitch")

        assert ("prev_pitch", "pitch") not in network.get_edges()
        assert network.get_parents("pitch") == []

    def test_observe(self):
        """Record observations."""
        network = BayesianNetwork()
        network.add_edge("prev_pitch", "pitch")

        network.observe({"prev_pitch": 60, "pitch": 62})
        network.observe({"prev_pitch": 60, "pitch": 64})
        network.observe({"prev_pitch": 62, "pitch": 64})

        assert network.get_num_observations() == 3

    def test_train(self, simple_observations):
        """Train network from observations."""
        config = NetworkConfig(mode=NetworkMode.PITCH_ONLY, temporal_order=1)
        network = BayesianNetwork(config)
        network.train(simple_observations)

        assert network.get_num_observations() > 0
        assert VAR_PITCH in network.get_variables()

    def test_sample(self, simple_observations):
        """Sample from trained network."""
        config = NetworkConfig(mode=NetworkMode.PITCH_ONLY, temporal_order=1)
        network = BayesianNetwork(config)
        network.train(simple_observations)

        sample = network.sample()
        assert VAR_PITCH in sample

    def test_sample_with_evidence(self, simple_observations):
        """Sample with evidence."""
        config = NetworkConfig(mode=NetworkMode.PITCH_ONLY, temporal_order=1)
        network = BayesianNetwork(config)
        network.train(simple_observations)

        evidence = {var_pitch_lag(1): 60}
        sample = network.sample(evidence)

        assert var_pitch_lag(1) in sample
        assert sample[var_pitch_lag(1)] == 60

    def test_get_probability(self, simple_observations):
        """Get probability of specific value."""
        config = NetworkConfig(mode=NetworkMode.PITCH_ONLY, temporal_order=1)
        network = BayesianNetwork(config)
        network.train(simple_observations)

        # Should be able to get probability
        prob = network.get_probability(VAR_PITCH, 60, {})
        assert 0 <= prob <= 1

    def test_get_distribution(self, simple_observations):
        """Get full distribution."""
        config = NetworkConfig(mode=NetworkMode.PITCH_ONLY, temporal_order=1)
        network = BayesianNetwork(config)
        network.train(simple_observations)

        dist = network.get_distribution(VAR_PITCH, {})
        assert len(dist) > 0
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_fixed_structure_pitch_only(self):
        """Test fixed structure for pitch-only mode."""
        config = NetworkConfig(mode=NetworkMode.PITCH_ONLY, temporal_order=1)
        network = BayesianNetwork(config)
        network._setup_fixed_structure()

        assert VAR_PITCH in network.get_variables()
        assert var_pitch_lag(1) in network.get_variables()
        assert VAR_DURATION not in network.get_variables()

    def test_fixed_structure_full(self):
        """Test fixed structure for full mode."""
        config = NetworkConfig(mode=NetworkMode.FULL, temporal_order=1)
        network = BayesianNetwork(config)
        network._setup_fixed_structure()

        assert VAR_PITCH in network.get_variables()
        assert VAR_DURATION in network.get_variables()
        assert VAR_VELOCITY in network.get_variables()
        assert VAR_IOI in network.get_variables()

    def test_network_serialization(self, simple_observations):
        """Test network serialization."""
        config = NetworkConfig(mode=NetworkMode.PITCH_DURATION, temporal_order=1)
        network = BayesianNetwork(config)
        network.train(simple_observations)

        data = network.to_dict()
        restored = BayesianNetwork.from_dict(data)

        assert restored.config.mode == network.config.mode
        assert len(restored.get_variables()) == len(network.get_variables())
        assert len(restored.get_edges()) == len(network.get_edges())

    def test_network_json_serialization(self, simple_observations):
        """Test JSON serialization."""
        config = NetworkConfig(mode=NetworkMode.PITCH_ONLY, temporal_order=1)
        network = BayesianNetwork(config)
        network.train(simple_observations)

        json_str = network.to_json()
        restored = BayesianNetwork.from_json(json_str)

        assert restored.config.mode == network.config.mode
        assert restored.get_num_observations() == network.get_num_observations()

    def test_network_save_load(self, simple_observations, tmp_path):
        """Test save and load."""
        config = NetworkConfig(mode=NetworkMode.PITCH_ONLY, temporal_order=1)
        network = BayesianNetwork(config)
        network.train(simple_observations)

        filepath = tmp_path / "network.json"
        network.save(filepath)

        loaded = BayesianNetwork.load(filepath)
        assert loaded.get_num_observations() == network.get_num_observations()

    def test_load_nonexistent_file(self, tmp_path):
        """Raise error for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            BayesianNetwork.load(tmp_path / "nonexistent.json")

    def test_discretization_pitch(self):
        """Test pitch discretization."""
        config = NetworkConfig(pitch_bins=12)  # Octave-based binning
        network = BayesianNetwork(config)

        binned = network._discretize_pitch(60)
        restored = network._undiscretize_pitch(binned)

        # Should be approximately equal (binning loses precision)
        assert abs(restored - 60) < 12

    def test_discretization_duration(self):
        """Test duration discretization."""
        network = BayesianNetwork()

        # 0.5 beats = 2 sixteenths
        binned = network._discretize_duration(0.5)
        restored = network._undiscretize_duration(binned)

        assert abs(restored - 0.5) < 0.25

    def test_entropy(self, simple_observations):
        """Test entropy calculation."""
        config = NetworkConfig(mode=NetworkMode.PITCH_ONLY, temporal_order=1)
        network = BayesianNetwork(config)
        network.train(simple_observations)

        entropy = network.get_entropy(VAR_PITCH)
        assert entropy >= 0

    def test_average_entropy(self, simple_observations):
        """Test average entropy."""
        config = NetworkConfig(mode=NetworkMode.PITCH_ONLY, temporal_order=1)
        network = BayesianNetwork(config)
        network.train(simple_observations)

        avg_entropy = network.get_average_entropy()
        assert avg_entropy >= 0

    def test_repr(self):
        """Test string representation."""
        network = BayesianNetwork()
        network.add_edge("prev_pitch", "pitch")
        network.observe({"prev_pitch": 60, "pitch": 62})

        repr_str = repr(network)
        assert "BayesianNetwork" in repr_str
        assert "variables=2" in repr_str


# ============================================================================
# MIDIBayesAnalyzer Tests
# ============================================================================


class TestMIDIBayesAnalyzer:
    """Tests for MIDIBayesAnalyzer."""

    def test_create_analyzer(self):
        """Create analyzer with defaults."""
        analyzer = MIDIBayesAnalyzer()
        assert analyzer.config.mode == NetworkMode.PITCH_DURATION_VELOCITY

    def test_create_analyzer_custom_config(self):
        """Create analyzer with custom config."""
        config = NetworkConfig(mode=NetworkMode.FULL, temporal_order=2)
        analyzer = MIDIBayesAnalyzer(config=config)
        assert analyzer.config.mode == NetworkMode.FULL
        assert analyzer.config.temporal_order == 2

    def test_analyze_file(self, test_midi_file):
        """Analyze a MIDI file."""
        analyzer = MIDIBayesAnalyzer()
        # Track index 1 because Format 1 MIDI has tempo track at index 0
        network = analyzer.analyze_file(test_midi_file, track_index=1)

        assert network.get_num_observations() > 0
        assert VAR_PITCH in network.get_variables()

    def test_analyze_track(self, test_midi_file):
        """Analyze a specific track."""
        analyzer = MIDIBayesAnalyzer()
        sequence = MIDISequence.load(str(test_midi_file))
        # Track index 1 because Format 1 MIDI has tempo track at index 0
        network = analyzer.analyze_track(sequence, track_index=1)

        assert network.get_num_observations() > 0

    def test_analyze_all_tracks(self, test_midi_file):
        """Analyze all tracks."""
        analyzer = MIDIBayesAnalyzer()
        networks = analyzer.analyze_all_tracks(test_midi_file)

        assert len(networks) >= 1

    def test_analyze_invalid_track(self, test_midi_file):
        """Raise error for invalid track index."""
        analyzer = MIDIBayesAnalyzer()
        with pytest.raises(ValueError):
            analyzer.analyze_file(test_midi_file, track_index=100)

    def test_analyze_pitch_only(self, test_midi_file):
        """Analyze with pitch-only mode."""
        analyzer = MIDIBayesAnalyzer(mode=NetworkMode.PITCH_ONLY)
        network = analyzer.analyze_file(test_midi_file, track_index=1)

        assert VAR_PITCH in network.get_variables()
        assert VAR_DURATION not in network.get_variables()

    def test_analyze_full_mode(self, test_midi_file):
        """Analyze with full mode."""
        analyzer = MIDIBayesAnalyzer(mode=NetworkMode.FULL)
        network = analyzer.analyze_file(test_midi_file, track_index=1)

        assert VAR_PITCH in network.get_variables()
        assert VAR_DURATION in network.get_variables()
        assert VAR_VELOCITY in network.get_variables()
        assert VAR_IOI in network.get_variables()

    def test_analyze_higher_order(self, test_midi_file):
        """Analyze with higher temporal order."""
        analyzer = MIDIBayesAnalyzer(temporal_order=2)
        network = analyzer.analyze_file(test_midi_file, track_index=1)

        assert var_pitch_lag(1) in network.get_variables()
        assert var_pitch_lag(2) in network.get_variables()


# ============================================================================
# MIDIBayesGenerator Tests
# ============================================================================


class TestMIDIBayesGenerator:
    """Tests for MIDIBayesGenerator."""

    def test_create_generator(self, test_midi_file):
        """Create generator from network."""
        analyzer = MIDIBayesAnalyzer()
        network = analyzer.analyze_file(test_midi_file, track_index=1)
        generator = MIDIBayesGenerator(network)

        assert generator.network == network

    def test_generate_sequence(self, test_midi_file):
        """Generate a sequence."""
        analyzer = MIDIBayesAnalyzer()
        network = analyzer.analyze_file(test_midi_file, track_index=1)
        generator = MIDIBayesGenerator(network)

        sequence = generator.generate(num_notes=16, tempo=120.0)

        assert len(sequence.tracks) == 1
        note_ons = [e for e in sequence.tracks[0].events if e.is_note_on]
        assert len(note_ons) == 16

    def test_generate_with_start_pitch(self, test_midi_file):
        """Generate with specific start pitch."""
        analyzer = MIDIBayesAnalyzer()
        network = analyzer.analyze_file(test_midi_file, track_index=1)
        generator = MIDIBayesGenerator(network)

        sequence = generator.generate(num_notes=8, start_pitch=60)

        # First note should be influenced by start pitch
        note_ons = [e for e in sequence.tracks[0].events if e.is_note_on]
        assert len(note_ons) == 8

    def test_generate_reproducibility(self, test_midi_file):
        """Test reproducibility with seed."""
        config = NetworkConfig(seed=42)
        analyzer = MIDIBayesAnalyzer(config=config)
        network = analyzer.analyze_file(test_midi_file, track_index=1)

        gen1 = MIDIBayesGenerator(network)
        seq1 = gen1.generate(num_notes=16)

        # Reset seed
        network._rng.seed(42)
        gen2 = MIDIBayesGenerator(network)
        gen2._rng.seed(42)
        seq2 = gen2.generate(num_notes=16)

        notes1 = [e.data1 for e in seq1.tracks[0].events if e.is_note_on]
        notes2 = [e.data1 for e in seq2.tracks[0].events if e.is_note_on]

        assert notes1 == notes2

    def test_generate_to_track(self, test_midi_file):
        """Generate into existing track."""
        analyzer = MIDIBayesAnalyzer()
        network = analyzer.analyze_file(test_midi_file, track_index=1)
        generator = MIDIBayesGenerator(network)

        sequence = MIDISequence(tempo=120.0)
        track = sequence.add_track("Test")
        track.add_note(0.0, 60, 100, 0.5)  # Existing note

        generator.generate_to_track(track, num_notes=8, start_time=1.0, tempo=120.0)

        note_ons = [e for e in track.events if e.is_note_on]
        assert len(note_ons) == 9  # 1 original + 8 generated

    def test_generate_from_empty_network(self):
        """Generate from empty network returns empty sequence."""
        network = BayesianNetwork()
        generator = MIDIBayesGenerator(network)

        sequence = generator.generate(num_notes=16)

        note_ons = [e for e in sequence.tracks[0].events if e.is_note_on]
        assert len(note_ons) == 0


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_analyze_and_generate(self, test_midi_file, tmp_path):
        """Test convenience function."""
        output_file = tmp_path / "output.mid"

        sequence = analyze_and_generate(
            test_midi_file,
            output_file,
            num_notes=16,
            mode=NetworkMode.PITCH_ONLY,
            track_index=1,
        )

        assert output_file.exists()
        note_ons = [e for e in sequence.tracks[0].events if e.is_note_on]
        assert len(note_ons) == 16

    def test_merge_networks(self, test_midi_file):
        """Test merging networks."""
        analyzer = MIDIBayesAnalyzer()

        # Create two networks from same file (simulating different sources)
        network1 = analyzer.analyze_file(test_midi_file, track_index=1)
        network2 = analyzer.analyze_file(test_midi_file, track_index=1)

        merged = merge_networks([network1, network2])

        assert len(merged.get_variables()) == len(network1.get_variables())

    def test_merge_networks_with_weights(self, test_midi_file):
        """Test merging with weights."""
        analyzer = MIDIBayesAnalyzer()

        network1 = analyzer.analyze_file(test_midi_file, track_index=1)
        network2 = analyzer.analyze_file(test_midi_file, track_index=1)

        merged = merge_networks([network1, network2], weights=[0.7, 0.3])

        assert merged is not None

    def test_merge_networks_empty(self):
        """Raise error for empty list."""
        with pytest.raises(ValueError):
            merge_networks([])

    def test_network_statistics(self, test_midi_file):
        """Test statistics function."""
        analyzer = MIDIBayesAnalyzer()
        network = analyzer.analyze_file(test_midi_file, track_index=1)

        stats = network_statistics(network)

        assert 'mode' in stats
        assert 'num_variables' in stats
        assert 'num_edges' in stats
        assert 'num_observations' in stats
        assert 'average_entropy' in stats


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow(self, test_midi_file, tmp_path):
        """Test complete analyze-generate workflow."""
        # Analyze
        config = NetworkConfig(mode=NetworkMode.PITCH_DURATION, temporal_order=1)
        analyzer = MIDIBayesAnalyzer(config=config)
        network = analyzer.analyze_file(test_midi_file, track_index=1)

        # Check network
        assert network.get_num_observations() > 0
        assert VAR_PITCH in network.get_variables()
        assert VAR_DURATION in network.get_variables()

        # Generate
        generator = MIDIBayesGenerator(network)
        sequence = generator.generate(num_notes=32, tempo=120.0)

        # Save and verify
        output_file = tmp_path / "variation.mid"
        sequence.save(str(output_file))

        assert output_file.exists()

        # Load and verify - check track 1 since track 0 is tempo track in Format 1
        loaded = MIDISequence.load(str(output_file))
        # Find track with notes
        note_ons = []
        for track in loaded.tracks:
            track_notes = [e for e in track.events if e.is_note_on]
            if track_notes:
                note_ons = track_notes
                break
        assert len(note_ons) == 32

    def test_save_and_reload_network(self, test_midi_file, tmp_path):
        """Test saving and reloading network."""
        # Train network
        analyzer = MIDIBayesAnalyzer()
        network = analyzer.analyze_file(test_midi_file, track_index=1)

        # Save
        network_file = tmp_path / "network.json"
        network.save(network_file)

        # Reload
        loaded = BayesianNetwork.load(network_file)

        # Generate from reloaded
        generator = MIDIBayesGenerator(loaded)
        sequence = generator.generate(num_notes=16)

        note_ons = [e for e in sequence.tracks[0].events if e.is_note_on]
        assert len(note_ons) == 16


# ============================================================================
# MIDI File Generation Tests
# ============================================================================


class TestMIDIFileGeneration:
    """Tests that generate actual MIDI files for audition."""

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Create output directory."""
        self.output_dir = Path(__file__).parent.parent / "build" / "midi_files" / "bayes"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_generate_variation_basic(self, test_midi_file):
        """Generate basic variation."""
        analyzer = MIDIBayesAnalyzer()
        network = analyzer.analyze_file(test_midi_file, track_index=1)
        generator = MIDIBayesGenerator(network)

        sequence = generator.generate(num_notes=32, tempo=120.0)

        output_file = self.output_dir / "bayes_variation_basic.mid"
        sequence.save(str(output_file))
        assert output_file.exists()

    def test_generate_variation_full_mode(self, test_midi_file):
        """Generate with full modeling."""
        config = NetworkConfig(mode=NetworkMode.FULL, temporal_order=1)
        analyzer = MIDIBayesAnalyzer(config=config)
        network = analyzer.analyze_file(test_midi_file, track_index=1)
        generator = MIDIBayesGenerator(network)

        sequence = generator.generate(num_notes=32, tempo=120.0)

        output_file = self.output_dir / "bayes_variation_full.mid"
        sequence.save(str(output_file))
        assert output_file.exists()

    def test_generate_variation_second_order(self, test_midi_file):
        """Generate with second-order network."""
        config = NetworkConfig(mode=NetworkMode.PITCH_DURATION, temporal_order=2)
        analyzer = MIDIBayesAnalyzer(config=config)
        network = analyzer.analyze_file(test_midi_file, track_index=1)
        generator = MIDIBayesGenerator(network)

        sequence = generator.generate(num_notes=32, tempo=120.0)

        output_file = self.output_dir / "bayes_variation_order2.mid"
        sequence.save(str(output_file))
        assert output_file.exists()


# ============================================================================
# Variable Name Helper Tests
# ============================================================================


class TestVariableNames:
    """Tests for variable name helpers."""

    def test_var_pitch_lag(self):
        """Test pitch lag variable names."""
        assert var_pitch_lag(0) == VAR_PITCH
        assert var_pitch_lag(1) == "pitch_lag1"
        assert var_pitch_lag(2) == "pitch_lag2"

    def test_var_duration_lag(self):
        """Test duration lag variable names."""
        assert var_duration_lag(0) == VAR_DURATION
        assert var_duration_lag(1) == "duration_lag1"
        assert var_duration_lag(2) == "duration_lag2"
