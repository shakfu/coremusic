#!/usr/bin/env python3
"""Tests for neural network training with classical MIDI files.

These tests use the classical MIDI files in tests/data/midi/classical/
to train neural network models and generate new music.

Output MIDI files are saved to build/midi_files/neural/
"""

import os
import glob
import tempfile
import pytest

from conftest import CLASSICAL_DIR

# Import neural network components
from coremusic.music.neural import (
    NoteEncoder,
    EventEncoder,
    MIDIDataset,
    ModelFactory,
    TrainingConfig,
    Trainer,
    MusicGenerator,
    TemperatureSampling,
    train_music_model,
    generate_music,
)
from coremusic.kann import NeuralNetwork

# Output directory for generated MIDI files
BUILD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
OUTPUT_DIR = os.path.join(BUILD_DIR, "midi_files", "neural")


def ensure_output_dir():
    """Ensure the output directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def is_valid_midi_file(filepath: str) -> bool:
    """Check if a file is a valid Standard MIDI File."""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4)
            return header == b'MThd'
    except Exception:
        return False


def get_classical_files(count: int = 3) -> list:
    """Get a subset of classical MIDI files for testing."""
    if not os.path.exists(CLASSICAL_DIR):
        pytest.skip(f"Classical MIDI directory not found: {CLASSICAL_DIR}")

    # Get all MIDI files and filter to valid ones
    all_files = sorted(glob.glob(os.path.join(CLASSICAL_DIR, "*.mid")))
    files = [f for f in all_files if is_valid_midi_file(f)]

    if len(files) < count:
        pytest.skip(f"Not enough valid MIDI files (need {count}, found {len(files)})")

    # Return smaller files first for faster tests
    files_with_size = [(f, os.path.getsize(f)) for f in files]
    files_with_size.sort(key=lambda x: x[1])
    return [f[0] for f in files_with_size[:count]]


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def classical_files_small():
    """Get 2 small classical MIDI files for quick tests."""
    return get_classical_files(2)


@pytest.fixture
def classical_files_medium():
    """Get 4 classical MIDI files for medium tests."""
    return get_classical_files(4)


@pytest.fixture
def output_dir():
    """Ensure output directory exists and return path."""
    return ensure_output_dir()


# ============================================================================
# MLP Model Tests
# ============================================================================


class TestMLPTraining:
    """Test MLP model training with classical MIDI files."""

    def test_mlp_train_and_generate_note_encoder(self, classical_files_small, output_dir):
        """Train MLP with NoteEncoder and generate music."""
        # Setup
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=16)

        # Load files
        for midi_file in classical_files_small:
            dataset.load_file(midi_file)

        assert dataset.n_sequences > 0, "No sequences loaded"

        # Create model
        model = ModelFactory.create(
            model_type="mlp",
            encoder=encoder,
            seq_length=16,
            hidden_size=64,
        )

        # Prepare training data
        x_train, y_train = dataset.prepare_training_data(use_numpy=True)

        # Train (quick, few epochs)
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=32,
            max_epochs=5,
            min_epochs=1,
            validation_split=0.0,
            verbose=0,
        )
        trainer = Trainer(model, config)
        trainer.train(x_train, y_train)

        # Generate music
        generator = MusicGenerator(
            model, encoder, seq_length=16,
            sampling=TemperatureSampling(0.8)
        )

        output_path = os.path.join(output_dir, "test_mlp_note.mid")
        sequence = generator.generate_midi(
            duration_beats=16,
            tempo=120.0,
            track_name="MLP Generated"
        )
        sequence.save(output_path)

        assert os.path.exists(output_path), "Output MIDI file not created"
        assert os.path.getsize(output_path) > 0, "Output MIDI file is empty"

    def test_mlp_train_and_generate_event_encoder(self, classical_files_small, output_dir):
        """Train MLP with EventEncoder and generate music."""
        # Setup
        encoder = EventEncoder()
        dataset = MIDIDataset(encoder, seq_length=16)

        # Load files
        for midi_file in classical_files_small:
            dataset.load_file(midi_file)

        assert dataset.n_sequences > 0, "No sequences loaded"

        # Create model
        model = ModelFactory.create(
            model_type="mlp",
            encoder=encoder,
            seq_length=16,
            hidden_size=64,
        )

        # Prepare training data
        x_train, y_train = dataset.prepare_training_data(use_numpy=True)

        # Train
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=32,
            max_epochs=5,
            min_epochs=1,
            validation_split=0.0,
            verbose=0,
        )
        trainer = Trainer(model, config)
        trainer.train(x_train, y_train)

        # Generate music
        generator = MusicGenerator(
            model, encoder, seq_length=16,
            sampling=TemperatureSampling(0.8)
        )

        output_path = os.path.join(output_dir, "test_mlp_event.mid")
        sequence = generator.generate_midi(
            duration_beats=16,
            tempo=120.0,
            track_name="MLP Event Generated"
        )
        sequence.save(output_path)

        assert os.path.exists(output_path), "Output MIDI file not created"


# ============================================================================
# LSTM Model Tests
# ============================================================================


class TestLSTMTraining:
    """Test LSTM model training with classical MIDI files."""

    def test_lstm_train_and_generate(self, classical_files_small, output_dir):
        """Train LSTM model and generate music."""
        # Setup
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=16)

        # Load files
        for midi_file in classical_files_small:
            dataset.load_file(midi_file)

        # Create LSTM model
        model = ModelFactory.create(
            model_type="lstm",
            encoder=encoder,
            seq_length=16,
            hidden_size=32,
        )

        # Prepare training data
        x_train, y_train = dataset.prepare_training_data(use_numpy=True)

        # Train
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=32,
            max_epochs=3,
            min_epochs=1,
            validation_split=0.0,
            verbose=0,
        )
        trainer = Trainer(model, config)
        trainer.train(x_train, y_train)

        # Generate music
        generator = MusicGenerator(
            model, encoder, seq_length=16,
            sampling=TemperatureSampling(0.8)
        )

        output_path = os.path.join(output_dir, "test_lstm.mid")
        sequence = generator.generate_midi(
            duration_beats=16,
            tempo=120.0,
            track_name="LSTM Generated"
        )
        sequence.save(output_path)

        assert os.path.exists(output_path), "Output MIDI file not created"


# ============================================================================
# GRU Model Tests
# ============================================================================


class TestGRUTraining:
    """Test GRU model training with classical MIDI files."""

    def test_gru_train_and_generate(self, classical_files_small, output_dir):
        """Train GRU model and generate music."""
        # Setup
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=16)

        # Load files
        for midi_file in classical_files_small:
            dataset.load_file(midi_file)

        # Create GRU model
        model = ModelFactory.create(
            model_type="gru",
            encoder=encoder,
            seq_length=16,
            hidden_size=32,
        )

        # Prepare training data
        x_train, y_train = dataset.prepare_training_data(use_numpy=True)

        # Train
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=32,
            max_epochs=3,
            min_epochs=1,
            validation_split=0.0,
            verbose=0,
        )
        trainer = Trainer(model, config)
        trainer.train(x_train, y_train)

        # Generate music
        generator = MusicGenerator(
            model, encoder, seq_length=16,
            sampling=TemperatureSampling(0.8)
        )

        output_path = os.path.join(output_dir, "test_gru.mid")
        sequence = generator.generate_midi(
            duration_beats=16,
            tempo=120.0,
            track_name="GRU Generated"
        )
        sequence.save(output_path)

        assert os.path.exists(output_path), "Output MIDI file not created"


# ============================================================================
# High-Level API Tests
# ============================================================================


class TestHighLevelAPI:
    """Test high-level training and generation API."""

    def test_train_music_model_api(self, classical_files_small, output_dir):
        """Test train_music_model() high-level function."""
        model_path = os.path.join(output_dir, "test_api_model.kan")

        # Train using high-level API
        model = train_music_model(
            midi_files=classical_files_small,
            model_type="mlp",
            output_path=model_path,
            encoder_type="note",
            seq_length=16,
            hidden_size=64,
            max_epochs=3,
            early_stopping_patience=0,
            validation_split=0.0,
            augment=False,
            verbose=0,
        )

        assert model is not None, "Model not returned"
        assert os.path.exists(model_path), "Model file not saved"

    def test_generate_music_api(self, classical_files_small, output_dir):
        """Test generate_music() high-level function."""
        model_path = os.path.join(output_dir, "test_gen_model.kan")
        output_path = os.path.join(output_dir, "test_api_generated.mid")

        # Train model first
        train_music_model(
            midi_files=classical_files_small,
            model_type="mlp",
            output_path=model_path,
            seq_length=16,
            hidden_size=64,
            max_epochs=3,
            early_stopping_patience=0,
            validation_split=0.0,
            augment=False,
            verbose=0,
        )

        # Generate using high-level API
        result_path = generate_music(
            model_path=model_path,
            output_path=output_path,
            duration=16,
            temperature=0.8,
            encoder_type="note",
            seq_length=16,
            tempo=120.0,
        )

        assert result_path == output_path
        assert os.path.exists(output_path), "Generated MIDI file not created"


# ============================================================================
# Model Save/Load Tests
# ============================================================================


class TestModelPersistence:
    """Test model saving and loading."""

    def test_save_and_load_model(self, classical_files_small, output_dir):
        """Test saving and loading trained model."""
        # Setup and train
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=16)

        for midi_file in classical_files_small:
            dataset.load_file(midi_file)

        model = ModelFactory.create(
            model_type="mlp",
            encoder=encoder,
            seq_length=16,
            hidden_size=64,
        )

        x_train, y_train = dataset.prepare_training_data(use_numpy=True)

        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=32,
            max_epochs=3,
            verbose=0,
        )
        trainer = Trainer(model, config)
        trainer.train(x_train, y_train)

        # Save model
        model_path = os.path.join(output_dir, "test_persistence.kan")
        model.save(model_path)
        assert os.path.exists(model_path), "Model file not saved"

        # Load model
        loaded_model = NeuralNetwork.load(model_path)
        assert loaded_model is not None, "Model not loaded"
        assert loaded_model.n_var == model.n_var, "Model parameters mismatch"

        # Generate with loaded model
        generator = MusicGenerator(
            loaded_model, encoder, seq_length=16,
            sampling=TemperatureSampling(0.8)
        )

        output_path = os.path.join(output_dir, "test_loaded_model.mid")
        sequence = generator.generate_midi(duration_beats=16)
        sequence.save(output_path)

        assert os.path.exists(output_path), "Output from loaded model not created"


# ============================================================================
# Data Augmentation Tests
# ============================================================================


class TestDataAugmentation:
    """Test data augmentation with classical MIDI files."""

    def test_augmentation_increases_data(self, classical_files_small):
        """Test that augmentation increases dataset size."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=16)

        for midi_file in classical_files_small:
            dataset.load_file(midi_file)

        original_sequences = dataset.n_sequences

        # Apply augmentation
        dataset.augment(transpose_range=(-3, 3))

        augmented_sequences = dataset.n_sequences

        # Augmentation should increase sequences
        # (original + transposed versions)
        assert augmented_sequences >= original_sequences, \
            "Augmentation should not decrease sequences"

    def test_train_with_augmentation(self, classical_files_small, output_dir):
        """Test training with data augmentation enabled."""
        model_path = os.path.join(output_dir, "test_augmented_model.kan")

        model = train_music_model(
            midi_files=classical_files_small,
            model_type="mlp",
            output_path=model_path,
            seq_length=16,
            hidden_size=64,
            max_epochs=3,
            early_stopping_patience=0,
            validation_split=0.0,
            augment=True,  # Enable augmentation
            transpose_range=(-2, 2),
            verbose=0,
        )

        assert model is not None
        assert os.path.exists(model_path)
