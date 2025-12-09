#!/usr/bin/env python3
"""Tests for coremusic.music.neural module.

Tests Phase 1 (data pipeline), Phase 2 (models), Phase 3 (training),
Phase 4 (generation), Phase 5 (evaluation), and Phase 6 (high-level API/CLI)
of the neural network integration for music generation.
"""

import os
import tempfile
import pytest
import array

from conftest import CANON_MID_PATH

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Import the neural module
from coremusic.music.neural import (
    # Phase 1: Data
    BaseEncoder,
    NoteEncoder,
    EventEncoder,
    PianoRollEncoder,
    RelativePitchEncoder,
    MIDIDataset,
    # Phase 2: Models
    create_mlp_model,
    create_rnn_model,
    create_lstm_model,
    create_gru_model,
    ModelFactory,
    # Phase 3: Training
    TrainingConfig,
    Trainer,
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ProgressLogger,
    LearningRateScheduler,
    # Phase 4: Generation
    SamplingStrategy,
    GreedySampling,
    TemperatureSampling,
    TopKSampling,
    NucleusSampling,
    MusicGenerator,
    # Phase 5: Evaluation
    MusicMetrics,
    ModelResult,
    ModelComparison,
    # Phase 6: High-level API
    train_music_model,
    generate_music,
    continue_music,
    quick_train_and_generate,
)

from coremusic.midi.utilities import MIDISequence, MIDIEvent, MIDIStatus
from coremusic.kann import Array2D, NeuralNetwork


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_midi_events():
    """Create sample MIDI events for testing."""
    events = []
    notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale
    for i, note in enumerate(notes):
        time = i * 0.5
        # Note On
        events.append(MIDIEvent(time, MIDIStatus.NOTE_ON, 0, note, 100))
        # Note Off
        events.append(MIDIEvent(time + 0.4, MIDIStatus.NOTE_OFF, 0, note, 0))
    return events


@pytest.fixture
def sample_midi_file(tmp_path):
    """Create a temporary MIDI file for testing."""
    midi_path = tmp_path / "test.mid"

    seq = MIDISequence(tempo=120.0)
    track = seq.add_track("Test")

    # Add a simple melody
    notes = [60, 62, 64, 65, 67, 69, 71, 72, 72, 71, 69, 67, 65, 64, 62, 60]
    for i, note in enumerate(notes):
        track.add_note(i * 0.25, note, 100, 0.2)

    seq.save(str(midi_path))
    return str(midi_path)


@pytest.fixture
def canon_midi_file():
    """Path to the Canon MIDI file for testing."""
    if os.path.exists(CANON_MID_PATH):
        return CANON_MID_PATH
    return None


# ============================================================================
# Phase 1 Tests: NoteEncoder
# ============================================================================


class TestNoteEncoder:
    """Tests for NoteEncoder class."""

    def test_vocab_size(self):
        """Test that vocab_size is 128."""
        encoder = NoteEncoder()
        assert encoder.vocab_size == 128

    def test_encode_basic(self, sample_midi_events):
        """Test encoding MIDI events to note tokens."""
        encoder = NoteEncoder()
        tokens = encoder.encode(sample_midi_events)

        # Should have 8 notes (C major scale)
        assert len(tokens) == 8
        assert tokens == [60, 62, 64, 65, 67, 69, 71, 72]

    def test_encode_empty(self):
        """Test encoding empty event list."""
        encoder = NoteEncoder()
        tokens = encoder.encode([])
        assert tokens == []

    def test_encode_filters_note_off(self):
        """Test that note-off events are filtered out."""
        encoder = NoteEncoder()
        events = [
            MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100),
            MIDIEvent(0.5, MIDIStatus.NOTE_OFF, 0, 60, 0),
            MIDIEvent(1.0, MIDIStatus.NOTE_ON, 0, 64, 100),
        ]
        tokens = encoder.encode(events)
        assert tokens == [60, 64]

    def test_decode_basic(self):
        """Test decoding tokens back to MIDI events."""
        encoder = NoteEncoder()
        tokens = [60, 62, 64]
        events = encoder.decode(tokens)

        # Should have note-on and note-off for each
        assert len(events) == 6

        # Check first note
        note_ons = [e for e in events if e.is_note_on]
        assert len(note_ons) == 3
        assert note_ons[0].data1 == 60
        assert note_ons[1].data1 == 62
        assert note_ons[2].data1 == 64

    def test_decode_empty(self):
        """Test decoding empty token list."""
        encoder = NoteEncoder()
        events = encoder.decode([])
        assert events == []

    def test_roundtrip(self, sample_midi_events):
        """Test encode-decode roundtrip preserves notes."""
        encoder = NoteEncoder()
        tokens = encoder.encode(sample_midi_events)
        events = encoder.decode(tokens)
        tokens2 = encoder.encode(events)

        assert tokens == tokens2

    def test_transpose(self):
        """Test transposition function."""
        encoder = NoteEncoder()
        tokens = [60, 62, 64]

        # Transpose up
        transposed = encoder.transpose(tokens, 12)
        assert transposed == [72, 74, 76]

        # Transpose down
        transposed = encoder.transpose(tokens, -12)
        assert transposed == [48, 50, 52]

    def test_transpose_clipping(self):
        """Test that transposition clips to valid MIDI range."""
        encoder = NoteEncoder()
        tokens = [5, 60, 120]

        # Transpose down past 0
        transposed = encoder.transpose(tokens, -10)
        assert transposed == [0, 50, 110]

        # Transpose up past 127
        transposed = encoder.transpose(tokens, 10)
        assert transposed == [15, 70, 127]

    def test_encode_file(self, sample_midi_file):
        """Test encoding from a MIDI file."""
        encoder = NoteEncoder()
        tokens = encoder.encode_file(sample_midi_file)

        assert len(tokens) > 0
        assert all(0 <= t <= 127 for t in tokens)


# ============================================================================
# Phase 1 Tests: MIDIDataset
# ============================================================================


class TestMIDIDataset:
    """Tests for MIDIDataset class."""

    def test_init(self):
        """Test dataset initialization."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=32)

        assert dataset.encoder is encoder
        assert dataset.seq_length == 32
        assert dataset.n_sequences == 0
        assert dataset.total_tokens == 0

    def test_add_sequence(self):
        """Test adding a sequence directly."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)

        dataset.add_sequence([60, 62, 64, 65, 67, 69, 71, 72] * 5)

        assert dataset.n_sequences == 1
        assert dataset.total_tokens == 40

    def test_load_file(self, sample_midi_file):
        """Test loading a MIDI file."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)

        n_tokens = dataset.load_file(sample_midi_file)

        assert n_tokens > 0
        assert dataset.n_sequences == 1
        assert dataset.total_tokens == n_tokens

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises error."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)

        with pytest.raises(FileNotFoundError):
            dataset.load_file("/nonexistent/file.mid")

    def test_load_directory(self, tmp_path):
        """Test loading from a directory."""
        # Create multiple MIDI files
        for i in range(3):
            seq = MIDISequence(tempo=120.0)
            track = seq.add_track(f"Track {i}")
            for j in range(20):
                track.add_note(j * 0.25, 60 + (j % 12), 100, 0.2)
            seq.save(str(tmp_path / f"test_{i}.mid"))

        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)

        total_tokens = dataset.load_directory(tmp_path)

        assert total_tokens > 0
        assert dataset.n_sequences == 3

    def test_clear(self):
        """Test clearing the dataset."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)

        dataset.add_sequence([60, 62, 64] * 10)
        assert dataset.n_sequences == 1

        dataset.clear()
        assert dataset.n_sequences == 0
        assert dataset.total_tokens == 0

    def test_augment(self):
        """Test data augmentation via transposition."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)

        dataset.add_sequence([60, 62, 64] * 10)
        initial_count = dataset.n_sequences

        # Augment with transposition range -2 to +2 (5 transpositions total)
        new_seqs = dataset.augment(transpose_range=(-2, 2), include_original=True)

        # Original + 4 transpositions (excluding 0)
        assert dataset.n_sequences == initial_count + 4

    def test_augment_without_original(self):
        """Test augmentation without keeping original."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)

        dataset.add_sequence([60, 62, 64] * 10)

        dataset.augment(transpose_range=(-1, 1), include_original=False)

        # Only transposed sequences (-1, 0, +1), original removed
        assert dataset.n_sequences == 3

    def test_prepare_training_data(self):
        """Test preparing training data."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=4)

        # Add sequence longer than seq_length
        dataset.add_sequence([60, 62, 64, 65, 67, 69, 71, 72])

        x_train, y_train = dataset.prepare_training_data(use_numpy=True)

        # Should have (8 - 4) = 4 samples
        assert x_train.shape[0] == 4
        assert y_train.shape[0] == 4

        # X should be one-hot encoded: seq_length * vocab_size
        assert x_train.shape[1] == 4 * 128

        # Y should be one-hot: vocab_size
        assert y_train.shape[1] == 128

    def test_prepare_training_data_empty(self):
        """Test that empty dataset raises error."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)

        with pytest.raises(ValueError):
            dataset.prepare_training_data()

    def test_prepare_training_data_too_short(self):
        """Test that sequences shorter than seq_length are skipped."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=10)

        # Add short sequence
        dataset.add_sequence([60, 62, 64])

        with pytest.raises(ValueError):
            dataset.prepare_training_data()

    def test_get_sample_sequence(self):
        """Test getting a sample sequence for generation."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)

        original = [60, 62, 64, 65, 67, 69, 71, 72] * 3
        dataset.add_sequence(original)

        sample = dataset.get_sample_sequence(8)

        assert len(sample) == 8
        # Sample should be a subsequence of original
        assert all(s in original for s in sample)

    def test_repr(self):
        """Test string representation."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=16)

        rep = repr(dataset)
        assert "NoteEncoder" in rep
        assert "seq_length=16" in rep


# ============================================================================
# Phase 2 Tests: Model Factory Functions
# ============================================================================


class TestCreateModels:
    """Tests for model factory functions."""

    def test_create_mlp_model(self):
        """Test creating an MLP model."""
        model = create_mlp_model(
            input_size=32 * 128,
            hidden_sizes=[256, 128],
            output_size=128,
            dropout=0.2,
        )

        assert isinstance(model, NeuralNetwork)
        assert model.input_dim == 32 * 128
        assert model.n_var > 0

    def test_create_mlp_model_defaults(self):
        """Test MLP model with default hidden sizes."""
        model = create_mlp_model(
            input_size=1024,
            output_size=128,
        )

        assert isinstance(model, NeuralNetwork)

    def test_create_rnn_model(self):
        """Test creating a simple RNN model."""
        model = create_rnn_model(
            input_size=128,
            hidden_size=256,
            output_size=128,
        )

        assert isinstance(model, NeuralNetwork)
        assert model.input_dim == 128
        assert model.n_var > 0

    def test_create_lstm_model(self):
        """Test creating an LSTM model."""
        model = create_lstm_model(
            input_size=128,
            hidden_size=256,
            output_size=128,
        )

        assert isinstance(model, NeuralNetwork)
        assert model.input_dim == 128
        assert model.n_var > 0

    def test_create_gru_model(self):
        """Test creating a GRU model."""
        model = create_gru_model(
            input_size=128,
            hidden_size=256,
            output_size=128,
        )

        assert isinstance(model, NeuralNetwork)
        assert model.input_dim == 128
        assert model.n_var > 0


# ============================================================================
# Phase 2 Tests: ModelFactory Class
# ============================================================================


class TestModelFactory:
    """Tests for ModelFactory class."""

    def test_create_mlp(self):
        """Test creating MLP via factory."""
        encoder = NoteEncoder()
        model = ModelFactory.create(
            model_type="mlp",
            encoder=encoder,
            seq_length=16,
            hidden_size=128,
        )

        assert isinstance(model, NeuralNetwork)
        # MLP input size = seq_length * vocab_size
        assert model.input_dim == 16 * 128

    def test_create_lstm(self):
        """Test creating LSTM via factory."""
        encoder = NoteEncoder()
        model = ModelFactory.create(
            model_type="lstm",
            encoder=encoder,
            seq_length=16,
            hidden_size=128,
        )

        assert isinstance(model, NeuralNetwork)
        # RNN input size = vocab_size (per timestep)
        assert model.input_dim == 128

    def test_create_gru(self):
        """Test creating GRU via factory."""
        encoder = NoteEncoder()
        model = ModelFactory.create(
            model_type="gru",
            encoder=encoder,
            seq_length=16,
            hidden_size=128,
        )

        assert isinstance(model, NeuralNetwork)
        assert model.input_dim == 128

    def test_create_rnn(self):
        """Test creating RNN via factory."""
        encoder = NoteEncoder()
        model = ModelFactory.create(
            model_type="rnn",
            encoder=encoder,
            seq_length=16,
            hidden_size=128,
        )

        assert isinstance(model, NeuralNetwork)
        assert model.input_dim == 128

    def test_create_case_insensitive(self):
        """Test that model type is case-insensitive."""
        encoder = NoteEncoder()

        model1 = ModelFactory.create("LSTM", encoder, seq_length=8, hidden_size=64)
        model2 = ModelFactory.create("lstm", encoder, seq_length=8, hidden_size=64)
        model3 = ModelFactory.create("LsTm", encoder, seq_length=8, hidden_size=64)

        assert isinstance(model1, NeuralNetwork)
        assert isinstance(model2, NeuralNetwork)
        assert isinstance(model3, NeuralNetwork)

    def test_create_invalid_type(self):
        """Test that invalid model type raises error."""
        encoder = NoteEncoder()

        with pytest.raises(ValueError, match="Unknown model type"):
            ModelFactory.create("transformer", encoder, seq_length=16)

    def test_list_models(self):
        """Test listing available models."""
        models = ModelFactory.list_models()

        assert "mlp" in models
        assert "rnn" in models
        assert "lstm" in models
        assert "gru" in models

    def test_get_recommended_config(self):
        """Test getting recommended configuration."""
        config = ModelFactory.get_recommended_config("lstm", dataset_size=5000)

        assert "learning_rate" in config
        assert "mini_batch_size" in config
        assert "max_epochs" in config
        assert "hidden_size" in config

    def test_describe_model(self):
        """Test model description."""
        desc = ModelFactory.describe_model("lstm")

        assert "LSTM" in desc
        assert "long-range" in desc.lower() or "long" in desc.lower()

    def test_describe_invalid_model(self):
        """Test that describing invalid model raises error."""
        with pytest.raises(ValueError):
            ModelFactory.describe_model("invalid_model")


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the full pipeline."""

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required for training")
    def test_encode_train_mlp(self, sample_midi_file):
        """Test encoding data and training an MLP model."""
        # Load and encode
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)
        dataset.load_file(sample_midi_file)

        # Prepare data (numpy arrays for training)
        x_train, y_train = dataset.prepare_training_data(use_numpy=True)

        # Create model
        model = ModelFactory.create(
            model_type="mlp",
            encoder=encoder,
            seq_length=8,
            hidden_size=64,
        )

        # Train for just 1 epoch to verify it works
        epochs = model.train(
            x_train,
            y_train,
            learning_rate=0.01,
            mini_batch_size=8,
            max_epochs=1,
            validation_fraction=0.0,
        )

        assert epochs >= 0

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required for training")
    def test_model_save_load(self, sample_midi_file, tmp_path):
        """Test saving and loading a trained model."""
        # Create and minimally train a model
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)
        dataset.load_file(sample_midi_file)

        x_train, y_train = dataset.prepare_training_data(use_numpy=True)

        model = create_mlp_model(
            input_size=8 * 128,
            hidden_sizes=[64],
            output_size=128,
        )

        model.train(
            x_train,
            y_train,
            learning_rate=0.01,
            max_epochs=1,
            validation_fraction=0.0,
        )

        # Save
        model_path = str(tmp_path / "model.kan")
        model.save(model_path)

        assert os.path.exists(model_path)

        # Load
        loaded = NeuralNetwork.load(model_path)
        assert loaded.input_dim == model.input_dim
        assert loaded.n_var == model.n_var

    def test_inference(self, sample_midi_file):
        """Test running inference with a model."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)
        dataset.load_file(sample_midi_file)

        # Get a sample
        sample = dataset.get_sample_sequence(8)

        # Create one-hot input
        vocab_size = encoder.vocab_size
        input_data = array.array('f', [0.0] * (8 * vocab_size))
        for i, note in enumerate(sample):
            if 0 <= note < vocab_size:
                input_data[i * vocab_size + note] = 1.0

        # Create model
        model = create_mlp_model(
            input_size=8 * 128,
            hidden_sizes=[64],
            output_size=128,
        )

        # Run inference
        output = model.apply(input_data)

        assert len(output) == 128
        # Output should be probabilities (sum ~= 1 after softmax)
        assert sum(output) > 0


# ============================================================================
# Canon MIDI File Test (if available)
# ============================================================================


def is_valid_midi_file(path):
    """Check if a file is a valid MIDI file."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'rb') as f:
            header = f.read(4)
            return header == b'MThd'
    except Exception:
        return False


@pytest.mark.skipif(
    not is_valid_midi_file(CANON_MID_PATH),
    reason="canon.mid not available or invalid"
)
class TestCanonMidi:
    """Tests using the Canon MIDI file."""

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required for training")
    def test_encode_canon(self, canon_midi_file):
        """Test encoding canon.mid."""
        encoder = NoteEncoder()
        tokens = encoder.encode_file(canon_midi_file)

        assert len(tokens) > 0
        print(f"Canon MIDI: {len(tokens)} notes encoded")

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required for training")
    def test_train_on_canon(self, canon_midi_file):
        """Test training a small model on canon.mid."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=16)
        dataset.load_file(canon_midi_file)

        # Small augmentation
        dataset.augment(transpose_range=(-2, 2))

        print(f"Dataset: {dataset.n_sequences} sequences, {dataset.total_tokens} tokens")

        x_train, y_train = dataset.prepare_training_data(use_numpy=True)
        print(f"Training samples: {x_train.shape[0]}")

        # Train a small model
        model = create_mlp_model(
            input_size=16 * 128,
            hidden_sizes=[128],
            output_size=128,
            dropout=0.2,
        )

        epochs = model.train(
            x_train,
            y_train,
            learning_rate=0.001,
            mini_batch_size=32,
            max_epochs=5,
            validation_fraction=0.1,
        )

        print(f"Trained for {epochs} epochs")
        assert epochs > 0


# ============================================================================
# Phase 3 Tests: Training Infrastructure
# ============================================================================


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.learning_rate == 0.001
        assert config.batch_size == 64
        assert config.max_epochs == 100
        assert config.min_epochs == 10
        assert config.early_stopping_patience == 10
        assert config.validation_split == 0.1
        assert config.verbose == 1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=32,
            max_epochs=50,
        )

        assert config.learning_rate == 0.01
        assert config.batch_size == 32
        assert config.max_epochs == 50


class TestCallbacks:
    """Tests for training callbacks."""

    def test_early_stopping_init(self):
        """Test EarlyStopping initialization."""
        callback = EarlyStopping(patience=5, min_delta=0.001)

        assert callback.patience == 5
        assert callback.min_delta == 0.001

    def test_early_stopping_improvement(self):
        """Test EarlyStopping detects improvement."""
        callback = EarlyStopping(patience=3)
        callback.on_train_begin()

        # Improving losses should not trigger early stop
        assert callback.on_epoch_end(0, {"val_loss": 1.0}) is True
        assert callback.on_epoch_end(1, {"val_loss": 0.8}) is True
        assert callback.on_epoch_end(2, {"val_loss": 0.6}) is True
        assert callback.wait == 0

    def test_early_stopping_no_improvement(self):
        """Test EarlyStopping triggers after no improvement."""
        callback = EarlyStopping(patience=2)
        callback.on_train_begin()

        assert callback.on_epoch_end(0, {"val_loss": 0.5}) is True  # Best so far
        assert callback.on_epoch_end(1, {"val_loss": 0.6}) is True  # No improvement, wait=1
        assert callback.on_epoch_end(2, {"val_loss": 0.7}) is False  # No improvement, wait=2, triggers stop

    def test_model_checkpoint_init(self):
        """Test ModelCheckpoint initialization."""
        callback = ModelCheckpoint("model.kan", save_best_only=True)

        assert callback.filepath == "model.kan"
        assert callback.save_best_only is True

    def test_progress_logger_init(self):
        """Test ProgressLogger initialization."""
        callback = ProgressLogger(log_frequency=5)

        assert callback.log_frequency == 5

    def test_learning_rate_scheduler(self):
        """Test LearningRateScheduler."""
        def schedule(epoch, lr):
            if epoch > 0 and epoch % 5 == 0:
                return lr * 0.9
            return lr

        callback = LearningRateScheduler(schedule, verbose=False)
        callback.current_lr = 0.001
        callback.on_epoch_begin(0)
        assert callback.current_lr == 0.001

        callback.on_epoch_begin(5)
        assert callback.current_lr == pytest.approx(0.0009, rel=0.01)


class TestTrainer:
    """Tests for Trainer class."""

    def test_trainer_init(self):
        """Test Trainer initialization."""
        model = create_mlp_model(128, [64], 128)
        config = TrainingConfig(max_epochs=10)
        trainer = Trainer(model, config)

        assert trainer.model is model
        assert trainer.config is config
        assert trainer.callbacks == []

    def test_trainer_add_callback(self):
        """Test adding callbacks to trainer."""
        model = create_mlp_model(128, [64], 128)
        trainer = Trainer(model)

        trainer.add_callback(EarlyStopping(patience=5))
        trainer.add_callback(ProgressLogger())

        assert len(trainer.callbacks) == 2

    def test_trainer_add_callback_chaining(self):
        """Test callback chaining."""
        model = create_mlp_model(128, [64], 128)
        trainer = Trainer(model)

        result = trainer.add_callback(EarlyStopping()).add_callback(ProgressLogger())

        assert result is trainer
        assert len(trainer.callbacks) == 2

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required for training")
    def test_trainer_train(self, sample_midi_file):
        """Test training with Trainer class."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)
        dataset.load_file(sample_midi_file)

        x_train, y_train = dataset.prepare_training_data(use_numpy=True)

        model = create_mlp_model(8 * 128, [64], 128)
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=8,
            max_epochs=2,
            validation_split=0.0,
            verbose=0,
        )
        trainer = Trainer(model, config)

        history = trainer.train(x_train, y_train)

        assert "loss" in history
        assert len(history["loss"]) > 0

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required for training")
    def test_trainer_evaluate(self, sample_midi_file):
        """Test evaluation with Trainer class."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)
        dataset.load_file(sample_midi_file)

        x_train, y_train = dataset.prepare_training_data(use_numpy=True)

        model = create_mlp_model(8 * 128, [64], 128)
        trainer = Trainer(model, TrainingConfig(verbose=0))

        metrics = trainer.evaluate(x_train, y_train)

        assert "loss" in metrics
        assert metrics["loss"] > 0


# ============================================================================
# Phase 4 Tests: Sampling Strategies
# ============================================================================


class TestSamplingStrategies:
    """Tests for sampling strategies."""

    def test_greedy_sampling(self):
        """Test greedy sampling always picks max."""
        sampler = GreedySampling()
        probs = [0.1, 0.7, 0.2]

        # Should always return index 1 (highest prob)
        for _ in range(10):
            assert sampler.sample(probs) == 1

    def test_greedy_sampling_array(self):
        """Test greedy sampling with array.array."""
        sampler = GreedySampling()
        probs = array.array('f', [0.1, 0.2, 0.5, 0.2])

        assert sampler.sample(probs) == 2

    def test_temperature_sampling_init(self):
        """Test TemperatureSampling initialization."""
        sampler = TemperatureSampling(temperature=0.5)
        assert sampler.temperature == 0.5

    def test_temperature_sampling_invalid(self):
        """Test TemperatureSampling rejects invalid temperature."""
        with pytest.raises(ValueError):
            TemperatureSampling(temperature=0)

        with pytest.raises(ValueError):
            TemperatureSampling(temperature=-1)

    def test_temperature_sampling_distribution(self):
        """Test temperature sampling produces valid indices."""
        sampler = TemperatureSampling(temperature=1.0)
        probs = [0.25, 0.25, 0.25, 0.25]

        # Sample many times and verify all indices are valid
        samples = [sampler.sample(probs) for _ in range(100)]
        assert all(0 <= s < 4 for s in samples)

    def test_temperature_low_more_deterministic(self):
        """Test that low temperature is more deterministic."""
        probs = [0.1, 0.6, 0.3]

        # Low temperature should mostly pick index 1
        sampler_low = TemperatureSampling(temperature=0.1)
        samples_low = [sampler_low.sample(probs) for _ in range(100)]
        count_1_low = sum(1 for s in samples_low if s == 1)

        # Higher temperature should be more varied
        sampler_high = TemperatureSampling(temperature=2.0)
        samples_high = [sampler_high.sample(probs) for _ in range(100)]
        count_1_high = sum(1 for s in samples_high if s == 1)

        # Low temp should pick index 1 more often
        assert count_1_low >= count_1_high

    def test_topk_sampling_init(self):
        """Test TopKSampling initialization."""
        sampler = TopKSampling(k=5, temperature=0.8)

        assert sampler.k == 5
        assert sampler.temperature == 0.8

    def test_topk_sampling_invalid(self):
        """Test TopKSampling rejects invalid k."""
        with pytest.raises(ValueError):
            TopKSampling(k=0)

    def test_topk_sampling_limits_choices(self):
        """Test TopKSampling only picks from top k."""
        sampler = TopKSampling(k=2, temperature=1.0)
        # Index 0 and 2 have highest probs
        probs = [0.4, 0.1, 0.4, 0.1]

        samples = [sampler.sample(probs) for _ in range(100)]
        # Should only sample indices 0 and 2
        assert all(s in [0, 2] for s in samples)

    def test_nucleus_sampling_init(self):
        """Test NucleusSampling initialization."""
        sampler = NucleusSampling(p=0.9, temperature=0.8)

        assert sampler.p == 0.9
        assert sampler.temperature == 0.8

    def test_nucleus_sampling_invalid(self):
        """Test NucleusSampling rejects invalid p."""
        with pytest.raises(ValueError):
            NucleusSampling(p=0)

        with pytest.raises(ValueError):
            NucleusSampling(p=1.5)

    def test_nucleus_sampling_limits_choices(self):
        """Test NucleusSampling respects cumulative probability threshold."""
        sampler = NucleusSampling(p=0.5, temperature=1.0)
        # Only first token (0.6) exceeds 0.5 threshold
        probs = [0.6, 0.2, 0.1, 0.1]

        samples = [sampler.sample(probs) for _ in range(100)]
        # Should predominantly sample index 0
        count_0 = sum(1 for s in samples if s == 0)
        assert count_0 > 50  # Majority should be index 0


# ============================================================================
# Phase 4 Tests: Music Generator
# ============================================================================


class TestMusicGenerator:
    """Tests for MusicGenerator class."""

    def test_generator_init(self):
        """Test MusicGenerator initialization."""
        model = create_mlp_model(8 * 128, [64], 128)
        encoder = NoteEncoder()

        generator = MusicGenerator(model, encoder, seq_length=8)

        assert generator.model is model
        assert generator.encoder is encoder
        assert generator.seq_length == 8
        assert isinstance(generator.sampling, TemperatureSampling)

    def test_generator_set_sampling(self):
        """Test setting sampling strategy."""
        model = create_mlp_model(8 * 128, [64], 128)
        encoder = NoteEncoder()
        generator = MusicGenerator(model, encoder, seq_length=8)

        result = generator.set_sampling(GreedySampling())

        assert result is generator
        assert isinstance(generator.sampling, GreedySampling)

    def test_generator_generate(self):
        """Test token generation."""
        model = create_mlp_model(8 * 128, [64], 128)
        encoder = NoteEncoder()
        generator = MusicGenerator(model, encoder, seq_length=8)

        tokens = generator.generate(length=20)

        # Should have seed (8) + generated (20) = 28 tokens
        assert len(tokens) == 28
        assert all(0 <= t < 128 for t in tokens)

    def test_generator_generate_with_seed(self):
        """Test generation with seed sequence."""
        model = create_mlp_model(8 * 128, [64], 128)
        encoder = NoteEncoder()
        generator = MusicGenerator(model, encoder, seq_length=8)

        seed = [60, 62, 64, 65, 67, 69, 71, 72]
        tokens = generator.generate(seed=seed, length=10)

        # First 8 tokens should be the seed
        assert tokens[:8] == seed
        assert len(tokens) == 18

    def test_generator_generate_with_short_seed(self):
        """Test generation with seed shorter than seq_length."""
        model = create_mlp_model(8 * 128, [64], 128)
        encoder = NoteEncoder()
        generator = MusicGenerator(model, encoder, seq_length=8)

        seed = [60, 62, 64]  # Only 3 tokens
        tokens = generator.generate(seed=seed, length=10)

        # Should pad and generate
        assert len(tokens) == 18

    def test_generator_generate_with_temperature(self):
        """Test generation with temperature override."""
        model = create_mlp_model(8 * 128, [64], 128)
        encoder = NoteEncoder()
        generator = MusicGenerator(model, encoder, seq_length=8)

        tokens = generator.generate(length=10, temperature=0.5)

        assert len(tokens) == 18

    def test_generator_generate_midi(self):
        """Test MIDI sequence generation."""
        model = create_mlp_model(8 * 128, [64], 128)
        encoder = NoteEncoder()
        generator = MusicGenerator(model, encoder, seq_length=8)

        sequence = generator.generate_midi(duration_beats=8, tempo=120)

        assert isinstance(sequence, MIDISequence)
        assert sequence.tempo == 120
        assert len(sequence.tracks) == 1
        assert sequence.tracks[0].name == "Generated"

    def test_generator_generate_variations(self):
        """Test generating multiple variations."""
        model = create_mlp_model(8 * 128, [64], 128)
        encoder = NoteEncoder()
        generator = MusicGenerator(model, encoder, seq_length=8)

        seed = [60, 62, 64, 65, 67, 69, 71, 72]
        variations = generator.generate_variations(
            seed=seed,
            num_variations=3,
            length=10,
        )

        assert len(variations) == 3
        # Each variation should be different (with high probability)
        # and start with the same seed
        for var in variations:
            assert len(var) == 18

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required for training")
    def test_generator_with_trained_model(self, sample_midi_file, tmp_path):
        """Test generation with a minimally trained model."""
        # Train a small model
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)
        dataset.load_file(sample_midi_file)

        x_train, y_train = dataset.prepare_training_data(use_numpy=True)

        model = create_mlp_model(8 * 128, [32], 128)
        model.train(x_train, y_train, learning_rate=0.01, max_epochs=1, validation_fraction=0.0)

        # Generate
        generator = MusicGenerator(model, encoder, seq_length=8)
        generator.set_sampling(TemperatureSampling(0.8))

        tokens = generator.generate(length=20)
        assert len(tokens) == 28
        assert all(0 <= t < 128 for t in tokens)

        # Generate MIDI
        sequence = generator.generate_midi(duration_beats=8)
        assert len(sequence.tracks[0].events) > 0

        # Save and verify
        output_path = tmp_path / "generated.mid"
        sequence.save(str(output_path))
        assert output_path.exists()


# ============================================================================
# Full Pipeline Integration Test
# ============================================================================


class TestFullPipeline:
    """End-to-end integration tests for the complete pipeline."""

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required for training")
    def test_full_training_and_generation_pipeline(self, sample_midi_file, tmp_path):
        """Test complete pipeline: load -> encode -> train -> generate -> save."""
        # Phase 1: Data loading and encoding
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)
        dataset.load_file(sample_midi_file)
        dataset.augment(transpose_range=(-2, 2))

        x_train, y_train = dataset.prepare_training_data(use_numpy=True)

        # Phase 2: Model creation
        model = ModelFactory.create(
            model_type="mlp",
            encoder=encoder,
            seq_length=8,
            hidden_size=32,
        )

        # Phase 3: Training with callbacks
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=8,
            max_epochs=2,
            validation_split=0.0,
            verbose=0,
        )
        trainer = Trainer(model, config)
        trainer.add_callback(EarlyStopping(patience=5))

        history = trainer.train(x_train, y_train)
        assert "loss" in history

        # Save model
        model_path = str(tmp_path / "model.kan")
        model.save(model_path)
        assert os.path.exists(model_path)

        # Phase 4: Generation
        generator = MusicGenerator(model, encoder, seq_length=8)
        generator.set_sampling(TemperatureSampling(0.8))

        # Generate tokens
        seed = dataset.get_sample_sequence(8)
        tokens = generator.generate(seed=seed, length=32)
        assert len(tokens) == 40

        # Generate MIDI
        sequence = generator.generate_midi(duration_beats=16, tempo=120)
        assert isinstance(sequence, MIDISequence)

        # Save generated MIDI
        output_path = str(tmp_path / "generated.mid")
        sequence.save(output_path)
        assert os.path.exists(output_path)

        # Load and verify
        loaded = MIDISequence.load(output_path)
        assert len(loaded.tracks) > 0


# ============================================================================
# Phase 5 Tests: EventEncoder
# ============================================================================


class TestEventEncoder:
    """Tests for EventEncoder class."""

    def test_vocab_size(self):
        """Test that vocab_size includes all token types."""
        encoder = EventEncoder()
        # 128 NOTE_ON + 128 NOTE_OFF + 100 TIME_SHIFT + 32 VELOCITY = 388
        assert encoder.vocab_size == 388

    def test_encode_basic(self, sample_midi_events):
        """Test encoding MIDI events to event tokens."""
        encoder = EventEncoder()
        tokens = encoder.encode(sample_midi_events)

        # Should have tokens for velocities, time shifts, note on/off
        assert len(tokens) > 0

        # Check that NOTE_ON tokens are in valid range
        note_on_tokens = [t for t in tokens if 0 <= t < 128]
        assert len(note_on_tokens) > 0

    def test_encode_empty(self):
        """Test encoding empty event list."""
        encoder = EventEncoder()
        tokens = encoder.encode([])
        assert tokens == []

    def test_decode_basic(self):
        """Test decoding tokens back to MIDI events."""
        encoder = EventEncoder()

        # Create tokens: velocity, note on, time shift, note off
        tokens = [
            encoder.VELOCITY_OFFSET + 15,  # Velocity ~50%
            60,  # NOTE_ON C4
            encoder.TIME_SHIFT_OFFSET + 9,  # 10 time steps
            encoder.NOTE_OFF_OFFSET + 60,  # NOTE_OFF C4
        ]
        events = encoder.decode(tokens)

        # Should have at least note on and note off
        assert len(events) >= 2

        note_ons = [e for e in events if e.is_note_on]
        assert len(note_ons) >= 1
        assert note_ons[0].data1 == 60  # C4

    def test_decode_empty(self):
        """Test decoding empty token list."""
        encoder = EventEncoder()
        events = encoder.decode([])
        assert events == []

    def test_transpose(self):
        """Test transposition function preserves non-note tokens."""
        encoder = EventEncoder()

        # Mix of note tokens and other tokens
        tokens = [
            60,  # NOTE_ON C4
            encoder.TIME_SHIFT_OFFSET + 5,  # Time shift
            encoder.NOTE_OFF_OFFSET + 60,  # NOTE_OFF C4
            encoder.VELOCITY_OFFSET + 10,  # Velocity
        ]

        transposed = encoder.transpose(tokens, 12)

        # Note tokens should be transposed
        assert transposed[0] == 72  # NOTE_ON C5
        # Time shift should be unchanged
        assert transposed[1] == encoder.TIME_SHIFT_OFFSET + 5
        # Note off should be transposed
        assert transposed[2] == encoder.NOTE_OFF_OFFSET + 72
        # Velocity should be unchanged
        assert transposed[3] == encoder.VELOCITY_OFFSET + 10

    def test_transpose_clipping(self):
        """Test transposition clips notes to valid range."""
        encoder = EventEncoder()

        tokens = [5, 120]  # Low and high notes

        transposed_down = encoder.transpose(tokens, -10)
        assert transposed_down[0] == 0  # Clipped to 0
        assert transposed_down[1] == 110

        transposed_up = encoder.transpose(tokens, 10)
        assert transposed_up[0] == 15
        assert transposed_up[1] == 127  # Clipped to 127


# ============================================================================
# Phase 5 Tests: PianoRollEncoder
# ============================================================================


class TestPianoRollEncoder:
    """Tests for PianoRollEncoder class."""

    def test_vocab_size(self):
        """Test that vocab_size is 128."""
        encoder = PianoRollEncoder()
        assert encoder.vocab_size == 128

    def test_default_resolution(self):
        """Test default resolution is 16."""
        encoder = PianoRollEncoder()
        assert encoder.resolution == 16

    def test_custom_resolution(self):
        """Test custom resolution."""
        encoder = PianoRollEncoder(resolution=8)
        assert encoder.resolution == 8

    def test_encode_basic(self, sample_midi_events):
        """Test encoding MIDI events."""
        encoder = PianoRollEncoder(resolution=4)
        tokens = encoder.encode(sample_midi_events)

        # Should have tokens representing active notes
        assert len(tokens) > 0
        assert all(0 <= t <= 127 for t in tokens)

    def test_encode_empty(self):
        """Test encoding empty event list."""
        encoder = PianoRollEncoder()
        tokens = encoder.encode([])
        assert tokens == []

    def test_encode_to_piano_roll(self, sample_midi_events):
        """Test encoding to full piano roll matrix."""
        encoder = PianoRollEncoder(resolution=4)
        piano_roll = encoder.encode_to_piano_roll(sample_midi_events, num_beats=4)

        # Should be 4 beats * 4 steps = 16 time steps
        assert len(piano_roll) == 16
        # Each step should have 128 note slots
        assert all(len(step) == 128 for step in piano_roll)

    def test_decode_piano_roll(self):
        """Test decoding a piano roll back to events."""
        encoder = PianoRollEncoder(resolution=4)

        # Create simple piano roll: C4 (60) active for 4 steps
        piano_roll = [[0] * 128 for _ in range(8)]
        for step in range(4):
            piano_roll[step][60] = 1

        events = encoder.decode_piano_roll(piano_roll, tempo=120)

        # Should have note-on at start and note-off at step 4
        note_ons = [e for e in events if e.is_note_on]
        note_offs = [e for e in events if e.is_note_off]

        assert len(note_ons) == 1
        assert note_ons[0].data1 == 60
        assert len(note_offs) == 1

    def test_decode_empty(self):
        """Test decoding empty piano roll."""
        encoder = PianoRollEncoder()
        events = encoder.decode_piano_roll([])
        assert events == []

    def test_transpose(self):
        """Test transposition of note tokens."""
        encoder = PianoRollEncoder()
        tokens = [60, 62, 64]

        transposed = encoder.transpose(tokens, 12)
        assert transposed == [72, 74, 76]

    def test_transpose_piano_roll(self):
        """Test transposition of piano roll."""
        encoder = PianoRollEncoder()

        # Create piano roll with C4 active
        piano_roll = [[0] * 128 for _ in range(4)]
        piano_roll[0][60] = 1
        piano_roll[1][60] = 1

        transposed = encoder.transpose_piano_roll(piano_roll, 12)

        # C4 should now be C5
        assert transposed[0][60] == 0
        assert transposed[0][72] == 1
        assert transposed[1][72] == 1

    def test_transpose_clipping(self):
        """Test transposition clips to valid range."""
        encoder = PianoRollEncoder()
        tokens = [5, 120]

        transposed = encoder.transpose(tokens, -10)
        assert transposed[0] == 0  # Clipped

        transposed = encoder.transpose(tokens, 10)
        assert transposed[1] == 127  # Clipped

    def test_polyphony(self):
        """Test encoding polyphonic notes."""
        encoder = PianoRollEncoder(resolution=4)

        # Create polyphonic events: C major chord
        events = [
            MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100),  # C4
            MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 64, 100),  # E4
            MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 67, 100),  # G4
            MIDIEvent(0.5, MIDIStatus.NOTE_OFF, 0, 60, 0),
            MIDIEvent(0.5, MIDIStatus.NOTE_OFF, 0, 64, 0),
            MIDIEvent(0.5, MIDIStatus.NOTE_OFF, 0, 67, 0),
        ]

        piano_roll = encoder.encode_to_piano_roll(events, num_beats=1)

        # First time step should have all 3 notes active
        assert piano_roll[0][60] == 1
        assert piano_roll[0][64] == 1
        assert piano_roll[0][67] == 1


# ============================================================================
# Phase 5 Tests: RelativePitchEncoder
# ============================================================================


class TestRelativePitchEncoder:
    """Tests for RelativePitchEncoder class."""

    def test_vocab_size(self):
        """Test vocabulary size."""
        encoder = RelativePitchEncoder()
        # 49 intervals (-24 to +24) + REST + HOLD + START = 52
        assert encoder.vocab_size == 52

    def test_special_tokens(self):
        """Test special token values."""
        encoder = RelativePitchEncoder()
        assert encoder.REST_TOKEN == 49
        assert encoder.HOLD_TOKEN == 50
        assert encoder.START_TOKEN == 51

    def test_encode_basic(self, sample_midi_events):
        """Test encoding MIDI events to intervals."""
        encoder = RelativePitchEncoder()
        tokens = encoder.encode(sample_midi_events)

        # Should start with START token
        assert tokens[0] == encoder.START_TOKEN
        # Should have more tokens for the notes
        assert len(tokens) > 1

    def test_encode_empty(self):
        """Test encoding empty event list."""
        encoder = RelativePitchEncoder()
        tokens = encoder.encode([])
        assert tokens == []

    def test_encode_intervals(self):
        """Test that intervals are correctly encoded."""
        # Disable time quantization to avoid REST tokens
        encoder = RelativePitchEncoder(quantize_time=False)

        # Create simple ascending sequence: C4, D4, E4
        events = [
            MIDIEvent(0.0, MIDIStatus.NOTE_ON, 0, 60, 100),
            MIDIEvent(0.05, MIDIStatus.NOTE_ON, 0, 62, 100),  # +2
            MIDIEvent(0.1, MIDIStatus.NOTE_ON, 0, 64, 100),   # +2
        ]

        tokens = encoder.encode(events)

        # START + first_interval + +2 + +2
        assert tokens[0] == encoder.START_TOKEN
        # First note: interval from 60 (middle C) is 0
        assert tokens[1] == encoder._interval_to_token(0)
        # Second note: +2 semitones
        assert tokens[2] == encoder._interval_to_token(2)
        # Third note: +2 semitones
        assert tokens[3] == encoder._interval_to_token(2)

    def test_decode_basic(self):
        """Test decoding interval tokens."""
        encoder = RelativePitchEncoder()

        # Encode: START, 0 interval (stays at 60), +2, +2
        tokens = [
            encoder.START_TOKEN,
            encoder._interval_to_token(0),   # C4 (60)
            encoder._interval_to_token(2),   # D4 (62)
            encoder._interval_to_token(2),   # E4 (64)
        ]

        events = encoder.decode(tokens, start_note=60)
        note_ons = [e for e in events if e.is_note_on]

        assert len(note_ons) == 3
        assert note_ons[0].data1 == 60
        assert note_ons[1].data1 == 62
        assert note_ons[2].data1 == 64

    def test_decode_with_different_start(self):
        """Test decoding with different start note (transposition)."""
        encoder = RelativePitchEncoder()

        # Same intervals but starting from G4 (67)
        tokens = [
            encoder.START_TOKEN,
            encoder._interval_to_token(0),   # G4 (67)
            encoder._interval_to_token(2),   # A4 (69)
            encoder._interval_to_token(2),   # B4 (71)
        ]

        events = encoder.decode(tokens, start_note=67)
        note_ons = [e for e in events if e.is_note_on]

        assert note_ons[0].data1 == 67
        assert note_ons[1].data1 == 69
        assert note_ons[2].data1 == 71

    def test_transpose_is_noop(self):
        """Test that transpose doesn't change relative tokens."""
        encoder = RelativePitchEncoder()
        tokens = [encoder.START_TOKEN, 24, 26, 28]  # Some intervals

        transposed = encoder.transpose(tokens, 12)

        # Should be identical (transposition is a no-op for relative encoding)
        assert transposed == tokens

    def test_rest_tokens(self):
        """Test REST token handling."""
        encoder = RelativePitchEncoder()

        # Decode sequence with rest
        tokens = [
            encoder.START_TOKEN,
            encoder._interval_to_token(0),   # Note at time 0
            encoder.REST_TOKEN,              # Rest
            encoder._interval_to_token(0),   # Note at time 2
        ]

        events = encoder.decode(tokens, start_note=60)
        note_ons = [e for e in events if e.is_note_on]

        # Should have 2 notes with gap between them
        assert len(note_ons) == 2
        assert note_ons[1].time > note_ons[0].time

    def test_tokens_to_notes(self):
        """Test converting tokens to absolute notes."""
        encoder = RelativePitchEncoder()

        tokens = [
            encoder.START_TOKEN,
            encoder._interval_to_token(0),   # 60
            encoder._interval_to_token(4),   # 64
            encoder.REST_TOKEN,
            encoder._interval_to_token(-3),  # 61
        ]

        notes = encoder.tokens_to_notes(tokens, start_note=60)

        assert notes[0] is None  # START
        assert notes[1] == 60
        assert notes[2] == 64
        assert notes[3] is None  # REST
        assert notes[4] == 61

    def test_interval_clipping(self):
        """Test that large intervals are clipped."""
        encoder = RelativePitchEncoder(max_interval=24)

        # Interval of 30 should be clipped to 24
        token = encoder._interval_to_token(30)
        interval = encoder._token_to_interval(token)
        assert interval == 24

        # Interval of -30 should be clipped to -24
        token = encoder._interval_to_token(-30)
        interval = encoder._token_to_interval(token)
        assert interval == -24

    def test_roundtrip(self, sample_midi_events):
        """Test encode-decode roundtrip."""
        encoder = RelativePitchEncoder()

        tokens = encoder.encode(sample_midi_events)
        events = encoder.decode(tokens, start_note=60)
        notes = [e.data1 for e in events if e.is_note_on]

        # Should have same number of notes
        original_notes = [e.data1 for e in sample_midi_events if e.is_note_on]
        assert len(notes) == len(original_notes)


# ============================================================================
# Phase 4b Tests: ScaleEncoder
# ============================================================================


class TestScaleEncoder:
    """Tests for ScaleEncoder class."""

    def test_vocab_size_major_scale(self):
        """Test vocabulary size for major scale."""
        from coremusic.music.neural import ScaleEncoder
        from coremusic.music.theory import Note, Scale, ScaleType

        scale = Scale(Note("C", 4), ScaleType.MAJOR)
        encoder = ScaleEncoder(scale, octave_range=(3, 6))

        # 7 degrees * 3 octaves + 2 special tokens (REST, OUT_OF_SCALE)
        assert encoder.vocab_size == 7 * 3 + 2
        assert encoder.degrees_per_octave == 7
        assert encoder.num_octaves == 3

    def test_vocab_size_pentatonic_scale(self):
        """Test vocabulary size for pentatonic scale."""
        from coremusic.music.neural import ScaleEncoder
        from coremusic.music.theory import Note, Scale, ScaleType

        scale = Scale(Note("C", 4), ScaleType.MAJOR_PENTATONIC)
        encoder = ScaleEncoder(scale, octave_range=(2, 7))

        # 5 degrees * 5 octaves + 2 special tokens
        assert encoder.vocab_size == 5 * 5 + 2
        assert encoder.degrees_per_octave == 5

    def test_encode_scale_notes(self):
        """Test encoding notes in the scale."""
        from coremusic.music.neural import ScaleEncoder
        from coremusic.music.theory import Note, Scale, ScaleType

        scale = Scale(Note("C", 4), ScaleType.MAJOR)
        encoder = ScaleEncoder(scale, octave_range=(3, 6))

        # C4 (MIDI 60) should encode to degree 0 in octave 1
        token = encoder.midi_to_token(60)
        assert token == 7  # octave_idx=1, degree=0: 1*7 + 0 = 7

        # E4 (MIDI 64) should encode to degree 2 in octave 1
        token = encoder.midi_to_token(64)
        assert token == 9  # octave_idx=1, degree=2: 1*7 + 2 = 9

        # G4 (MIDI 67) should encode to degree 4 in octave 1
        token = encoder.midi_to_token(67)
        assert token == 11  # octave_idx=1, degree=4: 1*7 + 4 = 11

    def test_decode_tokens(self):
        """Test decoding tokens back to MIDI."""
        from coremusic.music.neural import ScaleEncoder
        from coremusic.music.theory import Note, Scale, ScaleType

        scale = Scale(Note("C", 4), ScaleType.MAJOR)
        encoder = ScaleEncoder(scale, octave_range=(3, 6))

        # Token 7 (C4) should decode back to MIDI 60
        midi = encoder.token_to_midi(7)
        assert midi == 60

        # Token 9 (E4) should decode back to MIDI 64
        midi = encoder.token_to_midi(9)
        assert midi == 64

    def test_special_tokens(self):
        """Test REST and OUT_OF_SCALE tokens."""
        from coremusic.music.neural import ScaleEncoder
        from coremusic.music.theory import Note, Scale, ScaleType

        scale = Scale(Note("C", 4), ScaleType.MAJOR)
        encoder = ScaleEncoder(scale, octave_range=(3, 6))

        # REST token should return None
        midi = encoder.token_to_midi(encoder.REST_TOKEN)
        assert midi is None

        # OUT_OF_SCALE token should return None
        midi = encoder.token_to_midi(encoder.OUT_OF_SCALE_TOKEN)
        assert midi is None

    def test_snap_to_scale(self):
        """Test snapping chromatic notes to nearest scale tone."""
        from coremusic.music.neural import ScaleEncoder
        from coremusic.music.theory import Note, Scale, ScaleType

        scale = Scale(Note("C", 4), ScaleType.MAJOR)
        encoder = ScaleEncoder(scale, octave_range=(3, 6), snap_to_scale=True)

        # C#4 (MIDI 61) should snap to C4 (60) or D4 (62)
        token = encoder.midi_to_token(61)
        midi = encoder.token_to_midi(token)
        assert midi in [60, 62]  # Should snap to nearest scale tone

        # F#4 (MIDI 66) should snap to F4 (65) or G4 (67)
        token = encoder.midi_to_token(66)
        midi = encoder.token_to_midi(token)
        assert midi in [65, 67]

    def test_no_snap_returns_out_of_scale(self):
        """Test that chromatic notes return OUT_OF_SCALE when snap disabled."""
        from coremusic.music.neural import ScaleEncoder
        from coremusic.music.theory import Note, Scale, ScaleType

        scale = Scale(Note("C", 4), ScaleType.MAJOR)
        encoder = ScaleEncoder(scale, octave_range=(3, 6), snap_to_scale=False)

        # C#4 (MIDI 61) should return OUT_OF_SCALE token
        token = encoder.midi_to_token(61)
        assert token == encoder.OUT_OF_SCALE_TOKEN

    def test_encode_midi_events(self, sample_midi_events):
        """Test encoding list of MIDI events."""
        from coremusic.music.neural import ScaleEncoder
        from coremusic.music.theory import Note, Scale, ScaleType

        scale = Scale(Note("C", 4), ScaleType.MAJOR)
        encoder = ScaleEncoder(scale, octave_range=(2, 7))

        tokens = encoder.encode(sample_midi_events)

        # Should have some tokens
        assert len(tokens) > 0
        # All tokens should be valid
        for token in tokens:
            assert 0 <= token < encoder.vocab_size

    def test_decode_to_events(self):
        """Test decoding tokens to MIDI events."""
        from coremusic.music.neural import ScaleEncoder
        from coremusic.music.theory import Note, Scale, ScaleType

        scale = Scale(Note("C", 4), ScaleType.MAJOR)
        encoder = ScaleEncoder(scale, octave_range=(3, 6))

        # Encode C major triad
        tokens = [7, 9, 11]  # C4, E4, G4

        events = encoder.decode(tokens, tempo=120.0)

        # Should have note-on and note-off for each token
        note_ons = [e for e in events if e.is_note_on]
        assert len(note_ons) == 3
        assert note_ons[0].data1 == 60  # C4
        assert note_ons[1].data1 == 64  # E4
        assert note_ons[2].data1 == 67  # G4

    def test_transpose_by_degree(self):
        """Test transposing by scale degrees."""
        from coremusic.music.neural import ScaleEncoder
        from coremusic.music.theory import Note, Scale, ScaleType

        scale = Scale(Note("C", 4), ScaleType.MAJOR)
        encoder = ScaleEncoder(scale, octave_range=(3, 6))

        # C4 (token 7) transposed up 2 degrees should be E4 (token 9)
        tokens = [7]
        transposed = encoder.transpose_by_degree(tokens, 2)
        assert transposed == [9]

        # E4 (token 9) transposed up 2 degrees should be G4 (token 11)
        tokens = [9]
        transposed = encoder.transpose_by_degree(tokens, 2)
        assert transposed == [11]

    def test_transpose_wraps_octave(self):
        """Test that transposition wraps to next octave correctly."""
        from coremusic.music.neural import ScaleEncoder
        from coremusic.music.theory import Note, Scale, ScaleType

        scale = Scale(Note("C", 4), ScaleType.MAJOR)
        encoder = ScaleEncoder(scale, octave_range=(3, 6))

        # B4 (token 13, degree 6 octave 1) + 1 degree = C5 (token 14, degree 0 octave 2)
        tokens = [13]
        transposed = encoder.transpose_by_degree(tokens, 1)
        assert transposed == [14]

    def test_get_scale_notes(self):
        """Test getting all scale notes."""
        from coremusic.music.neural import ScaleEncoder
        from coremusic.music.theory import Note, Scale, ScaleType

        scale = Scale(Note("C", 4), ScaleType.MAJOR)
        encoder = ScaleEncoder(scale, octave_range=(4, 5))  # Just 1 octave

        notes = encoder.get_scale_notes()

        # Should have 7 notes for one octave
        assert len(notes) == 7

        # Check first few notes
        assert notes[0] == (0, 60, "C4")  # C4
        assert notes[1] == (1, 62, "D4")  # D4
        assert notes[2] == (2, 64, "E4")  # E4

    def test_different_keys(self):
        """Test encoding works for different keys."""
        from coremusic.music.neural import ScaleEncoder
        from coremusic.music.theory import Note, Scale, ScaleType

        # G major scale
        scale = Scale(Note("G", 4), ScaleType.MAJOR)
        encoder = ScaleEncoder(scale, octave_range=(4, 5))

        # G4 (MIDI 67) should be degree 0 in G major
        token = encoder.midi_to_token(67)
        midi = encoder.token_to_midi(token)
        assert midi == 67

        # F#5 (MIDI 78) is in G major at scale octave 4, should encode properly
        # Note: F#4 (MIDI 66) would be in scale octave 3, below our range
        token = encoder.midi_to_token(78)
        midi = encoder.token_to_midi(token)
        assert midi == 78

    def test_minor_scale(self):
        """Test encoding with minor scale."""
        from coremusic.music.neural import ScaleEncoder
        from coremusic.music.theory import Note, Scale, ScaleType

        # A minor scale
        scale = Scale(Note("A", 4), ScaleType.NATURAL_MINOR)
        encoder = ScaleEncoder(scale, octave_range=(4, 5))

        # A4 (MIDI 69) should be degree 0
        token = encoder.midi_to_token(69)
        assert encoder.token_to_midi(token) == 69

        # C5 (MIDI 72) is b3 in A minor, should be degree 2
        token = encoder.midi_to_token(72)
        assert encoder.token_to_midi(token) == 72

    def test_roundtrip(self, sample_midi_events):
        """Test encode-decode roundtrip preserves notes in scale."""
        from coremusic.music.neural import ScaleEncoder
        from coremusic.music.theory import Note, Scale, ScaleType

        scale = Scale(Note("C", 4), ScaleType.MAJOR)
        encoder = ScaleEncoder(scale, octave_range=(2, 7), snap_to_scale=True)

        tokens = encoder.encode(sample_midi_events)
        events = encoder.decode(tokens, tempo=120.0)

        # All decoded notes should be in the C major scale
        for event in events:
            if event.is_note_on:
                pc = event.data1 % 12
                assert pc in [0, 2, 4, 5, 7, 9, 11], f"Note {event.data1} not in C major"


# ============================================================================
# Phase 5 Tests: MusicMetrics
# ============================================================================


class TestMusicMetrics:
    """Tests for MusicMetrics class."""

    def test_pitch_histogram(self):
        """Test pitch histogram computation."""
        sequence = [60, 62, 64, 60, 60]  # C4 appears 3 times
        histogram = MusicMetrics.pitch_histogram(sequence, normalize=False)

        assert len(histogram) == 128
        assert histogram[60] == 3.0
        assert histogram[62] == 1.0
        assert histogram[64] == 1.0

    def test_pitch_histogram_normalized(self):
        """Test normalized pitch histogram."""
        sequence = [60, 62, 64, 60, 60]
        histogram = MusicMetrics.pitch_histogram(sequence, normalize=True)

        assert abs(sum(histogram) - 1.0) < 0.001
        assert histogram[60] == pytest.approx(0.6, rel=0.01)

    def test_pitch_histogram_similarity_identical(self):
        """Test similarity of identical sequences."""
        sequence = [60, 62, 64, 65, 67]
        similarity = MusicMetrics.pitch_histogram_similarity(sequence, sequence)

        assert similarity == pytest.approx(1.0, rel=0.01)

    def test_pitch_histogram_similarity_different(self):
        """Test similarity of different sequences."""
        seq1 = [60, 62, 64]  # C major triad
        seq2 = [61, 63, 65]  # C# major triad

        similarity = MusicMetrics.pitch_histogram_similarity(seq1, seq2)
        assert 0.0 <= similarity <= 1.0
        assert similarity < 1.0  # Should not be identical

    def test_pitch_histogram_similarity_empty(self):
        """Test similarity with empty sequences."""
        similarity = MusicMetrics.pitch_histogram_similarity([], [60, 62, 64])
        assert similarity == 0.0

    def test_interval_histogram(self):
        """Test interval histogram computation."""
        sequence = [60, 62, 64]  # Two +2 intervals
        histogram = MusicMetrics.interval_histogram(sequence, normalize=False)

        # Intervals: +2, +2
        # Index for +2 is 24 + 2 = 26
        assert len(histogram) == 49
        assert histogram[26] == 2.0

    def test_interval_histogram_similarity(self):
        """Test interval similarity of sequences."""
        seq1 = [60, 62, 64, 65]  # +2, +2, +1
        seq2 = [48, 50, 52, 53]  # Same intervals, different octave

        similarity = MusicMetrics.interval_histogram_similarity(seq1, seq2)
        assert similarity == pytest.approx(1.0, rel=0.01)

    def test_repetition_score_no_repetition(self):
        """Test repetition score with no repetition."""
        sequence = [60, 62, 64, 65, 67, 69, 71, 72]  # All unique
        score = MusicMetrics.repetition_score(sequence, n=4)

        # All 4-grams are unique
        assert score == pytest.approx(0.0, abs=0.01)

    def test_repetition_score_full_repetition(self):
        """Test repetition score with full repetition."""
        sequence = [60, 62, 64, 65] * 5  # Fully repeated
        score = MusicMetrics.repetition_score(sequence, n=4)

        # High repetition (score depends on exact n-gram count)
        # With 20 tokens and n=4: 17 n-grams, only 4 unique = 1 - 4/17 = 0.76
        assert score > 0.7

    def test_repetition_score_short_sequence(self):
        """Test repetition score with sequence shorter than n."""
        sequence = [60, 62]
        score = MusicMetrics.repetition_score(sequence, n=4)
        assert score == 0.0

    def test_unique_notes(self):
        """Test unique note counting."""
        sequence = [60, 62, 64, 60, 62, 65]  # 4 unique notes
        unique = MusicMetrics.unique_notes(sequence)
        assert unique == 4

    def test_average_interval(self):
        """Test average interval computation."""
        sequence = [60, 64, 60, 64]  # Intervals: +4, -4, +4
        avg = MusicMetrics.average_interval(sequence)
        assert avg == pytest.approx(4.0, rel=0.01)

    def test_pitch_range_from_sequence(self):
        """Test pitch range computation."""
        sequence = [60, 72, 48, 84, 55]
        min_pitch, max_pitch = MusicMetrics.pitch_range_from_sequence(sequence)

        assert min_pitch == 48
        assert max_pitch == 84

    def test_evaluate_sequence(self):
        """Test comprehensive sequence evaluation."""
        sequence = [60, 62, 64, 65, 67, 69, 71, 72]
        metrics = MusicMetrics.evaluate_sequence(sequence)

        assert "length" in metrics
        assert metrics["length"] == 8.0
        assert "unique_notes" in metrics
        assert "repetition_4gram" in metrics
        assert "average_interval" in metrics
        assert "pitch_range" in metrics

    def test_evaluate_sequence_with_reference(self):
        """Test evaluation with reference sequence."""
        generated = [60, 62, 64, 65, 67]
        reference = [60, 62, 64, 67, 69]

        metrics = MusicMetrics.evaluate_sequence(generated, reference)

        assert "pitch_similarity" in metrics
        assert "interval_similarity" in metrics
        assert 0.0 <= metrics["pitch_similarity"] <= 1.0


# ============================================================================
# Phase 5 Tests: ModelComparison
# ============================================================================


class TestModelComparison:
    """Tests for ModelComparison class."""

    def test_model_result_dataclass(self):
        """Test ModelResult dataclass."""
        result = ModelResult(
            name="TestModel",
            loss=0.5,
            metrics={"accuracy": 0.8},
            generated_samples=[[60, 62, 64]],
        )

        assert result.name == "TestModel"
        assert result.loss == 0.5
        assert result.metrics["accuracy"] == 0.8
        assert len(result.generated_samples) == 1

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required")
    def test_model_comparison_init(self, sample_midi_file):
        """Test ModelComparison initialization."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)
        dataset.load_file(sample_midi_file)

        comparison = ModelComparison(dataset, encoder)

        assert comparison.dataset is dataset
        assert comparison.encoder is encoder
        assert comparison.results == {}

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required")
    def test_model_comparison_evaluate(self, sample_midi_file):
        """Test evaluating a model."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)
        dataset.load_file(sample_midi_file)

        model = create_mlp_model(8 * 128, [32], 128)

        comparison = ModelComparison(dataset, encoder)
        result = comparison.evaluate_model(
            "TestMLP", model, num_samples=2, generation_length=16
        )

        assert result.name == "TestMLP"
        assert result.loss >= 0
        assert "loss" in result.metrics
        assert len(result.generated_samples) == 2

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required")
    def test_model_comparison_compare(self, sample_midi_file):
        """Test comparing multiple models."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)
        dataset.load_file(sample_midi_file)

        model1 = create_mlp_model(8 * 128, [32], 128)
        model2 = create_mlp_model(8 * 128, [64], 128)

        comparison = ModelComparison(dataset, encoder)
        comparison.evaluate_model("Small", model1, num_samples=1, generation_length=8)
        comparison.evaluate_model("Large", model2, num_samples=1, generation_length=8)

        results = comparison.compare()

        assert "Small" in results
        assert "Large" in results
        assert "loss" in results["Small"]

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required")
    def test_model_comparison_best_model(self, sample_midi_file):
        """Test finding best model."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)
        dataset.load_file(sample_midi_file)

        comparison = ModelComparison(dataset, encoder)

        # Add results directly
        comparison.results["ModelA"] = ModelResult("ModelA", 0.5, {"loss": 0.5})
        comparison.results["ModelB"] = ModelResult("ModelB", 0.3, {"loss": 0.3})

        best = comparison.best_model("loss", lower_is_better=True)
        assert best == "ModelB"

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required")
    def test_model_comparison_table(self, sample_midi_file):
        """Test comparison table generation."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)
        dataset.load_file(sample_midi_file)

        comparison = ModelComparison(dataset, encoder)
        comparison.results["TestModel"] = ModelResult(
            "TestModel", 0.5, {"loss": 0.5, "accuracy": 0.8}
        )

        table = comparison.compare_table()

        assert "TestModel" in table
        assert "loss" in table

    def test_model_comparison_empty(self):
        """Test comparison with no models."""
        encoder = NoteEncoder()
        dataset = MIDIDataset(encoder, seq_length=8)
        dataset.add_sequence([60, 62, 64, 65, 67, 69, 71, 72] * 3)

        comparison = ModelComparison(dataset, encoder)

        assert comparison.best_model() == ""
        assert "No models" in comparison.compare_table()


# ============================================================================
# Phase 6 Tests: High-level API
# ============================================================================


class TestHighLevelAPI:
    """Tests for high-level API functions."""

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required for training")
    def test_train_music_model(self, sample_midi_file, tmp_path):
        """Test train_music_model function."""
        output_path = str(tmp_path / "model.kan")

        model = train_music_model(
            midi_files=sample_midi_file,
            model_type="mlp",
            output_path=output_path,
            encoder_type="note",
            seq_length=8,
            hidden_size=32,
            max_epochs=2,
            early_stopping_patience=0,
            validation_split=0.0,
            augment=False,
            verbose=0,
        )

        assert isinstance(model, NeuralNetwork)
        assert os.path.exists(output_path)

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required for training")
    def test_train_music_model_with_list(self, sample_midi_file, tmp_path):
        """Test train_music_model with file list."""
        output_path = str(tmp_path / "model.kan")

        model = train_music_model(
            midi_files=[sample_midi_file],
            model_type="mlp",
            output_path=output_path,
            seq_length=8,
            hidden_size=32,
            max_epochs=1,
            early_stopping_patience=0,
            validation_split=0.0,
            verbose=0,
        )

        assert isinstance(model, NeuralNetwork)

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required for training")
    def test_train_music_model_lstm(self, sample_midi_file, tmp_path):
        """Test training LSTM model."""
        output_path = str(tmp_path / "lstm_model.kan")

        model = train_music_model(
            midi_files=sample_midi_file,
            model_type="lstm",
            output_path=output_path,
            seq_length=8,
            hidden_size=32,
            max_epochs=1,
            early_stopping_patience=0,
            validation_split=0.0,
            verbose=0,
        )

        assert isinstance(model, NeuralNetwork)

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required for training")
    def test_generate_music(self, sample_midi_file, tmp_path):
        """Test generate_music function."""
        # First train a model
        model_path = str(tmp_path / "model.kan")
        train_music_model(
            midi_files=sample_midi_file,
            model_type="mlp",
            output_path=model_path,
            seq_length=8,
            hidden_size=32,
            max_epochs=1,
            early_stopping_patience=0,
            validation_split=0.0,
            verbose=0,
        )

        # Generate music
        output_path = str(tmp_path / "generated.mid")
        result = generate_music(
            model_path=model_path,
            output_path=output_path,
            duration=16,
            temperature=0.8,
            encoder_type="note",
            seq_length=8,
        )

        assert result == output_path
        assert os.path.exists(output_path)

        # Verify the generated file is valid MIDI
        loaded = MIDISequence.load(output_path)
        assert len(loaded.tracks) > 0

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required for training")
    def test_continue_music(self, sample_midi_file, tmp_path):
        """Test continue_music function."""
        # First train a model
        model_path = str(tmp_path / "model.kan")
        train_music_model(
            midi_files=sample_midi_file,
            model_type="mlp",
            output_path=model_path,
            seq_length=8,
            hidden_size=32,
            max_epochs=1,
            early_stopping_patience=0,
            validation_split=0.0,
            verbose=0,
        )

        # Continue the sample file
        output_path = str(tmp_path / "continued.mid")
        result = continue_music(
            model_path=model_path,
            input_midi=sample_midi_file,
            output_path=output_path,
            bars=2,
            temperature=0.8,
            encoder_type="note",
            seq_length=8,
        )

        assert result == output_path
        assert os.path.exists(output_path)

    def test_train_no_files_raises(self, tmp_path):
        """Test that training with no valid files raises error."""
        with pytest.raises(ValueError, match="No MIDI files"):
            train_music_model(
                midi_files="/nonexistent/path/*.mid",
                model_type="mlp",
                output_path=str(tmp_path / "model.kan"),
                verbose=0,
            )


# ============================================================================
# Phase 6 Tests: CLI Integration
# ============================================================================


class TestNeuralCLI:
    """Tests for neural CLI commands."""

    def test_cli_import(self):
        """Test that CLI module can be imported."""
        from coremusic.cli import neural
        assert hasattr(neural, "register")
        assert hasattr(neural, "cmd_train")
        assert hasattr(neural, "cmd_generate")
        assert hasattr(neural, "cmd_continue")
        assert hasattr(neural, "cmd_evaluate")
        assert hasattr(neural, "cmd_info")

    def test_cli_model_types(self):
        """Test that CLI has correct model types."""
        from coremusic.cli.neural import MODEL_TYPES
        assert "mlp" in MODEL_TYPES
        assert "rnn" in MODEL_TYPES
        assert "lstm" in MODEL_TYPES
        assert "gru" in MODEL_TYPES

    def test_cli_encoder_types(self):
        """Test that CLI has correct encoder types."""
        from coremusic.cli.neural import ENCODER_TYPES
        assert "note" in ENCODER_TYPES
        assert "event" in ENCODER_TYPES

    def test_cli_register(self):
        """Test CLI registration."""
        import argparse
        from coremusic.cli import neural

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        neural.register(subparsers)

        # Verify neural subcommand was registered by checking it doesn't error
        # (--help causes SystemExit, so we just verify parsing without help)
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["neural", "--help"])
        # Help should exit with 0
        assert exc_info.value.code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
