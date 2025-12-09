#!/usr/bin/env python3
"""Neural network-based music learning and generation.

This module provides high-level APIs for training neural networks on MIDI data
and generating new music. It builds on:

- coremusic.kann - Low-level KANN neural network wrapper
- coremusic.midi.utilities - MIDI file I/O (MIDISequence, MIDIEvent)
- coremusic.music.theory - Music theory utilities

Submodules:
- data: MIDI data encoding and preprocessing
- models: Pre-built neural network architectures
- training: Training utilities and callbacks
- generation: Music generation strategies

Example:
    >>> from coremusic.music.neural import (
    ...     NoteEncoder, MIDIDataset, create_lstm_model,
    ...     Trainer, TrainingConfig, MusicGenerator
    ... )
    >>>
    >>> # Load and encode MIDI data
    >>> encoder = NoteEncoder()
    >>> dataset = MIDIDataset(encoder, seq_length=32)
    >>> dataset.load_file('music.mid')
    >>>
    >>> # Prepare training data
    >>> x_train, y_train = dataset.prepare_training_data()
    >>>
    >>> # Create and train model
    >>> model = create_lstm_model(
    ...     input_size=encoder.vocab_size,
    ...     hidden_size=256,
    ...     output_size=encoder.vocab_size
    ... )
    >>> trainer = Trainer(model, TrainingConfig(max_epochs=100))
    >>> trainer.train(x_train, y_train)
    >>>
    >>> # Generate new music
    >>> generator = MusicGenerator(model, encoder, seq_length=32)
    >>> sequence = generator.generate_midi(duration_beats=32)
    >>> sequence.save('output.mid')
"""

from .api import (continue_music, generate_music,  # High-level API functions
                  quick_train_and_generate, train_music_model)
from .data import (BaseEncoder,  # Base classes; Encoders; Dataset
                   EventEncoder, MIDIDataset, NoteEncoder, PianoRollEncoder,
                   RelativePitchEncoder)
from .evaluation import (ModelComparison,  # Metrics; Model comparison
                         ModelResult, MusicMetrics)
from .generation import (GreedySampling,  # Sampling strategies; Generator
                         MusicGenerator, NucleusSampling, SamplingStrategy,
                         TemperatureSampling, TopKSampling)
from .models import (  # Model factory functions; Model factory class; RNN flags
    ModelFactory, RNN_NORM, RNN_VAR_H0, create_gru_model, create_lstm_model,
    create_mlp_model, create_rnn_model)
from .training import (Callback,  # Configuration; Trainer; Callbacks
                       EarlyStopping, LearningRateScheduler, ModelCheckpoint,
                       ProgressLogger, Trainer, TrainingConfig)

__all__ = [
    # Data - Encoders
    "BaseEncoder",
    "NoteEncoder",
    "EventEncoder",
    "PianoRollEncoder",
    "RelativePitchEncoder",
    # Data - Dataset
    "MIDIDataset",
    # Models - Factory functions
    "create_mlp_model",
    "create_rnn_model",
    "create_lstm_model",
    "create_gru_model",
    "ModelFactory",
    # Models - RNN flags
    "RNN_NORM",
    "RNN_VAR_H0",
    # Training - Configuration
    "TrainingConfig",
    # Training - Trainer
    "Trainer",
    # Training - Callbacks
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "ProgressLogger",
    "LearningRateScheduler",
    # Generation - Sampling
    "SamplingStrategy",
    "GreedySampling",
    "TemperatureSampling",
    "TopKSampling",
    "NucleusSampling",
    # Generation - Generator
    "MusicGenerator",
    # Evaluation - Metrics
    "MusicMetrics",
    # Evaluation - Model comparison
    "ModelResult",
    "ModelComparison",
    # API - High-level functions
    "train_music_model",
    "generate_music",
    "continue_music",
    "quick_train_and_generate",
]
