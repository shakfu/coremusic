#!/usr/bin/env python3
"""High-level API for neural network music training and generation.

This module provides simple one-liner functions for common tasks:
- train_music_model: Train a model on MIDI files
- generate_music: Generate music from a trained model
- continue_music: Continue an existing MIDI file

Example:
    >>> from coremusic.music.neural import train_music_model, generate_music
    >>>
    >>> # Train a model
    >>> model = train_music_model(
    ...     ['bach/*.mid', 'beethoven/*.mid'],
    ...     model_type='lstm',
    ...     output_path='model.kan'
    ... )
    >>>
    >>> # Generate music
    >>> generate_music('model.kan', 'output.mid', duration=64, temperature=0.8)
"""

import glob
import logging
from typing import List, Optional, Union

from coremusic.kann import NeuralNetwork

from .data import BaseEncoder, EventEncoder, MIDIDataset, NoteEncoder
from .generation import MusicGenerator, TemperatureSampling
from .models import ModelFactory
from .training import EarlyStopping, ModelCheckpoint, Trainer, TrainingConfig

logger = logging.getLogger(__name__)


def train_music_model(
    midi_files: Union[str, List[str]],
    model_type: str = "lstm",
    output_path: Optional[str] = None,
    encoder_type: str = "note",
    seq_length: int = 32,
    hidden_size: int = 256,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    max_epochs: int = 100,
    early_stopping_patience: int = 10,
    validation_split: float = 0.1,
    augment: bool = True,
    transpose_range: tuple = (-5, 7),
    verbose: int = 1,
    **kwargs,
) -> NeuralNetwork:
    """Train a neural network model on MIDI files.

    This is a high-level convenience function that handles:
    - Loading and encoding MIDI files
    - Data augmentation
    - Model creation
    - Training with early stopping
    - Model saving

    Args:
        midi_files: Path(s) to MIDI files (supports glob patterns)
        model_type: Model architecture ('mlp', 'rnn', 'lstm', 'gru')
        output_path: Path to save trained model (optional)
        encoder_type: Encoder type ('note' or 'event')
        seq_length: Sequence length for training
        hidden_size: Hidden layer size
        learning_rate: Learning rate
        batch_size: Mini-batch size
        max_epochs: Maximum training epochs
        early_stopping_patience: Stop after this many epochs without improvement
        validation_split: Fraction of data for validation
        augment: Whether to apply data augmentation
        transpose_range: Transposition range for augmentation
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        **kwargs: Additional arguments passed to model factory

    Returns:
        Trained NeuralNetwork

    Example:
        >>> model = train_music_model(
        ...     'classical/*.mid',
        ...     model_type='lstm',
        ...     output_path='classical_model.kan',
        ...     max_epochs=100
        ... )
    """
    # Create encoder
    encoder: BaseEncoder
    if encoder_type == "event":
        encoder = EventEncoder()
    else:
        encoder = NoteEncoder()

    # Create dataset
    dataset = MIDIDataset(encoder, seq_length=seq_length)

    # Load files
    if isinstance(midi_files, str):
        midi_files = [midi_files]

    total_files = 0
    for pattern in midi_files:
        # Expand glob patterns
        matches = glob.glob(pattern, recursive=True)
        if matches:
            for path in matches:
                try:
                    dataset.load_file(path)
                    total_files += 1
                except Exception as e:
                    if verbose > 0:
                        logger.warning(f"Failed to load {path}: {e}")
        else:
            # Try as direct path
            try:
                dataset.load_file(pattern)
                total_files += 1
            except Exception as e:
                if verbose > 0:
                    logger.warning(f"Failed to load {pattern}: {e}")

    if total_files == 0:
        raise ValueError("No MIDI files loaded")

    if verbose > 0:
        logger.info(f"Loaded {total_files} MIDI files, {dataset.total_tokens} tokens")

    # Augment data
    if augment:
        dataset.augment(transpose_range=transpose_range)
        if verbose > 0:
            logger.info(f"Augmented to {dataset.n_sequences} sequences")

    # Prepare training data
    x_train, y_train = dataset.prepare_training_data(use_numpy=True)

    if verbose > 0:
        logger.info(f"Training samples: {x_train.shape[0]}")

    # Create model
    model = ModelFactory.create(
        model_type=model_type,
        encoder=encoder,
        seq_length=seq_length,
        hidden_size=hidden_size,
        **kwargs,
    )

    if verbose > 0:
        logger.info(f"Created {model_type.upper()} model with {model.n_var} parameters")

    # Configure training
    config = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        validation_split=validation_split,
        verbose=verbose,
    )

    # Train
    trainer = Trainer(model, config)
    if early_stopping_patience > 0:
        trainer.add_callback(EarlyStopping(patience=early_stopping_patience))

    if output_path:
        trainer.add_callback(ModelCheckpoint(output_path, save_best_only=True, verbose=verbose > 0))

    trainer.train(x_train, y_train)

    # Save final model if path provided and not using checkpoint
    if output_path and early_stopping_patience <= 0:
        model.save(output_path)
        if verbose > 0:
            logger.info(f"Saved model to {output_path}")

    return model


def generate_music(
    model_path: str,
    output_path: str,
    duration: int = 32,
    temperature: float = 1.0,
    seed_midi: Optional[str] = None,
    encoder_type: str = "note",
    seq_length: int = 32,
    tempo: float = 120.0,
    track_name: str = "Generated",
) -> str:
    """Generate music from a trained model.

    Args:
        model_path: Path to trained model (.kan file)
        output_path: Path to save generated MIDI
        duration: Duration in beats
        temperature: Sampling temperature (lower = more conservative)
        seed_midi: Optional MIDI file to use as seed
        encoder_type: Encoder type used during training ('note' or 'event')
        seq_length: Sequence length used during training
        tempo: Output tempo in BPM
        track_name: Name for generated track

    Returns:
        Path to generated MIDI file

    Example:
        >>> generate_music(
        ...     'model.kan',
        ...     'output.mid',
        ...     duration=64,
        ...     temperature=0.8
        ... )
    """
    # Load model
    model = NeuralNetwork.load(model_path)

    # Create encoder
    encoder: BaseEncoder
    if encoder_type == "event":
        encoder = EventEncoder()
    else:
        encoder = NoteEncoder()

    # Create generator
    generator = MusicGenerator(
        model, encoder, seq_length,
        sampling=TemperatureSampling(temperature)
    )

    # Generate
    sequence = generator.generate_midi(
        seed_midi=seed_midi,
        duration_beats=duration,
        tempo=tempo,
        track_name=track_name,
    )

    # Save
    sequence.save(output_path)

    return output_path


def continue_music(
    model_path: str,
    input_midi: str,
    output_path: str,
    bars: int = 8,
    temperature: float = 1.0,
    encoder_type: str = "note",
    seq_length: int = 32,
    time_signature: tuple = (4, 4),
) -> str:
    """Continue an existing MIDI file using a trained model.

    Args:
        model_path: Path to trained model (.kan file)
        input_midi: Path to input MIDI file to continue
        output_path: Path to save continued MIDI
        bars: Number of bars to add
        temperature: Sampling temperature
        encoder_type: Encoder type used during training
        seq_length: Sequence length used during training
        time_signature: Time signature (beats_per_bar, beat_unit)

    Returns:
        Path to output MIDI file

    Example:
        >>> continue_music(
        ...     'model.kan',
        ...     'seed.mid',
        ...     'continued.mid',
        ...     bars=8
        ... )
    """
    # Load model
    model = NeuralNetwork.load(model_path)

    # Create encoder
    encoder: BaseEncoder
    if encoder_type == "event":
        encoder = EventEncoder()
    else:
        encoder = NoteEncoder()

    # Create generator
    generator = MusicGenerator(
        model, encoder, seq_length,
        sampling=TemperatureSampling(temperature)
    )

    # Calculate beats from bars
    beats_per_bar = time_signature[0]
    continuation_beats = bars * beats_per_bar

    # Continue
    sequence = generator.continue_sequence(
        input_midi,
        continuation_length=continuation_beats,
    )

    # Save
    sequence.save(output_path)

    return output_path


def quick_train_and_generate(
    midi_file: str,
    output_midi: str,
    model_type: str = "mlp",
    epochs: int = 50,
    temperature: float = 0.8,
    verbose: int = 1,
) -> str:
    """Quick training and generation for experimentation.

    Trains a small model on a single MIDI file and generates new music.
    Useful for quick experiments and demos.

    Args:
        midi_file: Path to MIDI file for training
        output_midi: Path to save generated MIDI
        model_type: Model type ('mlp', 'lstm', 'gru', 'rnn')
        epochs: Number of training epochs
        temperature: Generation temperature
        verbose: Verbosity level

    Returns:
        Path to generated MIDI file

    Example:
        >>> quick_train_and_generate('bach.mid', 'output.mid')
    """
    import os
    import tempfile

    # Create temporary model file
    with tempfile.NamedTemporaryFile(suffix=".kan", delete=False) as f:
        model_path = f.name

    try:
        # Train
        train_music_model(
            midi_file,
            model_type=model_type,
            output_path=model_path,
            seq_length=16,
            hidden_size=128,
            max_epochs=epochs,
            early_stopping_patience=0,  # No early stopping for quick train
            validation_split=0.0,
            augment=True,
            transpose_range=(-3, 3),
            verbose=verbose,
        )

        # Generate
        generate_music(
            model_path,
            output_midi,
            duration=32,
            temperature=temperature,
            seq_length=16,
        )

        return output_midi

    finally:
        # Clean up temporary model
        if os.path.exists(model_path):
            os.remove(model_path)
