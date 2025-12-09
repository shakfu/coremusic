#!/usr/bin/env python3
"""LSTM Neural Network Music Generation Demo.

This demo trains a Long Short-Term Memory (LSTM) recurrent model on classical
MIDI files and generates new music. LSTMs are particularly well-suited for
sequential data like music as they can learn long-range dependencies.

Usage:
    python -m tests.demos.neural.lstm_demo

    Or from the project root:
    uv run python tests/demos/neural/lstm_demo.py

Output:
    - Trained model: build/midi_files/neural/lstm_model.kan
    - Generated MIDI: build/midi_files/neural/lstm_generated.mid
"""

import os
import glob
import sys
import time

from coremusic.music.neural import (
    NoteEncoder,
    EventEncoder,
    MIDIDataset,
    ModelFactory,
    RNN_NORM,
    TrainingConfig,
    Trainer,
    MusicGenerator,
    TemperatureSampling,
    TopKSampling,
)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
CLASSICAL_DIR = os.path.join(PROJECT_ROOT, "tests", "data", "midi", "classical")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "build", "midi_files", "neural")


def print_header(title: str, char: str = "=", width: int = 70):
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def print_section(title: str, char: str = "-", width: int = 50):
    """Print a section header."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def is_valid_midi_file(filepath: str) -> bool:
    """Check if a file is a valid Standard MIDI File."""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4)
            return header == b'MThd'
    except Exception:
        return False


def main():
    """Run the LSTM training and generation demo."""
    print_header("LSTM Neural Network Music Generation Demo")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get MIDI files
    print_section("Loading MIDI Files")
    all_files = sorted(glob.glob(os.path.join(CLASSICAL_DIR, "*.mid")))
    midi_files = [f for f in all_files if is_valid_midi_file(f)]

    if not midi_files:
        print(f"ERROR: No valid MIDI files found in {CLASSICAL_DIR}")
        return 1

    # Use more files for LSTM (it benefits from more data)
    midi_files = midi_files[:12]
    print(f"Found {len(midi_files)} MIDI files:")
    for f in midi_files:
        size_kb = os.path.getsize(f) / 1024
        print(f"  - {os.path.basename(f)} ({size_kb:.1f} KB)")

    # Create encoder and dataset
    print_section("Creating Dataset")
    encoder = NoteEncoder()
    seq_length = 64  # Longer sequences for LSTM
    dataset = MIDIDataset(encoder, seq_length=seq_length)

    # Load files
    total_loaded = 0
    for midi_file in midi_files:
        try:
            dataset.load_file(midi_file)
            total_loaded += 1
            print(f"  Loaded: {os.path.basename(midi_file)}")
        except Exception as e:
            print(f"  Warning: Failed to load {os.path.basename(midi_file)}: {e}")

    print(f"\nLoaded {total_loaded} files")
    print(f"Total tokens: {dataset.total_tokens:,}")
    print(f"Sequences: {dataset.n_sequences:,}")

    if dataset.n_sequences == 0:
        print("ERROR: No sequences extracted from MIDI files")
        return 1

    # Apply data augmentation
    print_section("Data Augmentation")
    original_sequences = dataset.n_sequences
    dataset.augment(transpose_range=(-5, 7))
    print(f"Original sequences: {original_sequences:,}")
    print(f"After augmentation: {dataset.n_sequences:,}")
    print(f"Augmentation factor: {dataset.n_sequences / original_sequences:.1f}x")

    # Create LSTM model
    # Larger hidden size helps LSTM learn more complex patterns
    # RNN_NORM enables layer normalization for better gradient flow
    print_section("Creating LSTM Model")
    hidden_size = 256  # Increased from 128 for better capacity
    model = ModelFactory.create(
        model_type="lstm",
        encoder=encoder,
        seq_length=seq_length,
        hidden_size=hidden_size,
        rnn_flags=RNN_NORM,  # Enable layer normalization
    )
    print(f"Model type: LSTM (Long Short-Term Memory)")
    print(f"Sequence length: {seq_length}")
    print(f"Hidden size: {hidden_size}")
    print(f"Layer normalization: enabled (RNN_NORM)")
    print(f"Input dimension: {model.input_dim}")
    print(f"Output dimension: {model.output_dim}")
    print(f"Total parameters: {model.n_var:,}")

    # Get sequences for RNN training
    print_section("Preparing Training Data")
    sequences = dataset.sequences
    print(f"Training sequences: {len(sequences)}")
    print(f"Total tokens: {dataset.total_tokens:,}")

    # Configure training
    # LSTM models with BPTT need:
    # - Moderate learning rate for BPTT
    # - Smaller batches for better gradient estimates
    # - Gradient clipping for stability
    print_section("Training Configuration")
    config = TrainingConfig(
        learning_rate=0.005,  # Moderate for BPTT
        batch_size=8,  # Smaller batches for BPTT
        max_epochs=50,  # More epochs for convergence
        min_epochs=10,
        early_stopping_patience=12,
        validation_split=0.1,
        verbose=1,
    )
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Max epochs: {config.max_epochs}")
    print(f"Early stopping patience: {config.early_stopping_patience}")
    print(f"Training method: BPTT (train_rnn_sequences)")

    # Train model with proper BPTT
    print_section("Training Model with BPTT")
    print("(Using backpropagation through time for proper RNN training...)")
    trainer = Trainer(model, config)

    start_time = time.time()
    history = trainer.train_rnn_sequences(
        sequences=sequences,
        seq_length=seq_length,
        vocab_size=encoder.vocab_size,
        grad_clip=5.0,
    )
    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.1f} seconds")
    print(f"Epochs trained: {len(history['loss'])}")
    print(f"Final training loss: {history['loss'][-1]:.6f}")

    # Validation metrics and overfitting detection
    if history['val_loss']:
        final_val_loss = history['val_loss'][-1]
        print(f"Final validation loss: {final_val_loss:.6f}")

        # Overfitting indicator
        gap = final_val_loss - history['loss'][-1]
        if gap > 0.5:
            print(f"WARNING: Possible overfitting (gap: {gap:.3f})")
        elif gap > 0.2:
            print(f"Note: Some overfitting (gap: {gap:.3f})")
        else:
            print(f"Good generalization (gap: {gap:.3f})")

        # Best validation epoch
        best_val_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
        best_val_loss = min(history['val_loss'])
        print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_val_epoch}")

    # Save model
    print_section("Saving Model")
    model_path = os.path.join(OUTPUT_DIR, "lstm_model.kan")
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    print(f"Model file size: {os.path.getsize(model_path):,} bytes")

    # Generate music with different sampling strategies
    print_section("Generating Music")

    # Temperature sampling
    print("\n1. Temperature Sampling:")
    generator = MusicGenerator(
        model, encoder, seq_length,
        sampling=TemperatureSampling(temperature=0.7)
    )

    for temp in [0.5, 0.7, 1.0]:
        generator.sampling = TemperatureSampling(temperature=temp)
        output_path = os.path.join(OUTPUT_DIR, f"lstm_temp{temp}.mid")
        sequence = generator.generate_midi(
            duration_beats=64,
            tempo=110.0,
            track_name=f"LSTM (temp={temp})"
        )
        sequence.save(output_path)
        print(f"   Generated: {os.path.basename(output_path)}")

    # Top-K sampling
    print("\n2. Top-K Sampling:")
    for k in [5, 10, 20]:
        generator.sampling = TopKSampling(k=k)
        output_path = os.path.join(OUTPUT_DIR, f"lstm_topk{k}.mid")
        sequence = generator.generate_midi(
            duration_beats=64,
            tempo=110.0,
            track_name=f"LSTM (top-k={k})"
        )
        sequence.save(output_path)
        print(f"   Generated: {os.path.basename(output_path)}")

    # Generate longer piece
    print("\n3. Extended Generation:")
    generator.sampling = TemperatureSampling(temperature=0.8)
    output_path = os.path.join(OUTPUT_DIR, "lstm_extended.mid")
    sequence = generator.generate_midi(
        duration_beats=128,  # Longer piece
        tempo=100.0,
        track_name="LSTM Extended"
    )
    sequence.save(output_path)
    print(f"   Generated: {os.path.basename(output_path)} (128 beats)")

    # Summary
    print_header("Demo Complete")
    print(f"Model saved: {model_path}")
    print(f"Generated files in: {OUTPUT_DIR}")
    print("\nLSTM models trained with BPTT (backpropagation through time) can learn")
    print("long-range temporal dependencies, producing more coherent musical structure")
    print("compared to MLPs which only see fixed windows of context.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
