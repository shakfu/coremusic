#!/usr/bin/env python3
"""MLP Neural Network Music Generation Demo.

This demo trains a Multi-Layer Perceptron (MLP) model on classical MIDI files
and generates new music. MLPs are simple feedforward networks that learn
patterns from fixed-length input sequences.

Usage:
    python -m tests.demos.neural.mlp_demo

    Or from the project root:
    uv run python tests/demos/neural/mlp_demo.py

Output:
    - Trained model: build/midi_files/neural/mlp_model.kan
    - Generated MIDI: build/midi_files/neural/mlp_generated.mid
"""

import os
import glob
import time

from coremusic.music.neural import (
    NoteEncoder,
    MIDIDataset,
    ModelFactory,
    TrainingConfig,
    Trainer,
    MusicGenerator,
    TemperatureSampling,
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
    """Run the MLP training and generation demo."""
    print_header("MLP Neural Network Music Generation Demo")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get MIDI files
    print_section("Loading MIDI Files")
    all_files = sorted(glob.glob(os.path.join(CLASSICAL_DIR, "*.mid")))
    midi_files = [f for f in all_files if is_valid_midi_file(f)]

    if not midi_files:
        print(f"ERROR: No valid MIDI files found in {CLASSICAL_DIR}")
        return 1

    # Use a subset for the demo (first 8 files)
    midi_files = midi_files[:8]
    print(f"Found {len(midi_files)} MIDI files:")
    for f in midi_files:
        print(f"  - {os.path.basename(f)}")

    # Create encoder and dataset
    print_section("Creating Dataset")
    encoder = NoteEncoder()
    seq_length = 32
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
    print(f"Total tokens: {dataset.total_tokens}")
    print(f"Sequences: {dataset.n_sequences}")

    if dataset.n_sequences == 0:
        print("ERROR: No sequences extracted from MIDI files")
        return 1

    # Apply data augmentation
    print_section("Data Augmentation")
    original_sequences = dataset.n_sequences
    dataset.augment(transpose_range=(-5, 7))
    print(f"Original sequences: {original_sequences}")
    print(f"After augmentation: {dataset.n_sequences}")

    # Create model
    print_section("Creating MLP Model")
    hidden_size = 256
    model = ModelFactory.create(
        model_type="mlp",
        encoder=encoder,
        seq_length=seq_length,
        hidden_size=hidden_size,
    )
    print(f"Model type: MLP (Multi-Layer Perceptron)")
    print(f"Sequence length: {seq_length}")
    print(f"Hidden size: {hidden_size}")
    print(f"Input dimension: {model.input_dim}")
    print(f"Output dimension: {model.output_dim}")
    print(f"Total parameters: {model.n_var:,}")

    # Prepare training data
    print_section("Preparing Training Data")
    x_train, y_train = dataset.prepare_training_data(use_numpy=True)
    print(f"Training samples: {len(x_train):,}")
    print(f"Input shape: {x_train.shape}")
    print(f"Output shape: {y_train.shape}")

    # Configure training
    print_section("Training Configuration")
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=64,
        max_epochs=50,
        min_epochs=10,
        early_stopping_patience=10,
        validation_split=0.1,
        verbose=1,
    )
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Max epochs: {config.max_epochs}")
    print(f"Early stopping patience: {config.early_stopping_patience}")

    # Train model
    print_section("Training Model")
    trainer = Trainer(model, config)

    start_time = time.time()
    history = trainer.train(x_train, y_train)
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
    model_path = os.path.join(OUTPUT_DIR, "mlp_model.kan")
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    print(f"Model file size: {os.path.getsize(model_path):,} bytes")

    # Generate music
    print_section("Generating Music")
    generator = MusicGenerator(
        model, encoder, seq_length,
        sampling=TemperatureSampling(temperature=0.8)
    )

    # Generate multiple variations
    temperatures = [0.5, 0.8, 1.0]
    for temp in temperatures:
        generator.sampling = TemperatureSampling(temperature=temp)

        output_path = os.path.join(OUTPUT_DIR, f"mlp_generated_temp{temp}.mid")
        sequence = generator.generate_midi(
            duration_beats=64,
            tempo=120.0,
            track_name=f"MLP Generated (temp={temp})"
        )
        sequence.save(output_path)
        print(f"Generated: {os.path.basename(output_path)} (temperature={temp})")

    # Summary
    print_header("Demo Complete")
    print(f"Model saved: {model_path}")
    print(f"Generated files in: {OUTPUT_DIR}")
    print("\nYou can open the generated MIDI files in any MIDI player or DAW.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
