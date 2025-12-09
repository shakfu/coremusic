#!/usr/bin/env python3
"""Comprehensive Neural Network Music Generation Demo.

This demo showcases all available model architectures and encoder types
for neural network music generation with the coremusic library.

Features demonstrated:
- Model architectures: 
    MLP: Multi-Layer Perceptron
    RNN: Recurrent Neural Network
    LSTM: Long Short-term Memory
    GRU: Gated Recurrent Unit
- Encoder types: NoteEncoder, EventEncoder, PianoRollEncoder
- Sampling strategies: Temperature, Top-K, Nucleus (Top-P)
- Data augmentation and training callbacks
- Model comparison and evaluation

Usage:
    python -m tests.demos.neural.comprehensive_demo

    Or from the project root:
    uv run python tests/demos/neural/comprehensive_demo.py

Output:
    - Multiple trained models in build/midi_files/neural/
    - Generated MIDI files for each model/encoder combination
    - Training comparison summary
"""

import os
import glob
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

from coremusic.music.neural import (
    # Encoders
    NoteEncoder,
    EventEncoder,
    PianoRollEncoder,
    BaseEncoder,
    # Dataset
    MIDIDataset,
    # Models
    ModelFactory,
    RNN_NORM,
    # Training
    TrainingConfig,
    Trainer,
    EarlyStopping,
    ModelCheckpoint,
    # Generation
    MusicGenerator,
    TemperatureSampling,
    TopKSampling,
    NucleusSampling,
    GreedySampling,
    # Evaluation
    MusicMetrics,
    # High-level API
    train_music_model,
    generate_music,
)
from coremusic.kann import NeuralNetwork

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
CLASSICAL_DIR = os.path.join(PROJECT_ROOT, "tests", "data", "midi", "classical")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "build", "midi_files", "neural")


@dataclass
class TrainingResult:
    """Results from training a model."""
    model_type: str
    encoder_type: str
    training_time: float
    final_loss: float
    final_val_loss: Optional[float]
    num_parameters: int
    model_path: str


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


def load_midi_files(max_files: int = 8) -> List[str]:
    """Load classical MIDI files."""
    all_files = sorted(glob.glob(os.path.join(CLASSICAL_DIR, "*.mid")))
    midi_files = [f for f in all_files if is_valid_midi_file(f)]

    if not midi_files:
        raise FileNotFoundError(f"No valid MIDI files found in {CLASSICAL_DIR}")

    # Sort by file size and take a mix of small and large files
    files_with_size = [(f, os.path.getsize(f)) for f in midi_files]
    files_with_size.sort(key=lambda x: x[1])

    # Take files from different size ranges
    selected = files_with_size[:max_files]
    return [f[0] for f in selected]


def create_dataset(
    midi_files: List[str],
    encoder: BaseEncoder,
    seq_length: int,
    augment: bool = True
) -> MIDIDataset:
    """Create and populate a dataset."""
    dataset = MIDIDataset(encoder, seq_length=seq_length)

    for midi_file in midi_files:
        try:
            dataset.load_file(midi_file)
        except Exception as e:
            print(f"    Warning: Failed to load {os.path.basename(midi_file)}: {e}")

    # Augment for encoders that support transposition (NoteEncoder, EventEncoder, etc.)
    if augment and dataset.n_sequences > 0 and hasattr(encoder, 'transpose'):
        dataset.augment(transpose_range=(-5, 7))

    return dataset


def train_model(
    model_type: str,
    encoder: BaseEncoder,
    encoder_name: str,
    dataset: MIDIDataset,
    seq_length: int,
    hidden_size: int,
    max_epochs: int,
    output_dir: str,
) -> Optional[TrainingResult]:
    """Train a single model and return results."""
    print(f"\n  Training {model_type.upper()} with {encoder_name}...")

    # RNN models now use proper BPTT training
    is_rnn = model_type in ("lstm", "gru", "rnn")
    if is_rnn:
        print(f"    Using BPTT training (train_rnn_sequences)")

    if dataset.n_sequences == 0:
        print(f"    Skipping: No sequences in dataset")
        return None

    try:
        # Adjust hyperparameters based on model type
        # Recurrent models (LSTM/GRU) need different settings than MLP
        if model_type in ("lstm", "gru"):
            # Recurrent models benefit from:
            # - Lower learning rate (avoid exploding gradients)
            # - Larger hidden size (more capacity for temporal patterns)
            # - Smaller batches (better gradient estimates)
            # - Layer normalization (RNN_NORM) for stable gradients
            model_hidden_size = hidden_size * 2  # 256 instead of 128
            learning_rate = 0.005  # Higher for BPTT (was 0.0005)
            batch_size = 8  # Smaller batches for BPTT
            patience = 10  # More patience for slower convergence
            rnn_flags = RNN_NORM  # Enable layer normalization
        else:
            # MLP settings
            model_hidden_size = hidden_size
            learning_rate = 0.001
            batch_size = 32
            patience = 8
            rnn_flags = 0

        # Create model
        model = ModelFactory.create(
            model_type=model_type,
            encoder=encoder,
            seq_length=seq_length,
            hidden_size=model_hidden_size,
            rnn_flags=rnn_flags,
        )
        print(f"    Parameters: {model.n_var:,}")
        norm_str = " (with layer norm)" if rnn_flags & RNN_NORM else ""
        print(f"    Hidden size: {model_hidden_size}, LR: {learning_rate}, Batch: {batch_size}{norm_str}")

        # Configure training with model-specific hyperparameters
        config = TrainingConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            min_epochs=5,
            early_stopping_patience=patience,
            validation_split=0.1,
            verbose=1,
        )

        # Train
        trainer = Trainer(model, config)
        start_time = time.time()

        if is_rnn:
            # Use proper BPTT training for RNN models
            sequences = dataset.sequences
            print(f"    Training sequences: {len(sequences)}")
            history = trainer.train_rnn_sequences(
                sequences=sequences,
                seq_length=seq_length,
                vocab_size=encoder.vocab_size,
                grad_clip=5.0,
            )
        else:
            # Use standard FNN training for MLP
            x_train, y_train = dataset.prepare_training_data(use_numpy=True)
            print(f"    Training samples: {len(x_train):,}")
            history = trainer.train(x_train, y_train)

        training_time = time.time() - start_time

        final_loss = history['loss'][-1] if history['loss'] else 0.0
        final_val_loss = history['val_loss'][-1] if history['val_loss'] else None
        # For MLP, epochs_trained is stored in history; for RNN, use len(loss)
        epochs_trained = history.get('epochs_trained', len(history['loss']))

        print(f"    Training time: {training_time:.1f}s")
        print(f"    Epochs trained: {epochs_trained}")
        print(f"    Final training loss: {final_loss:.6f}")

        # Show validation metrics and overfitting detection
        if final_val_loss is not None:
            print(f"    Final validation loss: {final_val_loss:.6f}")

            # Calculate overfitting indicator (gap between train and val loss)
            overfit_gap = final_val_loss - final_loss
            if overfit_gap > 0.5:
                print(f"    WARNING: Possible overfitting (gap: {overfit_gap:.3f})")
            elif overfit_gap > 0.2:
                print(f"    Note: Some overfitting (gap: {overfit_gap:.3f})")
            else:
                print(f"    Good generalization (gap: {overfit_gap:.3f})")

            # Show best validation loss epoch
            if history['val_loss']:
                best_val_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
                best_val_loss = min(history['val_loss'])
                print(f"    Best val loss: {best_val_loss:.6f} at epoch {best_val_epoch}")

        # Save model
        model_path = os.path.join(output_dir, f"{model_type}_{encoder_name}.kan")
        model.save(model_path)
        print(f"    Saved: {os.path.basename(model_path)}")

        return TrainingResult(
            model_type=model_type,
            encoder_type=encoder_name,
            training_time=training_time,
            final_loss=final_loss,
            final_val_loss=final_val_loss,
            num_parameters=model.n_var,
            model_path=model_path,
        )

    except Exception as e:
        print(f"    Error: {e}")
        return None


def generate_from_model(
    model_path: str,
    encoder: BaseEncoder,
    seq_length: int,
    output_dir: str,
    prefix: str,
) -> List[str]:
    """Generate music from a trained model with various sampling strategies."""
    generated_files = []

    try:
        model = NeuralNetwork.load(model_path)

        # Different sampling strategies
        samplings = [
            ("greedy", GreedySampling()),
            ("temp0.5", TemperatureSampling(0.5)),
            ("temp0.8", TemperatureSampling(0.8)),
            ("temp1.0", TemperatureSampling(1.0)),
            ("topk10", TopKSampling(10)),
            ("nucleus0.9", NucleusSampling(0.9)),
        ]

        generator = MusicGenerator(model, encoder, seq_length)

        for name, sampling in samplings:
            generator.sampling = sampling
            output_path = os.path.join(output_dir, f"{prefix}_{name}.mid")

            sequence = generator.generate_midi(
                duration_beats=32,
                tempo=120.0,
                track_name=f"{prefix} ({name})"
            )
            sequence.save(output_path)
            generated_files.append(output_path)

    except Exception as e:
        print(f"    Generation error: {e}")

    return generated_files


def print_comparison_table(results: List[TrainingResult]):
    """Print a comparison table of training results."""
    print_header("Training Results Comparison")

    # Table header
    # print(f"\n{'Model':<10} {'Encoder':<12} {'Params':>10} {'Time':>8} {'Train':>10} {'Val':>10} {'Gap':>8} {'Status':>12}")
    print(f"\n{'Model':<8} {'Encoder':<8} {'Params':>10} {'Time':>8} {'Train':>10} {'Val':>10} {'Gap':>8} {'Status':>12}")
    print("-" * 84)

    # Sort by validation loss if available, otherwise by training loss
    def sort_key(r):
        if r.final_val_loss is not None:
            return r.final_val_loss
        return r.final_loss

    sorted_results = sorted(results, key=sort_key)

    for r in sorted_results:
        val_loss_str = f"{r.final_val_loss:.4f}" if r.final_val_loss else "N/A"

        # Calculate overfitting gap
        if r.final_val_loss is not None:
            gap = r.final_val_loss - r.final_loss
            gap_str = f"{gap:.3f}"
            if gap > 0.5:
                status = "OVERFIT"
            elif gap > 0.2:
                status = "slight overfit"
            else:
                status = "good"
        else:
            gap_str = "N/A"
            status = "no val"

        # print(f"{r.model_type:<10} {r.encoder_type:<12} {r.num_parameters:>10,} "
        #       f"{r.training_time:>7.1f}s {r.final_loss:>10.4f} {val_loss_str:>10} {gap_str:>8} {status:>12}")
        print(f"{r.model_type:<8} {r.encoder_type:<8} {r.num_parameters:>10,} "
              f"{r.training_time:>7.1f}s {r.final_loss:>10.4f} {val_loss_str:>10} {gap_str:>8} {status:>12}")

    # Best model (by validation loss)
    best = sorted_results[0]
    metric = "val loss" if best.final_val_loss else "train loss"
    loss_val = best.final_val_loss if best.final_val_loss else best.final_loss
    print(f"\nBest model: {best.model_type.upper()} with {best.encoder_type} ({metric}: {loss_val:.4f})")

    # Recommendations
    print("\nRecommendations:")
    print("  - 'good': Model generalizes well, safe to use")
    print("  - 'slight overfit': May benefit from more regularization or less training")
    print("  - 'OVERFIT': Model memorizing training data, reduce epochs or add dropout")

    # Quality summary table
    print("\nQuality Summary:")
    print(f"  {'Model':<6} {'Encoder':<7} {'Gap':>7} {'Quality':<20}")
    print("  " + "-" * 42)
    for r in sorted_results:
        if r.final_val_loss is not None:
            gap = r.final_val_loss - r.final_loss
            if gap > 5.0:
                quality = "Severe overfit (skip)"
            elif gap > 0.5:
                quality = "Moderate overfit"
            elif gap > 0.2:
                quality = "Slight overfit"
            else:
                quality = "Excellent"
        else:
            gap = 0.0
            quality = "No validation"
        print(f"  {r.model_type:<6} {r.encoder_type:<7} {gap:>7.2f} {quality:<20}")


def main():
    """Run the comprehensive neural network demo."""
    print_header("Comprehensive Neural Network Music Generation Demo")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load MIDI files
    print_section("Loading MIDI Files")
    try:
        midi_files = load_midi_files(max_files=8)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    print(f"Loaded {len(midi_files)} MIDI files:")
    for f in midi_files[:5]:
        print(f"  - {os.path.basename(f)}")
    if len(midi_files) > 5:
        print(f"  ... and {len(midi_files) - 5} more")

    # Configuration
    # MLP works well with shorter sequences
    # LSTM/GRU need longer sequences and different hyperparameters
    seq_length = 32
    hidden_size = 128  # Base hidden size (adjusted per model type)
    max_epochs = 30  # Balanced for demo runtime vs learning

    print(f"\nConfiguration:")
    print(f"  Sequence length: {seq_length}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Max epochs: {max_epochs}")

    # Model types to test
    model_types = ["mlp", "lstm", "gru"]

    # Encoder configurations
    encoders = [
        ("note", NoteEncoder()),
        ("event", EventEncoder()),
    ]

    # Store results
    results: List[TrainingResult] = []

    # Train models
    print_header("Training Models")

    for encoder_name, encoder in encoders:
        print_section(f"Encoder: {encoder_name}")
        print(f"  Vocabulary size: {encoder.vocab_size}")

        # Create dataset for this encoder
        dataset = create_dataset(
            midi_files, encoder, seq_length, augment=True
        )
        print(f"  Dataset sequences: {dataset.n_sequences:,}")

        if dataset.n_sequences == 0:
            print(f"  Skipping encoder: no sequences")
            continue

        for model_type in model_types:
            # Skip MLP with event encoder - impractically slow due to large input dim
            # (seq_length * vocab_size = 32 * 388 = 12,416 input features)
            if model_type == "mlp" and encoder_name == "event":
                print(f"\n  Skipping MLP with event encoder (impractically slow)")
                continue

            result = train_model(
                model_type=model_type,
                encoder=encoder,
                encoder_name=encoder_name,
                dataset=dataset,
                seq_length=seq_length,
                hidden_size=hidden_size,
                max_epochs=max_epochs,
                output_dir=OUTPUT_DIR,
            )
            if result:
                results.append(result)

    # Print comparison
    if results:
        print_comparison_table(results)

    # Generate music from models that generalize well
    print_header("Generating Music")

    # Skip generation for severely overfitting models (gap > 5.0)
    SEVERE_OVERFIT_THRESHOLD = 5.0

    total_generated = 0
    skipped_models = 0
    for result in results:
        # Check for severe overfitting
        if result.final_val_loss is not None:
            gap = result.final_val_loss - result.final_loss
            if gap > SEVERE_OVERFIT_THRESHOLD:
                print(f"\n  Skipping {result.model_type}_{result.encoder_type} (severe overfit, gap: {gap:.1f})")
                skipped_models += 1
                continue

        print(f"\n  Generating from {result.model_type}_{result.encoder_type}...")

        # Get encoder
        if result.encoder_type == "note":
            encoder = NoteEncoder()
        elif result.encoder_type == "event":
            encoder = EventEncoder()
        else:
            continue

        files = generate_from_model(
            model_path=result.model_path,
            encoder=encoder,
            seq_length=seq_length,
            output_dir=OUTPUT_DIR,
            prefix=f"{result.model_type}_{result.encoder_type}",
        )
        total_generated += len(files)
        print(f"    Generated {len(files)} files")

    if skipped_models > 0:
        print(f"\n  ({skipped_models} severely overfitting model(s) skipped)")

    # Summary
    print_header("Demo Complete")
    print(f"\nModels trained: {len(results)}")
    print(f"Files generated: {total_generated}")
    print(f"Output directory: {OUTPUT_DIR}")

    print("\nGenerated files include variations with different sampling strategies:")
    print("  - greedy: Always picks highest probability note")
    print("  - temp0.5: Conservative temperature sampling")
    print("  - temp0.8: Moderate temperature sampling")
    print("  - temp1.0: Standard temperature sampling")
    print("  - topk10: Top-K sampling with K=10")
    print("  - nucleus0.9: Nucleus (top-p) sampling with p=0.9")

    print("\nModel comparisons:")
    print("  - MLP: Fast training with feedforward API, good for fixed-window patterns")
    print("  - LSTM: Trained with BPTT, can learn long-range temporal dependencies")
    print("  - GRU: Trained with BPTT, simplified LSTM alternative with fewer parameters")
    print("\nNote: RNN models (LSTM/GRU) use train_rnn_sequences() with proper")
    print("backpropagation through time (BPTT) for effective sequence learning.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
