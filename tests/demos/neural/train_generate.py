#!/usr/bin/env python3
"""Neural Network MIDI Training and Generation Demo.

Train neural network models on MIDI files and generate new music.

Examples:
    # Train on a single MIDI file
    python tests/demos/neural/train_generate.py tests/data/midi/demo.mid

    # Train on a folder of MIDI files
    python tests/demos/neural/train_generate.py tests/data/midi/classical/

    # Train on groove dataset
    python tests/demos/neural/train_generate.py tests/data/midi/groove/drummer1/

    # Specify model type and encoder
    python tests/demos/neural/train_generate.py tests/data/midi/classical/ --model lstm --encoder note

    # Train all model/encoder combinations
    python tests/demos/neural/train_generate.py tests/data/midi/classical/ --all

    # Custom output directory
    python tests/demos/neural/train_generate.py tests/data/midi/demo.mid -o my_output/

    # More epochs for better training
    python tests/demos/neural/train_generate.py tests/data/midi/classical/ --epochs 50
"""

import argparse
import glob
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from coremusic.music.neural import (
    # Encoders
    NoteEncoder,
    EventEncoder,
    PianoRollEncoder,
    RelativePitchEncoder,
    BaseEncoder,
    # Dataset
    MIDIDataset,
    # Models
    ModelFactory,
    RNN_NORM,
    # Training
    TrainingConfig,
    Trainer,
    # Generation
    MusicGenerator,
    TemperatureSampling,
    TopKSampling,
    NucleusSampling,
    GreedySampling,
)
from coremusic.kann import NeuralNetwork
from coremusic.midi.transform import Humanize, Reverse, Arpeggiate
from coremusic.midi.utilities import MIDISequence

# Default output directory
DEFAULT_OUTPUT_DIR = "build/midi_files"


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
    epochs_trained: int


def is_valid_midi_file(filepath: str) -> bool:
    """Check if a file is a valid Standard MIDI File."""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4)
            return header == b'MThd'
    except Exception:
        return False


def find_midi_files(path: str, max_files: int = 0) -> List[str]:
    """Find MIDI files from a file or directory path.

    Args:
        path: Path to a MIDI file or directory containing MIDI files
        max_files: Maximum number of files to return (0 = unlimited)

    Returns:
        List of valid MIDI file paths
    """
    path = os.path.abspath(path)

    if os.path.isfile(path):
        if is_valid_midi_file(path):
            return [path]
        else:
            raise ValueError(f"Not a valid MIDI file: {path}")

    if os.path.isdir(path):
        # Search for .mid and .midi files
        patterns = [
            os.path.join(path, "*.mid"),
            os.path.join(path, "*.midi"),
            os.path.join(path, "**", "*.mid"),
            os.path.join(path, "**", "*.midi"),
        ]

        all_files = set()
        for pattern in patterns:
            all_files.update(glob.glob(pattern, recursive=True))

        midi_files = sorted([f for f in all_files if is_valid_midi_file(f)])

        if not midi_files:
            raise ValueError(f"No valid MIDI files found in: {path}")

        if max_files > 0 and len(midi_files) > max_files:
            midi_files = midi_files[:max_files]

        return midi_files

    raise ValueError(f"Path does not exist: {path}")


def create_encoder(encoder_type: str) -> BaseEncoder:
    """Create an encoder by name."""
    encoders = {
        "note": NoteEncoder,
        "event": EventEncoder,
        "pianoroll": PianoRollEncoder,
        "relative": RelativePitchEncoder,
    }
    if encoder_type not in encoders:
        raise ValueError(f"Unknown encoder: {encoder_type}. Choose from: {list(encoders.keys())}")
    return encoders[encoder_type]()


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
            print(f"  Warning: Failed to load {os.path.basename(midi_file)}: {e}")

    # Augment for encoders that support transposition
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
    source_name: str = "",
    verbose: bool = True,
) -> Optional[TrainingResult]:
    """Train a single model and return results."""
    if verbose:
        print(f"\nTraining {model_type.upper()} with {encoder_name} encoder...")

    is_rnn = model_type in ("lstm", "gru", "rnn")

    if dataset.n_sequences == 0:
        print(f"  Skipping: No sequences in dataset")
        return None

    try:
        # Model-specific hyperparameters
        if model_type in ("lstm", "gru"):
            model_hidden_size = hidden_size * 2
            learning_rate = 0.005
            batch_size = 8
            patience = 10
            rnn_flags = RNN_NORM
        else:
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

        if verbose:
            norm_str = " (layer norm)" if rnn_flags & RNN_NORM else ""
            print(f"  Parameters: {model.n_var:,}{norm_str}")
            print(f"  Hidden: {model_hidden_size}, LR: {learning_rate}, Batch: {batch_size}")

        # Configure training
        config = TrainingConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            min_epochs=5,
            early_stopping_patience=patience,
            validation_split=0.1,
            verbose=1 if verbose else 0,
        )

        # Train
        trainer = Trainer(model, config)
        start_time = time.time()

        if is_rnn:
            sequences = dataset.sequences
            if verbose:
                print(f"  Sequences: {len(sequences)}")
            history = trainer.train_rnn_sequences(
                sequences=sequences,
                seq_length=seq_length,
                vocab_size=encoder.vocab_size,
                grad_clip=5.0,
            )
        else:
            x_train, y_train = dataset.prepare_training_data(use_numpy=True)
            if verbose:
                print(f"  Samples: {len(x_train):,}")
            history = trainer.train(x_train, y_train)

        training_time = time.time() - start_time

        final_loss = history['loss'][-1] if history['loss'] else 0.0
        final_val_loss = history['val_loss'][-1] if history['val_loss'] else None
        epochs_trained = history.get('epochs_trained', len(history['loss']))

        if verbose:
            print(f"  Time: {training_time:.1f}s, Epochs: {epochs_trained}")
            print(f"  Train loss: {final_loss:.4f}", end="")
            if final_val_loss is not None:
                gap = final_val_loss - final_loss
                print(f", Val loss: {final_val_loss:.4f}, Gap: {gap:.3f}")
            else:
                print()

        # Save model
        if source_name:
            model_path = os.path.join(output_dir, f"{source_name}_{model_type}_{encoder_name}.kan")
        else:
            model_path = os.path.join(output_dir, f"{model_type}_{encoder_name}.kan")
        model.save(model_path)
        if verbose:
            print(f"  Saved: {model_path}")

        return TrainingResult(
            model_type=model_type,
            encoder_type=encoder_name,
            training_time=training_time,
            final_loss=final_loss,
            final_val_loss=final_val_loss,
            num_parameters=model.n_var,
            model_path=model_path,
            epochs_trained=epochs_trained,
        )

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_transforms(transform_names: Optional[List[str]]) -> List:
    """Create transform instances from names.

    Available transforms:
        - humanize: Add timing/velocity variation (timing=0.02s, velocity=15)
        - reverse: Reverse the sequence
        - arpeggiate: Arpeggiate chords
    """
    if not transform_names:
        return []

    available = {
        "humanize": lambda: Humanize(timing=0.02, velocity=15),
        "reverse": lambda: Reverse(),
        "arpeggiate": lambda: Arpeggiate(rate=0.125),
    }

    transforms = []
    for name in transform_names:
        name_lower = name.lower()
        if name_lower not in available:
            raise ValueError(
                f"Unknown transform: {name}. Available: {list(available.keys())}"
            )
        transforms.append(available[name_lower]())

    return transforms


def apply_transforms(sequence: MIDISequence, transforms: List) -> MIDISequence:
    """Apply a list of transforms to a MIDI sequence."""
    result = sequence
    for transform in transforms:
        result = transform.transform(result)
    return result


def generate_from_model(
    model_path: str,
    encoder: BaseEncoder,
    seq_length: int,
    output_dir: str,
    prefix: str,
    transforms: Optional[List] = None,
    num_files: int = 6,
    verbose: bool = True,
) -> List[str]:
    """Generate music from a trained model.

    Args:
        model_path: Path to saved model
        encoder: Encoder instance
        seq_length: Sequence length model was trained on
        output_dir: Directory for output files
        prefix: Filename prefix
        transforms: List of MIDITransformer instances to apply
        num_files: Number of files to generate
        verbose: Print progress
    """
    generated_files = []

    try:
        model = NeuralNetwork.load(model_path)

        # Sampling strategies
        samplings = [
            ("greedy", GreedySampling()),
            ("temp0.5", TemperatureSampling(0.5)),
            ("temp0.8", TemperatureSampling(0.8)),
            ("temp1.0", TemperatureSampling(1.0)),
            ("topk10", TopKSampling(10)),
            ("nucleus0.9", NucleusSampling(0.9)),
        ][:num_files]

        generator = MusicGenerator(model, encoder, seq_length)

        for name, sampling in samplings:
            generator.sampling = sampling
            output_path = os.path.join(output_dir, f"{prefix}_{name}.mid")

            sequence = generator.generate_midi(
                duration_beats=32,
                tempo=120.0,
                track_name=f"{prefix} ({name})"
            )

            # Apply transforms if any
            if transforms:
                sequence = apply_transforms(sequence, transforms)

            sequence.save(output_path)
            generated_files.append(output_path)

            if verbose:
                print(f"    {os.path.basename(output_path)}")

    except Exception as e:
        print(f"  Generation error: {e}")

    return generated_files


def print_results_table(results: List[TrainingResult]):
    """Print a summary table of training results."""
    print("\n" + "=" * 70)
    print(" Training Results")
    print("=" * 70)

    print(f"\n{'Model':<8} {'Encoder':<8} {'Params':>10} {'Time':>8} {'Train':>8} {'Val':>8} {'Gap':>7}")
    print("-" * 65)

    for r in sorted(results, key=lambda x: x.final_val_loss or x.final_loss):
        val_str = f"{r.final_val_loss:.4f}" if r.final_val_loss else "N/A"
        if r.final_val_loss:
            gap = r.final_val_loss - r.final_loss
            gap_str = f"{gap:.3f}"
        else:
            gap_str = "N/A"

        print(f"{r.model_type:<8} {r.encoder_type:<8} {r.num_parameters:>10,} "
              f"{r.training_time:>7.1f}s {r.final_loss:>8.4f} {val_str:>8} {gap_str:>7}")


def get_model_encoder_combinations(
    model: Optional[str],
    encoder: Optional[str],
    train_all: bool
) -> List[Tuple[str, str]]:
    """Get list of (model_type, encoder_type) combinations to train."""
    all_models = ["mlp", "lstm", "gru"]
    all_encoders = ["note", "event", "pianoroll", "relative"]

    # Encoders with large vocab sizes that are impractical with MLP
    large_vocab_encoders = {"event", "pianoroll"}

    if train_all:
        # All combinations except MLP with large-vocab encoders (impractically slow)
        combinations = []
        for m in all_models:
            for e in all_encoders:
                if m == "mlp" and e in large_vocab_encoders:
                    continue  # Skip slow combinations
                combinations.append((m, e))
        return combinations

    # Single combination
    model_type = model or "lstm"
    encoder_type = encoder or "note"
    return [(model_type, encoder_type)]


def main():
    parser = argparse.ArgumentParser(
        description="Train neural networks on MIDI files and generate music",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "input",
        help="MIDI file or directory containing MIDI files"
    )

    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )

    parser.add_argument(
        "-m", "--model",
        choices=["mlp", "lstm", "gru", "rnn"],
        help="Model type (default: lstm)"
    )

    parser.add_argument(
        "-e", "--encoder",
        choices=["note", "event", "pianoroll", "relative"],
        help="Encoder type (default: note)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        dest="train_all",
        help="Train all model/encoder combinations"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum training epochs (default: 30)"
    )

    parser.add_argument(
        "--seq-length",
        type=int,
        default=32,
        help="Sequence length (default: 32)"
    )

    parser.add_argument(
        "--hidden",
        type=int,
        default=128,
        help="Hidden layer size (default: 128)"
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Maximum MIDI files to load (0 = unlimited)"
    )

    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation"
    )

    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip generation after training"
    )

    parser.add_argument(
        "-t", "--transform",
        action="append",
        dest="transforms",
        metavar="NAME",
        help="Apply transform(s) to generated MIDI. Can be specified multiple times. "
             "Available: humanize, reverse, arpeggiate. Example: -t humanize -t reverse"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Find MIDI files
    if verbose:
        print(f"Loading MIDI files from: {args.input}")

    try:
        midi_files = find_midi_files(args.input, args.max_files)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if verbose:
        print(f"Found {len(midi_files)} MIDI file(s)")
        for f in midi_files[:5]:
            print(f"  - {os.path.basename(f)}")
        if len(midi_files) > 5:
            print(f"  ... and {len(midi_files) - 5} more")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Determine source name for output file prefix
    # If single file: use filename without extension
    # If directory: use directory name
    input_path = os.path.abspath(args.input)
    if os.path.isfile(input_path):
        source_name = Path(input_path).stem  # e.g., "demo" from "demo.mid"
    else:
        source_name = Path(input_path).name  # e.g., "classical" from "classical/"

    # Get model/encoder combinations
    combinations = get_model_encoder_combinations(
        args.model, args.encoder, args.train_all
    )

    if verbose:
        print(f"\nTraining {len(combinations)} model(s):")
        for m, e in combinations:
            print(f"  - {m.upper()} with {e} encoder")
        print(f"\nConfiguration:")
        print(f"  Epochs: {args.epochs}")
        print(f"  Sequence length: {args.seq_length}")
        print(f"  Hidden size: {args.hidden}")
        print(f"  Augmentation: {'disabled' if args.no_augment else 'enabled'}")

    # Train models
    results: List[TrainingResult] = []
    datasets = {}  # Cache datasets by encoder

    for model_type, encoder_type in combinations:
        # Get or create dataset for this encoder
        if encoder_type not in datasets:
            encoder = create_encoder(encoder_type)
            dataset = create_dataset(
                midi_files,
                encoder,
                args.seq_length,
                augment=not args.no_augment
            )
            datasets[encoder_type] = (encoder, dataset)
            if verbose:
                print(f"\nDataset ({encoder_type}): {dataset.n_sequences} sequences")

        encoder, dataset = datasets[encoder_type]

        result = train_model(
            model_type=model_type,
            encoder=encoder,
            encoder_name=encoder_type,
            dataset=dataset,
            seq_length=args.seq_length,
            hidden_size=args.hidden,
            max_epochs=args.epochs,
            output_dir=args.output,
            source_name=source_name,
            verbose=verbose,
        )

        if result:
            results.append(result)

    if not results:
        print("No models were trained successfully")
        return 1

    # Print results
    if verbose and len(results) > 1:
        print_results_table(results)

    # Generate music
    if not args.no_generate:
        # Create transforms
        transforms = create_transforms(args.transforms)

        if verbose:
            print("\n" + "=" * 70)
            print(" Generating Music")
            if transforms:
                print(f" Transforms: {', '.join(args.transforms)}")
            print("=" * 70)

        total_generated = 0

        for result in results:
            # Skip severely overfitting models
            if result.final_val_loss is not None:
                gap = result.final_val_loss - result.final_loss
                if gap > 5.0:
                    if verbose:
                        print(f"\n  Skipping {result.model_type}_{result.encoder_type} (severe overfit)")
                    continue

            if verbose:
                print(f"\n  Generating from {result.model_type}_{result.encoder_type}:")

            encoder = create_encoder(result.encoder_type)
            files = generate_from_model(
                model_path=result.model_path,
                encoder=encoder,
                seq_length=args.seq_length,
                output_dir=args.output,
                prefix=f"{source_name}_{result.model_type}_{result.encoder_type}",
                transforms=transforms,
                verbose=verbose,
            )
            total_generated += len(files)

        if verbose:
            print(f"\nGenerated {total_generated} MIDI file(s)")

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print(f" Complete: {len(results)} model(s) trained")
        print(f" Output: {os.path.abspath(args.output)}")
        print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
