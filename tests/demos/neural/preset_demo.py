#!/usr/bin/env python3
"""Demo showing MIDIDataset presets for different music types.

This demo demonstrates how to use the preset factory methods to train
neural networks on different types of MIDI data:

1. Classical (multi-track): Use for_melody() to filter to melodic content
2. Groove (drum patterns): Use for_drums() to train on rhythm patterns

Usage:
    python preset_demo.py                    # Run both demos
    python preset_demo.py --type melody      # Classical melody only
    python preset_demo.py --type drums       # Drum patterns only
    python preset_demo.py --epochs 50        # More training epochs
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

from coremusic.music.neural import (
    MIDIDataset,
    NoteEncoder,
    ModelFactory,
    TrainingConfig,
    Trainer,
    MusicGenerator,
    TemperatureSampling,
)
from coremusic.midi.utilities import MIDISequence


def train_melody_model(data_dir: Path, output_dir: Path, epochs: int = 20):
    """Train a model on classical melody data using for_melody() preset."""
    print("\n" + "=" * 60)
    print("MELODY MODEL (Classical Dataset)")
    print("=" * 60)

    # Create dataset with melody preset
    # This filters out drums and limits pitch range to C3-C6
    encoder = NoteEncoder()
    dataset = MIDIDataset.for_melody(encoder, seq_length=16)

    print(f"\nDataset configuration:")
    print(f"  Preset: for_melody()")
    print(f"  exclude_drums: {dataset.exclude_drums}")
    print(f"  pitch_range: {dataset.pitch_range}")
    print(f"  seq_length: {dataset.seq_length}")

    # Load classical MIDI files
    classical_dir = data_dir / "classical"
    if not classical_dir.exists():
        print(f"  [SKIP] Classical directory not found: {classical_dir}")
        return

    print(f"\nLoading from: {classical_dir}")
    total_tokens = dataset.load_directory(classical_dir, pattern="*.mid")
    print(f"  Loaded {dataset.n_sequences} sequences, {total_tokens} total tokens")

    if dataset.n_sequences == 0:
        print("  [SKIP] No valid sequences found")
        return

    # Augment with transposition
    new_seqs = dataset.augment(transpose_range=(-3, 3))
    print(f"  Augmented: +{new_seqs} sequences (transposition)")
    print(f"  Total: {dataset.n_sequences} sequences, {dataset.total_tokens} tokens")

    # Build model using ModelFactory
    # Use GRU with proper BPTT training for sequence learning
    model = ModelFactory.create(
        model_type="gru",
        encoder=encoder,
        seq_length=dataset.seq_length,
        hidden_size=256,
    )
    print(f"\nModel: GRU (hidden_size=256)")

    # Train using Trainer with RNN-specific training (proper BPTT)
    config = TrainingConfig(
        learning_rate=0.001,
        max_epochs=epochs,
        min_epochs=epochs,
        batch_size=32,
        verbose=1,
    )
    trainer = Trainer(model, config)

    print(f"\nTraining for {epochs} epochs (using BPTT)...")
    history = trainer.train_rnn_sequences(
        sequences=dataset.sequences,
        seq_length=dataset.seq_length,
        vocab_size=encoder.vocab_size,
    )
    print(f"  Final loss: {history['loss'][-1]:.4f}")

    # Generate sample using MusicGenerator
    print("\nGenerating melody sample...")
    generator = MusicGenerator(model, encoder, seq_length=dataset.seq_length)
    generator.set_sampling(TemperatureSampling(temperature=0.8))
    seed = dataset.get_sample_sequence()

    generated = generator.generate(seed=seed, length=64)

    # Save output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "melody_generated.mid"

    events = encoder.decode(generated, tempo=100.0)
    seq = MIDISequence(tempo=100)
    track = seq.add_track("Generated Melody")
    for event in events:
        if event.is_note_on:
            # Find matching note-off
            for off_event in events:
                if (off_event.is_note_off and
                    off_event.data1 == event.data1 and
                    off_event.time > event.time):
                    duration = off_event.time - event.time
                    track.add_note(event.time, event.data1, event.data2, duration)
                    break

    seq.save(str(output_file))
    print(f"  Saved: {output_file}")
    print(f"  Generated {len(generated)} tokens")

    # Show pitch distribution
    pitches = [t for t in generated if 0 <= t <= 127]
    if pitches:
        print(f"  Pitch range: {min(pitches)} - {max(pitches)}")


def train_drums_model(data_dir: Path, output_dir: Path, epochs: int = 20):
    """Train a model on drum pattern data using for_drums() preset."""
    print("\n" + "=" * 60)
    print("DRUMS MODEL (Groove Dataset)")
    print("=" * 60)

    # Create dataset with drums preset
    # No filtering - notes represent different percussion instruments
    encoder = NoteEncoder()
    dataset = MIDIDataset.for_drums(encoder, seq_length=16)

    print(f"\nDataset configuration:")
    print(f"  Preset: for_drums()")
    print(f"  exclude_drums: {dataset.exclude_drums}")
    print(f"  pitch_range: {dataset.pitch_range}")
    print(f"  seq_length: {dataset.seq_length}")

    # Load groove MIDI files
    groove_dir = data_dir / "groove"
    if not groove_dir.exists():
        print(f"  [SKIP] Groove directory not found: {groove_dir}")
        return

    print(f"\nLoading from: {groove_dir}")
    total_tokens = dataset.load_directory(groove_dir, pattern="*.mid", recursive=True)
    print(f"  Loaded {dataset.n_sequences} sequences, {total_tokens} total tokens")

    if dataset.n_sequences == 0:
        print("  [SKIP] No valid sequences found")
        return

    # No transposition for drums - it would change instrument mapping
    print(f"  (No augmentation - transposition changes drum mapping)")

    # Build model
    # Use GRU with proper BPTT training for sequence learning
    model = ModelFactory.create(
        model_type="gru",
        encoder=encoder,
        seq_length=dataset.seq_length,
        hidden_size=256,
    )
    print(f"\nModel: GRU (hidden_size=256)")

    # Train using Trainer with RNN-specific training (proper BPTT)
    config = TrainingConfig(
        learning_rate=0.001,
        max_epochs=epochs,
        min_epochs=epochs,
        batch_size=32,
        verbose=1,
    )
    trainer = Trainer(model, config)

    print(f"\nTraining for {epochs} epochs (using BPTT)...")
    history = trainer.train_rnn_sequences(
        sequences=dataset.sequences,
        seq_length=dataset.seq_length,
        vocab_size=encoder.vocab_size,
    )
    print(f"  Final loss: {history['loss'][-1]:.4f}")

    # Generate sample
    print("\nGenerating drum pattern sample...")
    generator = MusicGenerator(model, encoder, seq_length=dataset.seq_length)
    generator.set_sampling(TemperatureSampling(temperature=0.9))
    seed = dataset.get_sample_sequence()

    generated = generator.generate(seed=seed, length=64)

    # Save output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "drums_generated.mid"

    events = encoder.decode(generated, tempo=120.0, channel=9)  # Channel 9 for drums
    seq = MIDISequence(tempo=120)
    track = seq.add_track("Generated Drums")
    for event in events:
        if event.is_note_on:
            for off_event in events:
                if (off_event.is_note_off and
                    off_event.data1 == event.data1 and
                    off_event.time > event.time):
                    duration = off_event.time - event.time
                    track.add_note(event.time, event.data1, event.data2, duration, channel=9)
                    break

    seq.save(str(output_file))
    print(f"  Saved: {output_file}")
    print(f"  Generated {len(generated)} tokens")

    # Show drum note distribution (GM drum map)
    drum_names = {
        35: "Kick 2", 36: "Kick 1", 38: "Snare 1", 40: "Snare 2",
        42: "HH Closed", 44: "HH Pedal", 46: "HH Open",
        49: "Crash 1", 51: "Ride 1", 53: "Ride Bell",
    }
    notes = [t for t in generated if 0 <= t <= 127]
    if notes:
        print(f"  Note range: {min(notes)} - {max(notes)}")
        # Count most common
        from collections import Counter
        counts = Counter(notes).most_common(5)
        print("  Top drums used:")
        for note, count in counts:
            name = drum_names.get(note, f"Note {note}")
            print(f"    {note}: {name} ({count}x)")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: MIDIDataset presets for melody and drum training"
    )
    parser.add_argument(
        "--type",
        choices=["melody", "drums", "both"],
        default="both",
        help="Which demo to run (default: both)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs (default: 50)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "tests" / "data" / "midi",
        help="Directory containing MIDI datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "build" / "midi_files" / "preset_demo",
        help="Output directory for generated files",
    )
    args = parser.parse_args()

    print("MIDIDataset Preset Demo")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")

    if args.type in ("melody", "both"):
        train_melody_model(args.data_dir, args.output_dir, args.epochs)

    if args.type in ("drums", "both"):
        train_drums_model(args.data_dir, args.output_dir, args.epochs)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
