#!/usr/bin/env python3
"""Demo showing ScaleEncoder for guaranteed in-key melody generation.

This demo demonstrates how to use ScaleEncoder to:
1. Encode MIDI data as scale degrees (smaller vocabulary)
2. Train a neural network on scale-relative patterns
3. Generate melodies that are guaranteed to be in the target scale

The key advantage of ScaleEncoder over NoteEncoder is that all generated
notes are guaranteed to be in the specified scale, making the output
more musically coherent.

Usage:
    python scale_encoder_demo.py                    # Run with defaults
    python scale_encoder_demo.py --scale C_MAJOR    # Use C major scale
    python scale_encoder_demo.py --scale A_MINOR    # Use A minor scale
    python scale_encoder_demo.py --scale G_MAJOR    # Use G major scale
    python scale_encoder_demo.py --epochs 50        # More training epochs
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

from coremusic.music.neural import (
    MIDIDataset,
    ScaleEncoder,
    ModelFactory,
    TrainingConfig,
    Trainer,
    MusicGenerator,
    TemperatureSampling,
)
from coremusic.music.theory import Note, Scale, ScaleType
from coremusic.midi.utilities import MIDISequence


# Predefined scales for the demo
SCALE_PRESETS = {
    "C_MAJOR": Scale(Note("C", 4), ScaleType.MAJOR),
    "G_MAJOR": Scale(Note("G", 4), ScaleType.MAJOR),
    "D_MAJOR": Scale(Note("D", 4), ScaleType.MAJOR),
    "A_MINOR": Scale(Note("A", 4), ScaleType.NATURAL_MINOR),
    "E_MINOR": Scale(Note("E", 4), ScaleType.NATURAL_MINOR),
    "C_PENTATONIC": Scale(Note("C", 4), ScaleType.MAJOR_PENTATONIC),
    "A_BLUES": Scale(Note("A", 3), ScaleType.BLUES),
}


def train_scale_model(
    data_dir: Path,
    output_dir: Path,
    scale: Scale,
    epochs: int = 30,
    seq_length: int = 16,
):
    """Train a model using ScaleEncoder for in-key melody generation."""
    print("\n" + "=" * 60)
    print(f"SCALE ENCODER MODEL ({scale})")
    print("=" * 60)

    # Create ScaleEncoder
    # octave_range=(3, 6) covers C3 to B5, typical melody range
    encoder = ScaleEncoder(
        scale=scale,
        octave_range=(3, 6),
        snap_to_scale=True,  # Snap out-of-scale notes to nearest scale tone
    )

    print(f"\nEncoder configuration:")
    print(f"  Scale: {scale}")
    print(f"  Scale type: {scale.scale_type.name}")
    print(f"  Octave range: {encoder.octave_range}")
    print(f"  Degrees per octave: {encoder.degrees_per_octave}")
    print(f"  Vocab size: {encoder.vocab_size}")
    print(f"    (vs NoteEncoder: 130 tokens)")

    # Show scale notes
    print(f"\nScale notes in vocabulary:")
    notes = encoder.get_scale_notes()
    for i, (token, midi, name) in enumerate(notes[:14]):  # Show first two octaves
        print(f"    Token {token:2d} -> MIDI {midi:3d} ({name})")
    if len(notes) > 14:
        print(f"    ... and {len(notes) - 14} more notes")

    # Create dataset - use melody preset filtering
    dataset = MIDIDataset(
        encoder=encoder,
        seq_length=seq_length,
        exclude_drums=True,
        pitch_range=(36, 84),  # C2 to C6
    )

    print(f"\nDataset configuration:")
    print(f"  seq_length: {dataset.seq_length}")
    print(f"  exclude_drums: {dataset.exclude_drums}")
    print(f"  pitch_range: {dataset.pitch_range}")

    # Load classical MIDI files
    classical_dir = data_dir / "classical"
    if not classical_dir.exists():
        print(f"  [SKIP] Classical directory not found: {classical_dir}")
        return None

    print(f"\nLoading from: {classical_dir}")
    total_tokens = dataset.load_directory(classical_dir, pattern="*.mid")
    print(f"  Loaded {dataset.n_sequences} sequences, {total_tokens} total tokens")

    if dataset.n_sequences == 0:
        print("  [SKIP] No valid sequences found")
        return None

    # Augment using scale-degree transposition (stays in scale!)
    # This is different from chromatic transposition
    print(f"\nNote: Transposition with ScaleEncoder transposes by scale degrees,")
    print(f"      keeping all notes in the original scale.")

    # Build model - smaller vocab means we can use smaller hidden size
    model = ModelFactory.create(
        model_type="gru",
        encoder=encoder,
        seq_length=dataset.seq_length,
        hidden_size=128,  # Smaller than NoteEncoder needs
    )
    print(f"\nModel: GRU (hidden_size=128)")
    print(f"  Input/Output size: {encoder.vocab_size}")

    # Train using BPTT
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

    # Generate samples at different temperatures
    print("\nGenerating melodies...")
    generator = MusicGenerator(model, encoder, seq_length=dataset.seq_length)

    output_dir.mkdir(parents=True, exist_ok=True)
    seed = dataset.get_sample_sequence()

    temperatures = [0.5, 0.8, 1.0]
    for temp in temperatures:
        generator.set_sampling(TemperatureSampling(temperature=temp))
        generated = generator.generate(seed=seed, length=64)

        # Save MIDI file
        scale_name = f"{scale.root.name.lower()}_{scale.scale_type.name.lower()}"
        output_file = output_dir / f"scale_{scale_name}_temp{temp}.mid"

        events = encoder.decode(generated, tempo=100.0)
        seq = MIDISequence(tempo=100)
        track = seq.add_track(f"Generated {scale}")

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

        # Analyze generated notes
        pitches = [encoder.token_to_midi(t) for t in generated
                   if t < encoder.vocab_size - 2]  # Exclude REST/OUT_OF_SCALE

        if pitches:
            # Verify all notes are in scale
            pitch_classes = set(p % 12 for p in pitches)
            scale_pcs = set((scale.root.pitch_class + interval) % 12
                           for interval in scale.intervals)
            all_in_scale = pitch_classes.issubset(scale_pcs)

            print(f"\n  Temperature {temp}:")
            print(f"    Saved: {output_file.name}")
            print(f"    Generated {len(generated)} tokens, {len(pitches)} notes")
            print(f"    Pitch range: {min(pitches)} - {max(pitches)}")
            print(f"    All notes in scale: {all_in_scale}")

    return history


def compare_encoders(data_dir: Path, output_dir: Path, scale: Scale, epochs: int = 20):
    """Compare ScaleEncoder vs NoteEncoder on the same data."""
    from coremusic.music.neural import NoteEncoder

    print("\n" + "=" * 60)
    print("ENCODER COMPARISON")
    print("=" * 60)

    classical_dir = data_dir / "classical"
    if not classical_dir.exists():
        print(f"  [SKIP] Classical directory not found: {classical_dir}")
        return

    # NoteEncoder baseline
    print("\n--- NoteEncoder (baseline) ---")
    note_encoder = NoteEncoder()
    note_dataset = MIDIDataset(
        encoder=note_encoder,
        seq_length=16,
        exclude_drums=True,
        pitch_range=(36, 84),
    )
    note_dataset.load_directory(classical_dir, pattern="*.mid")

    if note_dataset.n_sequences == 0:
        print("  [SKIP] No sequences loaded")
        return

    print(f"  Vocab size: {note_encoder.vocab_size}")
    print(f"  Sequences: {note_dataset.n_sequences}")

    note_model = ModelFactory.create(
        model_type="gru",
        encoder=note_encoder,
        seq_length=16,
        hidden_size=128,
    )

    note_trainer = Trainer(note_model, TrainingConfig(
        learning_rate=0.001,
        max_epochs=epochs,
        min_epochs=epochs,
        batch_size=32,
        verbose=0,
    ))

    print(f"  Training for {epochs} epochs...")
    note_history = note_trainer.train_rnn_sequences(
        sequences=note_dataset.sequences,
        seq_length=16,
        vocab_size=note_encoder.vocab_size,
    )
    print(f"  Final loss: {note_history['loss'][-1]:.4f}")

    # ScaleEncoder
    print(f"\n--- ScaleEncoder ({scale}) ---")
    scale_encoder = ScaleEncoder(scale, octave_range=(3, 6), snap_to_scale=True)
    scale_dataset = MIDIDataset(
        encoder=scale_encoder,
        seq_length=16,
        exclude_drums=True,
        pitch_range=(36, 84),
    )
    scale_dataset.load_directory(classical_dir, pattern="*.mid")

    print(f"  Vocab size: {scale_encoder.vocab_size}")
    print(f"  Sequences: {scale_dataset.n_sequences}")

    scale_model = ModelFactory.create(
        model_type="gru",
        encoder=scale_encoder,
        seq_length=16,
        hidden_size=128,
    )

    scale_trainer = Trainer(scale_model, TrainingConfig(
        learning_rate=0.001,
        max_epochs=epochs,
        min_epochs=epochs,
        batch_size=32,
        verbose=0,
    ))

    print(f"  Training for {epochs} epochs...")
    scale_history = scale_trainer.train_rnn_sequences(
        sequences=scale_dataset.sequences,
        seq_length=16,
        vocab_size=scale_encoder.vocab_size,
    )
    print(f"  Final loss: {scale_history['loss'][-1]:.4f}")

    # Generate and compare
    print("\n--- Generation Comparison ---")

    note_gen = MusicGenerator(note_model, note_encoder, seq_length=16)
    note_gen.set_sampling(TemperatureSampling(temperature=0.8))
    note_seed = note_dataset.get_sample_sequence()
    note_output = note_gen.generate(seed=note_seed, length=64)

    scale_gen = MusicGenerator(scale_model, scale_encoder, seq_length=16)
    scale_gen.set_sampling(TemperatureSampling(temperature=0.8))
    scale_seed = scale_dataset.get_sample_sequence()
    scale_output = scale_gen.generate(seed=scale_seed, length=64)

    # Analyze in-scale percentage
    scale_pcs = set((scale.root.pitch_class + interval) % 12
                   for interval in scale.intervals)

    note_pitches = [t for t in note_output if 0 <= t <= 127]
    note_in_scale = sum(1 for p in note_pitches if (p % 12) in scale_pcs)
    note_pct = (note_in_scale / len(note_pitches) * 100) if note_pitches else 0

    scale_pitches = [scale_encoder.token_to_midi(t) for t in scale_output
                    if t < scale_encoder.vocab_size - 2]
    scale_in_scale = sum(1 for p in scale_pitches if (p % 12) in scale_pcs)
    scale_pct = (scale_in_scale / len(scale_pitches) * 100) if scale_pitches else 0

    print(f"\n  NoteEncoder:")
    print(f"    Notes in {scale}: {note_in_scale}/{len(note_pitches)} ({note_pct:.1f}%)")

    print(f"\n  ScaleEncoder:")
    print(f"    Notes in {scale}: {scale_in_scale}/{len(scale_pitches)} ({scale_pct:.1f}%)")
    print(f"    (Should be 100% by design)")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: ScaleEncoder for in-key melody generation"
    )
    parser.add_argument(
        "--scale",
        choices=list(SCALE_PRESETS.keys()),
        default="C_MAJOR",
        help="Scale to use (default: C_MAJOR)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs (default: 30)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run encoder comparison",
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
        default=PROJECT_ROOT / "build" / "midi_files" / "scale_encoder",
        help="Output directory for generated files",
    )
    args = parser.parse_args()

    scale = SCALE_PRESETS[args.scale]

    print("ScaleEncoder Demo")
    print(f"Scale: {scale}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")

    # Train and generate with ScaleEncoder
    train_scale_model(args.data_dir, args.output_dir, scale, args.epochs)

    # Optionally compare with NoteEncoder
    if args.compare:
        compare_encoders(args.data_dir, args.output_dir, scale, args.epochs)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
