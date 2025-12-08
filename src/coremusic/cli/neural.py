"""Neural network music generation commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._formatters import output_json
from ._utils import EXIT_SUCCESS, CLIError

# Model type choices
MODEL_TYPES = ["mlp", "rnn", "lstm", "gru"]

# Encoder type choices
ENCODER_TYPES = ["note", "event"]


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register neural network commands."""
    parser = subparsers.add_parser("neural", help="Neural network music generation")
    neural_sub = parser.add_subparsers(dest="neural_command", metavar="<subcommand>")

    # neural train
    train_parser = neural_sub.add_parser("train", help="Train a neural network on MIDI files")
    train_parser.add_argument("input", nargs="+", help="Input MIDI file(s) or glob patterns")
    train_parser.add_argument(
        "--output", "-o", required=True,
        help="Output model file path (.kan)"
    )
    train_parser.add_argument(
        "--model", "-m", choices=MODEL_TYPES, default="lstm",
        help="Model architecture (default: lstm)"
    )
    train_parser.add_argument(
        "--encoder", "-e", choices=ENCODER_TYPES, default="note",
        help="Encoder type (default: note)"
    )
    train_parser.add_argument(
        "--seq-length", type=int, default=32,
        help="Sequence length (default: 32)"
    )
    train_parser.add_argument(
        "--hidden-size", type=int, default=256,
        help="Hidden layer size (default: 256)"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100,
        help="Maximum training epochs (default: 100)"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Mini-batch size (default: 64)"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=0.001,
        help="Learning rate (default: 0.001)"
    )
    train_parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable data augmentation"
    )
    train_parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Quiet mode (minimal output)"
    )
    train_parser.set_defaults(func=cmd_train)

    # neural generate
    gen_parser = neural_sub.add_parser("generate", help="Generate music from trained model")
    gen_parser.add_argument("model", help="Path to trained model (.kan)")
    gen_parser.add_argument("output", help="Output MIDI file path")
    gen_parser.add_argument(
        "--duration", "-d", type=int, default=32,
        help="Duration in beats (default: 32)"
    )
    gen_parser.add_argument(
        "--temperature", "-t", type=float, default=1.0,
        help="Sampling temperature (default: 1.0, lower=more conservative)"
    )
    gen_parser.add_argument(
        "--seed", "-s", type=str, default=None,
        help="Seed MIDI file (optional)"
    )
    gen_parser.add_argument(
        "--encoder", "-e", choices=ENCODER_TYPES, default="note",
        help="Encoder type used during training (default: note)"
    )
    gen_parser.add_argument(
        "--seq-length", type=int, default=32,
        help="Sequence length used during training (default: 32)"
    )
    gen_parser.add_argument(
        "--tempo", type=float, default=120.0,
        help="Output tempo in BPM (default: 120)"
    )
    gen_parser.set_defaults(func=cmd_generate)

    # neural continue
    cont_parser = neural_sub.add_parser("continue", help="Continue an existing MIDI file")
    cont_parser.add_argument("model", help="Path to trained model (.kan)")
    cont_parser.add_argument("input", help="Input MIDI file to continue")
    cont_parser.add_argument("output", help="Output MIDI file path")
    cont_parser.add_argument(
        "--bars", "-b", type=int, default=8,
        help="Number of bars to add (default: 8)"
    )
    cont_parser.add_argument(
        "--temperature", "-t", type=float, default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    cont_parser.add_argument(
        "--encoder", "-e", choices=ENCODER_TYPES, default="note",
        help="Encoder type used during training (default: note)"
    )
    cont_parser.add_argument(
        "--seq-length", type=int, default=32,
        help="Sequence length used during training (default: 32)"
    )
    cont_parser.set_defaults(func=cmd_continue)

    # neural evaluate
    eval_parser = neural_sub.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("model", help="Path to trained model (.kan)")
    eval_parser.add_argument(
        "--test", "-t", nargs="+", required=True,
        help="Test MIDI file(s)"
    )
    eval_parser.add_argument(
        "--encoder", "-e", choices=ENCODER_TYPES, default="note",
        help="Encoder type used during training (default: note)"
    )
    eval_parser.add_argument(
        "--seq-length", type=int, default=32,
        help="Sequence length used during training (default: 32)"
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    # neural info
    info_parser = neural_sub.add_parser("info", help="Show model information")
    info_parser.add_argument("model", help="Path to trained model (.kan)")
    info_parser.set_defaults(func=cmd_info)

    parser.set_defaults(func=lambda args: parser.print_help() or EXIT_SUCCESS)


def cmd_train(args: argparse.Namespace) -> int:
    """Train a neural network on MIDI files."""
    from coremusic.music.neural.api import train_music_model

    output_path = Path(args.output)
    verbose = 0 if args.quiet else 1

    try:
        model = train_music_model(
            midi_files=args.input,
            model_type=args.model,
            output_path=str(output_path),
            encoder_type=args.encoder,
            seq_length=args.seq_length,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            augment=not args.no_augment,
            verbose=verbose,
        )

        if args.json:
            output_json({
                "output": str(output_path.absolute()),
                "model_type": args.model,
                "encoder": args.encoder,
                "parameters": model.n_var,
            })
        elif not args.quiet:
            print(f"\nModel saved to: {output_path}")

        return EXIT_SUCCESS

    except Exception as e:
        raise CLIError(f"Training failed: {e}")


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate music from a trained model."""
    from coremusic.music.neural.api import generate_music

    model_path = Path(args.model)
    output_path = Path(args.output)

    if not model_path.exists():
        raise CLIError(f"Model not found: {model_path}")

    try:
        generate_music(
            model_path=str(model_path),
            output_path=str(output_path),
            duration=args.duration,
            temperature=args.temperature,
            seed_midi=args.seed,
            encoder_type=args.encoder,
            seq_length=args.seq_length,
            tempo=args.tempo,
        )

        if args.json:
            output_json({
                "output": str(output_path.absolute()),
                "duration": args.duration,
                "temperature": args.temperature,
                "tempo": args.tempo,
            })
        else:
            print(f"Generated music: {output_path}")
            print(f"  Duration: {args.duration} beats")
            print(f"  Temperature: {args.temperature}")
            print(f"  Tempo: {args.tempo} BPM")

        return EXIT_SUCCESS

    except Exception as e:
        raise CLIError(f"Generation failed: {e}")


def cmd_continue(args: argparse.Namespace) -> int:
    """Continue an existing MIDI file."""
    from coremusic.music.neural.api import continue_music

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not model_path.exists():
        raise CLIError(f"Model not found: {model_path}")
    if not input_path.exists():
        raise CLIError(f"Input MIDI not found: {input_path}")

    try:
        continue_music(
            model_path=str(model_path),
            input_midi=str(input_path),
            output_path=str(output_path),
            bars=args.bars,
            temperature=args.temperature,
            encoder_type=args.encoder,
            seq_length=args.seq_length,
        )

        if args.json:
            output_json({
                "input": str(input_path.absolute()),
                "output": str(output_path.absolute()),
                "bars_added": args.bars,
                "temperature": args.temperature,
            })
        else:
            print(f"Continued MIDI: {output_path}")
            print(f"  Input: {input_path}")
            print(f"  Added: {args.bars} bars")
            print(f"  Temperature: {args.temperature}")

        return EXIT_SUCCESS

    except Exception as e:
        raise CLIError(f"Continuation failed: {e}")


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a trained model."""
    from coremusic.kann import NeuralNetwork
    from coremusic.music.neural import (BaseEncoder, EventEncoder, MIDIDataset,
                                        MusicMetrics, NoteEncoder)

    model_path = Path(args.model)

    if not model_path.exists():
        raise CLIError(f"Model not found: {model_path}")

    try:
        # Load model
        model = NeuralNetwork.load(str(model_path))

        # Create encoder
        encoder: BaseEncoder
        if args.encoder == "event":
            encoder = EventEncoder()
        else:
            encoder = NoteEncoder()

        # Load test data
        dataset = MIDIDataset(encoder, seq_length=args.seq_length)
        for pattern in args.test:
            import glob
            matches = glob.glob(pattern, recursive=True)
            if matches:
                for path in matches:
                    try:
                        dataset.load_file(path)
                    except Exception:
                        pass
            else:
                try:
                    dataset.load_file(pattern)
                except Exception:
                    pass

        if dataset.n_sequences == 0:
            raise CLIError("No valid test files loaded")

        # Prepare data and compute loss
        x_test, y_test = dataset.prepare_training_data(use_numpy=True)
        loss = model.cost(x_test, y_test)

        # Generate sample and compute metrics
        from coremusic.music.neural import MusicGenerator, TemperatureSampling

        generator = MusicGenerator(
            model, encoder, args.seq_length,
            sampling=TemperatureSampling(0.8)
        )

        seed = dataset.get_sample_sequence(args.seq_length)
        generated = generator.generate(seed=seed, length=64)
        reference = dataset.sequences[0] if dataset.sequences else []

        metrics = MusicMetrics.evaluate_sequence(generated, reference)

        if args.json:
            output_json({
                "model": str(model_path.absolute()),
                "test_files": dataset.n_sequences,
                "test_samples": len(x_test),
                "loss": loss,
                "metrics": metrics,
            })
        else:
            print(f"Model Evaluation: {model_path}")
            print(f"  Test files: {dataset.n_sequences}")
            print(f"  Test samples: {len(x_test)}")
            print(f"  Loss: {loss:.6f}")
            print("\nGenerated Sample Metrics:")
            print(f"  Pitch similarity: {metrics.get('pitch_similarity', 0):.4f}")
            print(f"  Interval similarity: {metrics.get('interval_similarity', 0):.4f}")
            print(f"  4-gram repetition: {metrics.get('repetition_4gram', 0):.4f}")
            print(f"  Unique notes: {int(metrics.get('unique_notes', 0))}")
            print(f"  Pitch range: {int(metrics.get('pitch_range', 0))}")

        return EXIT_SUCCESS

    except Exception as e:
        raise CLIError(f"Evaluation failed: {e}")


def cmd_info(args: argparse.Namespace) -> int:
    """Show model information."""
    from coremusic.kann import NeuralNetwork

    model_path = Path(args.model)

    if not model_path.exists():
        raise CLIError(f"Model not found: {model_path}")

    try:
        model = NeuralNetwork.load(str(model_path))

        if args.json:
            output_json({
                "path": str(model_path.absolute()),
                "input_dim": model.input_dim,
                "output_dim": model.output_dim,
                "parameters": model.n_var,
                "constants": model.n_const,
            })
        else:
            print(f"Model: {model_path}")
            print(f"  Input dimension: {model.input_dim}")
            print(f"  Output dimension: {model.output_dim}")
            print(f"  Trainable parameters: {model.n_var}")
            print(f"  Constants: {model.n_const}")

        return EXIT_SUCCESS

    except Exception as e:
        raise CLIError(f"Failed to load model: {e}")
