"""Generative music commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._formatters import output_json
from ._utils import EXIT_SUCCESS, CLIError

# Arpeggio pattern choices
ARP_PATTERNS = ["up", "down", "up_down", "down_up", "random", "as_played"]

# Scale type choices
SCALE_TYPES = [
    "major", "natural_minor", "harmonic_minor", "melodic_minor",
    "dorian", "phrygian", "lydian", "mixolydian",
    "aeolian", "locrian", "major_pentatonic", "minor_pentatonic",
    "blues", "chromatic", "whole_tone"
]

# Chord type choices
CHORD_TYPES = [
    "major", "minor", "dim", "aug", "sus2", "sus4",
    "maj7", "min7", "dom7", "dim7", "m7b5"
]


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register generate commands."""
    parser = subparsers.add_parser("generate", help="Generative music algorithms")
    gen_sub = parser.add_subparsers(dest="generate_command", metavar="<subcommand>")

    # generate arpeggio
    arp_parser = gen_sub.add_parser("arpeggio", help="Generate arpeggio pattern")
    arp_parser.add_argument("output", help="Output MIDI file path")
    arp_parser.add_argument(
        "--root", "-r", default="C4",
        help="Root note (e.g., C4, F#3) (default: C4)"
    )
    arp_parser.add_argument(
        "--chord", "-c", choices=CHORD_TYPES, default="major",
        help="Chord type (default: major)"
    )
    arp_parser.add_argument(
        "--pattern", "-p", choices=ARP_PATTERNS, default="up",
        help="Arpeggio pattern (default: up)"
    )
    arp_parser.add_argument(
        "--cycles", type=int, default=4,
        help="Number of cycles (default: 4)"
    )
    arp_parser.add_argument(
        "--tempo", "-t", type=float, default=120.0,
        help="Tempo in BPM (default: 120)"
    )
    arp_parser.add_argument(
        "--velocity", "-v", type=int, default=100,
        help="Note velocity 0-127 (default: 100)"
    )
    arp_parser.set_defaults(func=cmd_arpeggio)

    # generate euclidean
    euclid_parser = gen_sub.add_parser("euclidean", help="Generate Euclidean rhythm")
    euclid_parser.add_argument("output", help="Output MIDI file path")
    euclid_parser.add_argument(
        "--pulses", "-p", type=int, default=5,
        help="Number of pulses/hits (default: 5)"
    )
    euclid_parser.add_argument(
        "--steps", "-s", type=int, default=8,
        help="Total number of steps (default: 8)"
    )
    euclid_parser.add_argument(
        "--pitch", type=int, default=36,
        help="MIDI pitch for hits (default: 36 = kick drum)"
    )
    euclid_parser.add_argument(
        "--cycles", type=int, default=4,
        help="Number of cycles (default: 4)"
    )
    euclid_parser.add_argument(
        "--tempo", "-t", type=float, default=120.0,
        help="Tempo in BPM (default: 120)"
    )
    euclid_parser.add_argument(
        "--velocity", "-v", type=int, default=100,
        help="Note velocity 0-127 (default: 100)"
    )
    euclid_parser.set_defaults(func=cmd_euclidean)

    # generate melody
    melody_parser = gen_sub.add_parser("melody", help="Generate melodic pattern")
    melody_parser.add_argument("output", help="Output MIDI file path")
    melody_parser.add_argument(
        "--root", "-r", default="C4",
        help="Root note (e.g., C4, F#3) (default: C4)"
    )
    melody_parser.add_argument(
        "--scale", "-s", choices=SCALE_TYPES, default="major",
        help="Scale type (default: major)"
    )
    melody_parser.add_argument(
        "--notes", "-n", type=int, default=32,
        help="Number of notes to generate (default: 32)"
    )
    melody_parser.add_argument(
        "--tempo", "-t", type=float, default=120.0,
        help="Tempo in BPM (default: 120)"
    )
    melody_parser.add_argument(
        "--velocity", "-v", type=int, default=100,
        help="Note velocity 0-127 (default: 100)"
    )
    melody_parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    melody_parser.set_defaults(func=cmd_melody)

    parser.set_defaults(func=lambda args: parser.print_help() or EXIT_SUCCESS)


def _parse_note(note_str: str):
    """Parse note string like 'C4' or 'F#3' into Note object."""
    from coremusic.music.theory import Note

    note_str = note_str.strip()
    if not note_str:
        raise CLIError("Empty note string")

    # Extract note name and octave
    if len(note_str) >= 2 and note_str[1] in "#b":
        note_name = note_str[:2]
        octave_str = note_str[2:]
    else:
        note_name = note_str[0]
        octave_str = note_str[1:]

    try:
        octave = int(octave_str) if octave_str else 4
    except ValueError:
        raise CLIError(f"Invalid octave in note: {note_str}")

    return Note(note_name, octave)


def _get_chord_type(chord_name: str):
    """Get ChordType enum from string."""
    from coremusic.music.theory import ChordType

    mapping = {
        "major": ChordType.MAJOR,
        "minor": ChordType.MINOR,
        "dim": ChordType.DIMINISHED,
        "aug": ChordType.AUGMENTED,
        "sus2": ChordType.SUS2,
        "sus4": ChordType.SUS4,
        "maj7": ChordType.MAJOR_7,
        "min7": ChordType.MINOR_7,
        "dom7": ChordType.DOMINANT_7,
        "dim7": ChordType.DIMINISHED_7,
        "m7b5": ChordType.HALF_DIMINISHED_7,
    }
    return mapping.get(chord_name, ChordType.MAJOR)


def _get_scale_type(scale_name: str):
    """Get ScaleType enum from string."""
    from coremusic.music.theory import ScaleType

    mapping = {
        "major": ScaleType.MAJOR,
        "natural_minor": ScaleType.NATURAL_MINOR,
        "harmonic_minor": ScaleType.HARMONIC_MINOR,
        "melodic_minor": ScaleType.MELODIC_MINOR,
        "dorian": ScaleType.DORIAN,
        "phrygian": ScaleType.PHRYGIAN,
        "lydian": ScaleType.LYDIAN,
        "mixolydian": ScaleType.MIXOLYDIAN,
        "aeolian": ScaleType.AEOLIAN,
        "locrian": ScaleType.LOCRIAN,
        "major_pentatonic": ScaleType.MAJOR_PENTATONIC,
        "minor_pentatonic": ScaleType.MINOR_PENTATONIC,
        "blues": ScaleType.BLUES,
        "chromatic": ScaleType.CHROMATIC,
        "whole_tone": ScaleType.WHOLE_TONE,
    }
    return mapping.get(scale_name, ScaleType.MAJOR)


def _get_arp_pattern(pattern_name: str):
    """Get ArpPattern enum from string."""
    from coremusic.music.generative import ArpPattern

    mapping = {
        "up": ArpPattern.UP,
        "down": ArpPattern.DOWN,
        "up_down": ArpPattern.UP_DOWN,
        "down_up": ArpPattern.DOWN_UP,
        "random": ArpPattern.RANDOM,
        "as_played": ArpPattern.AS_PLAYED,
    }
    return mapping.get(pattern_name, ArpPattern.UP)


def _save_events_to_midi(events, output_path: Path, tempo: float):
    """Save MIDIEvents to a MIDI file."""
    from coremusic.midi.utilities import MIDISequence, MIDIStatus

    seq = MIDISequence(tempo=tempo)
    track = seq.add_track("Generated")

    # Convert MIDIEvents to note on/off pairs
    # Events are already time-sorted note on/off events
    note_ons = {}  # Track note-on events to pair with note-off

    for event in events:
        if event.status == MIDIStatus.NOTE_ON and event.data2 > 0:
            # Note on - store start time
            key = (event.channel, event.data1)
            note_ons[key] = (event.time, event.data2)
        elif event.status == MIDIStatus.NOTE_OFF or (event.status == MIDIStatus.NOTE_ON and event.data2 == 0):
            # Note off - find matching note on
            key = (event.channel, event.data1)
            if key in note_ons:
                start_time, velocity = note_ons.pop(key)
                duration = event.time - start_time
                if duration > 0:
                    track.add_note(start_time, event.data1, velocity, duration, event.channel)

    seq.save(str(output_path))


def cmd_arpeggio(args: argparse.Namespace) -> int:
    """Generate arpeggio pattern."""
    from coremusic.music.generative import ArpConfig, Arpeggiator
    from coremusic.music.theory import Chord

    output_path = Path(args.output)

    # Parse root note and chord
    root = _parse_note(args.root)
    chord_type = _get_chord_type(args.chord)
    chord = Chord(root, chord_type)

    # Get pattern
    pattern = _get_arp_pattern(args.pattern)

    # Create config and generator
    config = ArpConfig(
        tempo=args.tempo,
        velocity=args.velocity,
    )
    arp = Arpeggiator(chord, pattern, config=config)

    # Generate events
    events = arp.generate(num_cycles=args.cycles)

    # Save to MIDI
    _save_events_to_midi(events, output_path, args.tempo)

    if args.json:
        output_json({
            "output": str(output_path.absolute()),
            "root": str(root),
            "chord": args.chord,
            "pattern": args.pattern,
            "cycles": args.cycles,
            "tempo": args.tempo,
            "events": len(events),
        })
    else:
        print(f"Generated arpeggio: {output_path.name}")
        print(f"  Chord: {root} {args.chord}")
        print(f"  Pattern: {args.pattern}")
        print(f"  Cycles: {args.cycles}")
        print(f"  Tempo: {args.tempo} BPM")
        print(f"  Events: {len(events)}")

    return EXIT_SUCCESS


def cmd_euclidean(args: argparse.Namespace) -> int:
    """Generate Euclidean rhythm."""
    from coremusic.music.generative import EuclideanConfig, EuclideanGenerator

    output_path = Path(args.output)

    if args.pulses > args.steps:
        raise CLIError(f"Pulses ({args.pulses}) cannot exceed steps ({args.steps})")

    # Create config and generator
    config = EuclideanConfig(
        tempo=args.tempo,
        velocity=args.velocity,
    )
    euclid = EuclideanGenerator(
        pulses=args.pulses,
        steps=args.steps,
        pitch=args.pitch,
        config=config,
    )

    # Generate events
    events = euclid.generate(cycles=args.cycles)

    # Save to MIDI
    _save_events_to_midi(events, output_path, args.tempo)

    # Get pattern as string
    pattern_list = euclid.get_pattern()
    pattern_str = "".join("x" if p else "." for p in pattern_list)

    if args.json:
        output_json({
            "output": str(output_path.absolute()),
            "pulses": args.pulses,
            "steps": args.steps,
            "pattern": pattern_str,
            "pitch": args.pitch,
            "cycles": args.cycles,
            "tempo": args.tempo,
            "events": len(events),
        })
    else:
        print(f"Generated Euclidean rhythm: {output_path.name}")
        print(f"  Pattern: E({args.pulses}, {args.steps}) = [{pattern_str}]")
        print(f"  Pitch: {args.pitch}")
        print(f"  Cycles: {args.cycles}")
        print(f"  Tempo: {args.tempo} BPM")
        print(f"  Events: {len(events)}")

    return EXIT_SUCCESS


def cmd_melody(args: argparse.Namespace) -> int:
    """Generate melodic pattern."""
    from coremusic.music.generative import MelodyConfig, MelodyGenerator
    from coremusic.music.theory import Scale

    output_path = Path(args.output)

    # Parse root note and scale
    root = _parse_note(args.root)
    scale_type = _get_scale_type(args.scale)
    scale = Scale(root, scale_type)

    # Create config and generator
    config = MelodyConfig(
        tempo=args.tempo,
        velocity=args.velocity,
        seed=args.seed,
    )
    melody = MelodyGenerator(scale, config=config)

    # Generate events
    events = melody.generate(num_notes=args.notes)

    # Save to MIDI
    _save_events_to_midi(events, output_path, args.tempo)

    if args.json:
        output_json({
            "output": str(output_path.absolute()),
            "root": str(root),
            "scale": args.scale,
            "notes": args.notes,
            "tempo": args.tempo,
            "seed": args.seed,
            "events": len(events),
        })
    else:
        print(f"Generated melody: {output_path.name}")
        print(f"  Scale: {root} {args.scale}")
        print(f"  Notes: {args.notes}")
        print(f"  Tempo: {args.tempo} BPM")
        if args.seed is not None:
            print(f"  Seed: {args.seed}")
        print(f"  Events: {len(events)}")

    return EXIT_SUCCESS
