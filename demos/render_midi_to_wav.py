#!/usr/bin/env python3
"""Render a MIDI file to a WAV file through an instrument AudioUnit.

Loads a Standard MIDI File, plays it offline through a software instrument
(Apple's ``DLSMusicDevice`` by default, which ships with macOS), and writes the
result as a 16-bit WAV. This is the file-to-file path in
``coremusic.audio.audiounit_host.render_midi_file``.

Usage::

    python demos/render_midi_to_wav.py                        # demo.mid -> out_midi.wav
    python demos/render_midi_to_wav.py song.mid song.wav
    python demos/render_midi_to_wav.py song.mid song.wav --instrument AUSampler
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coremusic.audio.audiounit_host import render_midi_file

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MIDI = REPO_ROOT / "tests" / "data" / "midi" / "demo.mid"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "midi",
        nargs="?",
        default=str(DEFAULT_MIDI),
        help=f"Input MIDI file (default: {DEFAULT_MIDI})",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="out_midi.wav",
        help="Output WAV file (default: out_midi.wav)",
    )
    parser.add_argument(
        "--instrument",
        default="DLSMusicDevice",
        help="Instrument AudioUnit name (default: DLSMusicDevice)",
    )
    parser.add_argument(
        "--sample-rate", type=float, default=44100.0, help="Sample rate in Hz"
    )
    parser.add_argument(
        "--tail",
        type=float,
        default=1.0,
        help="Extra seconds rendered after the last note, for reverb/release tails",
    )
    args = parser.parse_args()

    midi_path = Path(args.midi)
    if not midi_path.exists():
        print(f"MIDI file not found: {midi_path}", file=sys.stderr)
        return 1

    print(f"Instrument: {args.instrument}")
    print(f"MIDI      : {midi_path.name}")

    try:
        out = render_midi_file(
            args.instrument,
            str(midi_path),
            args.output,
            sample_rate=args.sample_rate,
            tail_seconds=args.tail,
        )
    except Exception as exc:  # noqa: BLE001 - report and exit for a CLI example
        print(f"Render failed: {exc}", file=sys.stderr)
        print(
            "The instrument must be an AudioUnit of type 'aumu'. "
            "List instruments with `get_audiounit_names(filter_type='aumu')`.",
            file=sys.stderr,
        )
        return 1

    print(f"Output    : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
