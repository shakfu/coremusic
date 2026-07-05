#!/usr/bin/env python3
"""Host an AudioUnit effect chain and process a WAV file through it.

Loads an input WAV, runs it through a chain of Apple AudioUnit effects
(AUDelay -> AUMatrixReverb by default), and writes the processed result to a
new WAV. This shows the buffer-in / buffer-out hosting API in
``coremusic.audio.audiounit_host``.

Usage::

    python demos/host_au_chain.py                       # amen.wav -> out_chain.wav
    python demos/host_au_chain.py in.wav out.wav
    python demos/host_au_chain.py in.wav out.wav --wet 0.6

The example uses the stdlib ``wave`` module for file I/O so the focus stays on
the AudioUnit hosting itself; audio is processed as 16-bit PCM, which the chain
converts to the AudioUnit-native float32 internally.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

from coremusic.audio.audiounit_host import AudioUnitChain, PluginAudioFormat

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "tests" / "data" / "wav" / "amen.wav"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help=f"Input WAV file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="out_chain.wav",
        help="Output WAV file (default: out_chain.wav)",
    )
    parser.add_argument(
        "--effects",
        nargs="+",
        default=["AUDelay", "AUMatrixReverb"],
        help="AudioUnit effect names, in order (default: AUDelay AUMatrixReverb)",
    )
    parser.add_argument(
        "--wet",
        type=float,
        default=1.0,
        help="Wet/dry mix, 0.0 (dry) to 1.0 (fully processed). Default: 1.0",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        return 1

    # Read the source WAV as interleaved 16-bit PCM.
    with wave.open(str(in_path), "rb") as w:
        channels = w.getnchannels()
        sample_rate = float(w.getframerate())
        sample_width = w.getsampwidth()
        num_frames = w.getnframes()
        pcm = w.readframes(num_frames)

    if sample_width != 2:
        print(
            f"This example expects 16-bit PCM input; got {sample_width * 8}-bit.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Input : {in_path.name}  {sample_rate:.0f} Hz, {channels} ch, "
        f"{num_frames} frames"
    )

    fmt = PluginAudioFormat(
        sample_rate=sample_rate,
        channels=channels,
        sample_format=PluginAudioFormat.INT16,
    )

    try:
        with AudioUnitChain(fmt) as chain:
            for name in args.effects:
                chain.add_plugin(name)
            print(f"Chain : {' -> '.join(args.effects)}  (wet={args.wet})")
            processed = chain.process(pcm, wet_dry_mix=args.wet)
    except Exception as exc:  # noqa: BLE001 - report and exit for a CLI example
        print(f"Could not host the effect chain: {exc}", file=sys.stderr)
        print(
            "Check the effect names with `coremusic doctor` or "
            "`get_audiounit_names()`.",
            file=sys.stderr,
        )
        return 1

    # Write the processed audio back out with the same PCM parameters.
    out_path = Path(args.output)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(processed)

    print(f"Output: {out_path}  ({len(processed)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
