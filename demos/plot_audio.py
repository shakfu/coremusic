#!/usr/bin/env python3
"""Plot a waveform, a spectrogram, or a frequency spectrum from an audio file.

The three plotters in ``coremusic.audio.visualization`` each render one view of
a file. This writes any of them to a PNG, so it works headless; drop
``--output`` to open an interactive window instead.

Usage::

    python demos/plot_audio.py tests/data/wav/amen.wav
    python demos/plot_audio.py song.wav --mode spectrogram -o spec.png
    python demos/plot_audio.py song.wav --mode spectrum --time 1.5
    python demos/plot_audio.py song.wav --mode waveform --rms --peaks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_plotter(mode: str, path: str):
    """Return the plotter for `mode`, imported lazily so --help always works."""
    from coremusic.audio.visualization import (
        FrequencySpectrumPlotter,
        SpectrogramPlotter,
        WaveformPlotter,
    )

    return {
        "waveform": WaveformPlotter,
        "spectrogram": SpectrogramPlotter,
        "spectrum": FrequencySpectrumPlotter,
    }[mode](path)


def plot_kwargs(args: argparse.Namespace) -> dict:
    """Options that differ per plotter."""
    if args.mode == "waveform":
        return {"show_rms": args.rms, "show_peaks": args.peaks}
    if args.mode == "spectrogram":
        return {"window_size": args.window_size, "cmap": args.cmap}
    return {"time": args.time, "window_size": args.window_size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="Audio file to plot")
    parser.add_argument(
        "--mode",
        choices=["waveform", "spectrogram", "spectrum"],
        default="waveform",
        help="Which view to render (default: waveform)",
    )
    parser.add_argument(
        "-o", "--output", help="Write a PNG here instead of showing a window"
    )
    parser.add_argument(
        "--time", type=float, default=0.0, help="Spectrum: time in seconds to analyse"
    )
    parser.add_argument(
        "--window-size", type=int, default=2048, help="FFT window size in samples"
    )
    parser.add_argument("--cmap", default="viridis", help="Spectrogram colour map")
    parser.add_argument(
        "--rms", action="store_true", help="Waveform: overlay the RMS envelope"
    )
    parser.add_argument(
        "--peaks", action="store_true", help="Waveform: overlay peak markers"
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output resolution")
    args = parser.parse_args()

    from coremusic.audio.visualization import MATPLOTLIB_AVAILABLE

    if not MATPLOTLIB_AVAILABLE:
        print("Plotting needs matplotlib: pip install 'coremusic[visualization]'")
        return 0

    if not Path(args.input).exists():
        print(f"File not found: {args.input}")
        return 1

    if args.output:
        # Render off-screen so this works over ssh and in CI
        import matplotlib

        matplotlib.use("Agg")

    plotter = build_plotter(args.mode, args.input)
    kwargs = plot_kwargs(args)

    if args.output:
        plotter.save(args.output, dpi=args.dpi, **kwargs)
        print(f"Wrote {args.mode} plot to {args.output}")
    else:
        import matplotlib.pyplot as plt

        plotter.plot(**kwargs)
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
