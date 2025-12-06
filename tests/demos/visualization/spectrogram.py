#!/usr/bin/env python3
"""Plot audio spectrogram.

Usage:
    python spectrogram.py [audio_file] [output_file]
"""

import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from coremusic.audio.visualization import SpectrogramPlotter
except ImportError:
    print("Requires NumPy and matplotlib")
    sys.exit(1)


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "tests/amen.wav"
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    plotter = SpectrogramPlotter(audio_path)
    fig, ax = plotter.plot(colormap="viridis", freq_scale="log")

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    else:
        print(f"Spectrogram: {audio_path}")
        print(f"  Figure size: {fig.get_size_inches()}")

    plt.close(fig)


if __name__ == "__main__":
    main()
