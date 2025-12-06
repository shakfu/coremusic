#!/usr/bin/env python3
"""Plot frequency spectrum at a specific time.

Usage:
    python spectrum.py [audio_file] [time_seconds]
"""

import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from coremusic.audio.visualization import FrequencySpectrumPlotter
except ImportError:
    print("Requires NumPy and matplotlib")
    sys.exit(1)


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "tests/amen.wav"
    time_point = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    plotter = FrequencySpectrumPlotter(audio_path)
    fig, ax = plotter.plot_at_time(time_point)

    print(f"Spectrum at {time_point}s: {audio_path}")
    print(f"  Figure size: {fig.get_size_inches()}")

    plt.close(fig)


if __name__ == "__main__":
    main()
