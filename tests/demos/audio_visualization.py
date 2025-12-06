#!/usr/bin/env python3
"""Demo: Audio Visualization

Demonstrates the audio visualization capabilities of CoreMusic including:
- Waveform plotting with RMS and peak envelopes
- Spectrogram visualization with various colormaps
- Frequency spectrum analysis at specific times
- Average frequency spectrum over time ranges
- Saving visualizations to files

Requires NumPy and matplotlib.
"""

import sys
from pathlib import Path

BUILD_DIR = Path.cwd() / "build"
OUTPUT_DIR = BUILD_DIR / "visualization_outputs"
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available - demos will be skipped")

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for demo
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not available - demos will be skipped")

if NUMPY_AVAILABLE and MATPLOTLIB_AVAILABLE:
    from coremusic.audio.visualization import (
        WaveformPlotter,
        SpectrogramPlotter,
        FrequencySpectrumPlotter,
    )


def demo_basic_waveform():
    """Demo 1: Basic waveform plotting."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Waveform Plotting")
    print("=" * 70)

    print("\nPlotting basic waveform...")
    plotter = WaveformPlotter("tests/amen.wav")
    fig, ax = plotter.plot()

    print(f"Results:")
    print(f"  Created waveform plot")
    print(f"  Figure size: {fig.get_size_inches()}")
    print(f"  Title: {ax.get_title()}")

    plt.close(fig)
    print("  Plot closed (non-interactive mode)")


def demo_waveform_with_envelopes():
    """Demo 2: Waveform with RMS and peak envelopes."""
    print("\n" + "=" * 70)
    print("DEMO 2: Waveform with Envelopes")
    print("=" * 70)

    print("\nPlotting waveform with RMS and peak envelopes...")
    plotter = WaveformPlotter("tests/amen.wav")
    fig, ax = plotter.plot(show_rms=True, show_peaks=True, rms_window=0.05)

    print(f"Results:")
    print(f"  Created waveform plot with envelopes")
    print(f"  RMS window: 0.05 seconds")
    print(f"  Features overlaid:")
    print(f"    - Original waveform")
    print(f"    - RMS envelope (red)")
    print(f"    - Peak envelope (green)")

    plt.close(fig)
    print("  Plot closed (non-interactive mode)")


def demo_waveform_time_range():
    """Demo 3: Waveform for specific time range."""
    print("\n" + "=" * 70)
    print("DEMO 3: Waveform Time Range")
    print("=" * 70)

    print("\nPlotting waveform for time range 0.5s to 1.5s...")
    plotter = WaveformPlotter("tests/amen.wav")
    fig, ax = plotter.plot(time_range=(0.5, 1.5), show_rms=True)

    print(f"Results:")
    print(f"  Created zoom-in waveform plot")
    print(f"  Time range: 0.5s - 1.5s")
    print(f"  Duration shown: 1.0 second")

    plt.close(fig)
    print("  Plot closed (non-interactive mode)")


def demo_spectrogram():
    """Demo 4: Basic spectrogram."""
    print("\n" + "=" * 70)
    print("DEMO 4: Spectrogram Visualization")
    print("=" * 70)

    print("\nPlotting spectrogram...")
    plotter = SpectrogramPlotter("tests/amen.wav")
    fig, ax = plotter.plot(window_size=2048, hop_size=512)

    print(f"Results:")
    print(f"  Created spectrogram plot")
    print(f"  Window size: 2048 samples")
    print(f"  Hop size: 512 samples")
    print(f"  Colormap: viridis (default)")

    plt.close(fig)
    print("  Plot closed (non-interactive mode)")


def demo_spectrogram_colormaps():
    """Demo 5: Spectrograms with different colormaps."""
    print("\n" + "=" * 70)
    print("DEMO 5: Spectrogram Colormaps")
    print("=" * 70)

    plotter = SpectrogramPlotter("tests/amen.wav")
    colormaps = ["viridis", "magma", "plasma", "inferno"]

    print("\nGenerating spectrograms with different colormaps...")
    for cmap in colormaps:
        print(f"  - {cmap}: ", end="")
        fig, ax = plotter.plot(cmap=cmap, window_size=1024, hop_size=256)
        print(f"Created ({fig.get_size_inches()[0]}x{fig.get_size_inches()[1]} inches)")
        plt.close(fig)

    print("\nAll colormaps tested successfully")


def demo_frequency_spectrum():
    """Demo 6: Frequency spectrum at specific time."""
    print("\n" + "=" * 70)
    print("DEMO 6: Frequency Spectrum Analysis")
    print("=" * 70)

    print("\nAnalyzing frequency spectrum at t=1.0s...")
    plotter = FrequencySpectrumPlotter("tests/amen.wav")
    fig, ax = plotter.plot(time=1.0, window_size=4096, min_freq=20, max_freq=20000)

    print(f"Results:")
    print(f"  Created frequency spectrum plot")
    print(f"  Analysis time: 1.0 seconds")
    print(f"  Window size: 4096 samples")
    print(f"  Frequency range: 20 Hz - 20 kHz")
    print(f"  Scale: Logarithmic (frequency axis)")

    plt.close(fig)
    print("  Plot closed (non-interactive mode)")


def demo_average_spectrum():
    """Demo 7: Average frequency spectrum over time range."""
    print("\n" + "=" * 70)
    print("DEMO 7: Average Frequency Spectrum")
    print("=" * 70)

    print("\nComputing average spectrum over 0-2 seconds...")
    plotter = FrequencySpectrumPlotter("tests/amen.wav")
    fig, ax = plotter.plot_average(
        time_range=(0, 2), window_size=4096, hop_size=1024, min_freq=50, max_freq=15000
    )

    print(f"Results:")
    print(f"  Created average frequency spectrum")
    print(f"  Time range: 0-2 seconds")
    print(f"  Window size: 4096 samples")
    print(f"  Hop size: 1024 samples")
    print(f"  Frequency range: 50 Hz - 15 kHz")

    plt.close(fig)
    print("  Plot closed (non-interactive mode)")


def demo_compare_time_points():
    """Demo 8: Compare spectra at different time points."""
    print("\n" + "=" * 70)
    print("DEMO 8: Comparing Spectra at Different Times")
    print("=" * 70)

    plotter = FrequencySpectrumPlotter("tests/amen.wav")
    time_points = [0.5, 1.0, 1.5, 2.0]

    print("\nAnalyzing frequency spectrum at multiple time points...")
    for t in time_points:
        print(f"  Time {t:.1f}s: ", end="")
        fig, ax = plotter.plot(time=t, window_size=2048)
        print(f"Analyzed")
        plt.close(fig)

    print("\nAll time points analyzed successfully")


def demo_save_visualizations(tmp_dir=OUTPUT_DIR):
    """Demo 9: Saving visualizations to files."""
    print("\n" + "=" * 70)
    print("DEMO 9: Saving Visualizations to Files")
    print("=" * 70)

    # Create output directory
    import os

    os.makedirs(tmp_dir, exist_ok=True)
    print(f"\nSaving visualizations to {tmp_dir}/...")

    # Save waveform
    print("  1. Waveform with envelopes: ", end="")
    waveform = WaveformPlotter("tests/amen.wav")
    waveform.save(
        f"{tmp_dir}/waveform.png",
        show_rms=True,
        show_peaks=True,
        dpi=150,
    )
    print("Saved")

    # Save spectrogram
    print("  2. Spectrogram: ", end="")
    spectrogram = SpectrogramPlotter("tests/amen.wav")
    spectrogram.save(
        f"{tmp_dir}/spectrogram.png",
        window_size=2048,
        hop_size=512,
        cmap="magma",
        dpi=150,
    )
    print("Saved")

    # Save frequency spectrum
    print("  3. Frequency spectrum: ", end="")
    spectrum = FrequencySpectrumPlotter("tests/amen.wav")
    spectrum.save(
        f"{tmp_dir}/spectrum.png",
        time=1.0,
        window_size=4096,
        dpi=150,
    )
    print("Saved")

    print(f"\nAll visualizations saved to {tmp_dir}/")


def demo_complete_workflow():
    """Demo 10: Complete visualization workflow."""
    print("\n" + "=" * 70)
    print("DEMO 10: Complete Visualization Workflow")
    print("=" * 70)

    audio_file = "tests/amen.wav"

    print(f"\nAnalyzing {audio_file}...")
    print("\n  Step 1: Load audio and plot full waveform")
    waveform = WaveformPlotter(audio_file)
    fig1, _ = waveform.plot(show_rms=True)
    print(f"    Created waveform plot")
    plt.close(fig1)

    print("\n  Step 2: Generate time-frequency representation")
    spectrogram = SpectrogramPlotter(audio_file)
    fig2, _ = spectrogram.plot(cmap="viridis")
    print(f"    Created spectrogram plot")
    plt.close(fig2)

    print("\n  Step 3: Analyze frequency content at key moments")
    spectrum = FrequencySpectrumPlotter(audio_file)
    for t in [0.5, 1.0, 1.5]:
        fig3, _ = spectrum.plot(time=t)
        print(f"    Analyzed spectrum at {t:.1f}s")
        plt.close(fig3)

    print("\n  Step 4: Compute average spectral characteristics")
    fig4, _ = spectrum.plot_average(time_range=(0, 2))
    print(f"    Created average spectrum (0-2s)")
    plt.close(fig4)

    print("\n  Workflow complete!")


def demo_comparison():
    """Demo 11: Comparing different visualization parameters."""
    print("\n" + "=" * 70)
    print("DEMO 11: Parameter Comparison")
    print("=" * 70)

    print("\nComparing different FFT window sizes...")
    plotter = SpectrogramPlotter("tests/amen.wav")

    window_sizes = [512, 1024, 2048, 4096]
    for ws in window_sizes:
        fig, ax = plotter.plot(window_size=ws, hop_size=ws // 4)
        print(f"  Window size {ws:4d}: Created ({ws // 4} hop)")
        plt.close(fig)

    print("\nComparison complete!")
    print("  Observations:")
    print("    - Smaller windows: Better time resolution, worse frequency resolution")
    print("    - Larger windows: Better frequency resolution, worse time resolution")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("COREMUSIC AUDIO VISUALIZATION DEMO")
    print("=" * 70)
    print("\nThis demo showcases audio visualization capabilities:")
    print("- Waveform plotting with envelopes")
    print("- Spectrogram generation")
    print("- Frequency spectrum analysis")
    print("- Saving visualizations to files")
    print("- Complete analysis workflows")

    if not NUMPY_AVAILABLE:
        print("\nERROR: NumPy is required for visualization.")
        print("Install with: pip install numpy")
        return

    if not MATPLOTLIB_AVAILABLE:
        print("\nERROR: matplotlib is required for visualization.")
        print("Install with: pip install matplotlib")
        return

    try:
        demo_basic_waveform()
        demo_waveform_with_envelopes()
        demo_waveform_time_range()
        demo_spectrogram()
        demo_spectrogram_colormaps()
        demo_frequency_spectrum()
        demo_average_spectrum()
        demo_compare_time_points()
        demo_save_visualizations()
        demo_complete_workflow()
        demo_comparison()

        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("- WaveformPlotter provides easy waveform visualization")
        print("- RMS and peak envelopes show signal dynamics")
        print("- SpectrogramPlotter reveals time-frequency structure")
        print("- Multiple colormaps available for different preferences")
        print("- FrequencySpectrumPlotter analyzes frequency content")
        print("- Average spectra show overall spectral characteristics")
        print("- All plotters support saving to file formats")
        print("- Integration with CoreMusic's AudioFile system")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
