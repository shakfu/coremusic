#!/usr/bin/env python3
"""Demo script showcasing SciPy signal processing integration with CoreMusic.

This script demonstrates various audio DSP operations using CoreMusic's SciPy integration,
including filtering, resampling, and spectral analysis.

Requirements:
    - NumPy (pip install numpy)
    - SciPy (pip install scipy)
    - matplotlib (optional, for visualization)
"""

import os
import sys

# Add src to path for demo purposes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import coremusic as cm
import coremusic.utils.scipy as spu


def check_dependencies():
    """Check if required dependencies are available"""
    if not cm.NUMPY_AVAILABLE:
        print("NumPy is required for this demo. Install with: pip install numpy")
        sys.exit(1)

    if not spu.SCIPY_AVAILABLE:
        print("SciPy is required for this demo. Install with: pip install scipy")
        sys.exit(1)

    try:
        import matplotlib.pyplot as plt

        return True
    except ImportError:
        print("Note: matplotlib not available - skipping visualization examples")
        return False


def demo_1_filter_design():
    """Demo 1: Design and inspect various filters"""
    print("=" * 80)
    print("DEMO 1: Filter Design")
    print("=" * 80)

    # Design lowpass filter
    print("\n1. Designing 5th-order Butterworth lowpass filter at 1kHz:")
    b, a = spu.design_butterworth_filter(cutoff=1000, sample_rate=44100, order=5)
    print(f"   Filter coefficients: b={len(b)} coeffs, a={len(a)} coeffs")

    # Design bandpass filter
    print("\n2. Designing bandpass filter (300-3000 Hz):")
    b, a = spu.design_butterworth_filter(
        cutoff=(300, 3000), sample_rate=44100, order=4, filter_type="bandpass"
    )
    print(f"   Filter coefficients: b={len(b)} coeffs, a={len(a)} coeffs")

    # Design Chebyshev filter
    print("\n3. Designing Chebyshev Type I filter:")
    b, a = spu.design_chebyshev_filter(
        cutoff=2000, sample_rate=44100, order=5, ripple_db=0.5
    )
    print(f"   Filter coefficients: b={len(b)} coeffs, a={len(a)} coeffs")

    print("\n✓ Filter design demo complete\n")


def demo_2_filter_application():
    """Demo 2: Apply filters to audio file"""
    print("=" * 80)
    print("DEMO 2: Filter Application")
    print("=" * 80)

    test_file = os.path.join("tests", "amen.wav")
    if not os.path.exists(test_file):
        print(f"Skipping demo - test file not found: {test_file}")
        return

    print(f"\nLoading audio file: {test_file}")
    with cm.AudioFile(test_file) as af:
        audio = af.read_as_numpy()
        sample_rate = af.format.sample_rate
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Shape: {audio.shape}")
        print(f"  Duration: {af.duration:.2f}s")

        # Apply lowpass filter - convenient wrapper
        print("\n1. Applying lowpass filter (cutoff=2000 Hz)...")
        filtered = spu.apply_lowpass_filter(audio, cutoff=2000, sample_rate=sample_rate)
        print(f"   Output shape: {filtered.shape}")

        # Apply highpass filter - convenient wrapper
        print("\n2. Applying highpass filter (cutoff=100 Hz)...")
        filtered = spu.apply_highpass_filter(audio, cutoff=100, sample_rate=sample_rate)
        print(f"   Output shape: {filtered.shape}")

        # Apply bandpass filter - convenient wrapper
        print("\n3. Applying bandpass filter (300-5000 Hz)...")
        filtered = spu.apply_bandpass_filter(
            audio, lowcut=300, highcut=5000, sample_rate=sample_rate
        )
        print(f"   Output shape: {filtered.shape}")

        # Apply scipy filter directly - NEW convenience API
        print("\n4. Using scipy.signal filter directly...")
        import scipy.signal

        filtered = spu.apply_scipy_filter(
            audio, scipy.signal.butter(5, 1000, "low", fs=sample_rate)
        )
        print(f"   Output shape: {filtered.shape}")
        print("   (Applied scipy.signal.butter output directly!)")

    print("\n✓ Filter application demo complete\n")


def demo_3_signal_processor():
    """Demo 3: Use AudioSignalProcessor for method chaining"""
    print("=" * 80)
    print("DEMO 3: AudioSignalProcessor (Method Chaining)")
    print("=" * 80)

    test_file = os.path.join("tests", "amen.wav")
    if not os.path.exists(test_file):
        print(f"Skipping demo - test file not found: {test_file}")
        return

    print(f"\nLoading audio file: {test_file}")
    with cm.AudioFile(test_file) as af:
        audio = af.read_as_numpy()
        sample_rate = af.format.sample_rate

        print("\nCreating AudioSignalProcessor...")
        processor = spu.AudioSignalProcessor(audio, sample_rate)

        print("\nChaining operations:")
        print("  1. Highpass filter at 50 Hz (remove rumble)")
        print("  2. Lowpass filter at 15000 Hz (remove ultrasonic)")
        print("  3. Normalize to 0.9")

        processed = processor.highpass(50).lowpass(15000).normalize(0.9).get_audio()

        print(f"\nOriginal shape: {audio.shape}")
        print(f"Processed shape: {processed.shape}")

        import numpy as np

        print(f"Original peak: {np.max(np.abs(audio)):.6f}")
        print(f"Processed peak: {np.max(np.abs(processed)):.6f}")

    print("\n✓ AudioSignalProcessor demo complete\n")


def demo_4_resampling():
    """Demo 4: Resample audio to different sample rates"""
    print("=" * 80)
    print("DEMO 4: Audio Resampling")
    print("=" * 80)

    test_file = os.path.join("tests", "amen.wav")
    if not os.path.exists(test_file):
        print(f"Skipping demo - test file not found: {test_file}")
        return

    print(f"\nLoading audio file: {test_file}")
    with cm.AudioFile(test_file) as af:
        audio = af.read_as_numpy()
        sample_rate = af.format.sample_rate
        print(f"  Original sample rate: {sample_rate} Hz")
        print(f"  Original length: {len(audio)} samples")

        # Resample to 48kHz
        print("\n1. Resampling to 48000 Hz (FFT method)...")
        resampled_48k = spu.resample_audio(
            audio, original_rate=sample_rate, target_rate=48000, method="fft"
        )
        print(f"   Resampled length: {len(resampled_48k)} samples")
        print(f"   Ratio: {len(resampled_48k) / len(audio):.4f}")

        # Resample to 22.05kHz
        print("\n2. Resampling to 22050 Hz (polyphase method)...")
        resampled_22k = spu.resample_audio(
            audio, original_rate=sample_rate, target_rate=22050, method="polyphase"
        )
        print(f"   Resampled length: {len(resampled_22k)} samples")
        print(f"   Ratio: {len(resampled_22k) / len(audio):.4f}")

    print("\n✓ Resampling demo complete\n")


def demo_5_spectral_analysis():
    """Demo 5: Perform spectral analysis"""
    print("=" * 80)
    print("DEMO 5: Spectral Analysis")
    print("=" * 80)

    test_file = os.path.join("tests", "amen.wav")
    if not os.path.exists(test_file):
        print(f"Skipping demo - test file not found: {test_file}")
        return

    print(f"\nLoading audio file: {test_file}")
    with cm.AudioFile(test_file) as af:
        audio = af.read_as_numpy()
        sample_rate = af.format.sample_rate

        # Compute FFT
        print("\n1. Computing FFT...")
        frequencies, magnitudes = spu.compute_fft(audio, sample_rate, window="hann")
        print(f"   Number of frequency bins: {len(frequencies)}")
        print(f"   Frequency range: 0 to {frequencies[-1]:.1f} Hz")

        import numpy as np

        peak_idx = np.argmax(magnitudes)
        peak_freq = frequencies[peak_idx]
        print(f"   Peak frequency: {peak_freq:.1f} Hz")

        # Compute spectrum
        print("\n2. Computing power spectrum...")
        frequencies, spectrum = spu.compute_spectrum(audio, sample_rate)
        print(f"   Number of frequency bins: {len(frequencies)}")
        print(f"   Max power: {np.max(spectrum):.6f}")

        # Compute spectrogram
        print("\n3. Computing spectrogram...")
        frequencies, times, spectrogram = spu.compute_spectrogram(
            audio, sample_rate, nperseg=256
        )
        print(f"   Frequency bins: {len(frequencies)}")
        print(f"   Time bins: {len(times)}")
        print(f"   Spectrogram shape: {spectrogram.shape}")

    print("\n✓ Spectral analysis demo complete\n")


def demo_6_visualization(has_matplotlib):
    """Demo 6: Visualize audio processing results"""
    if not has_matplotlib:
        print("=" * 80)
        print("DEMO 6: Visualization (SKIPPED - matplotlib not available)")
        print("=" * 80)
        return

    print("=" * 80)
    print("DEMO 6: Visualization")
    print("=" * 80)

    import matplotlib.pyplot as plt
    import numpy as np

    test_file = os.path.join("tests", "amen.wav")
    if not os.path.exists(test_file):
        print(f"Skipping demo - test file not found: {test_file}")
        return

    print(f"\nLoading and processing audio: {test_file}")
    with cm.AudioFile(test_file) as af:
        audio = af.read_as_numpy()
        sample_rate = af.format.sample_rate

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 1. Waveform
        print("  Plotting waveform...")
        t = np.arange(len(audio)) / sample_rate
        if audio.ndim > 1:
            audio_plot = audio[:, 0]  # Use first channel
        else:
            audio_plot = audio
        axes[0, 0].plot(t, audio_plot)
        axes[0, 0].set_title("Waveform")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Amplitude")

        # 2. FFT
        print("  Plotting FFT...")
        freqs, mags = spu.compute_fft(audio, sample_rate)
        axes[0, 1].semilogy(freqs, mags)
        axes[0, 1].set_title("FFT Magnitude")
        axes[0, 1].set_xlabel("Frequency (Hz)")
        axes[0, 1].set_ylabel("Magnitude")
        axes[0, 1].set_xlim(0, 10000)  # Show up to 10kHz

        # 3. Spectrum
        print("  Plotting power spectrum...")
        freqs, spectrum = spu.compute_spectrum(audio, sample_rate)
        axes[1, 0].semilogy(freqs, spectrum)
        axes[1, 0].set_title("Power Spectrum")
        axes[1, 0].set_xlabel("Frequency (Hz)")
        axes[1, 0].set_ylabel("Power")
        axes[1, 0].set_xlim(0, 10000)

        # 4. Spectrogram
        print("  Plotting spectrogram...")
        freqs, times, Sxx = spu.compute_spectrogram(audio, sample_rate, nperseg=512)
        pcm = axes[1, 1].pcolormesh(
            times, freqs, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="viridis"
        )
        axes[1, 1].set_title("Spectrogram")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Frequency (Hz)")
        axes[1, 1].set_ylim(0, 10000)
        plt.colorbar(pcm, ax=axes[1, 1], label="Power (dB)")

        plt.tight_layout()
        print("\nDisplaying plots... (close window to continue)")
        plt.show()

    print("\n✓ Visualization demo complete\n")


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print(" CoreMusic + SciPy Integration Demo")
    print("=" * 80 + "\n")

    has_matplotlib = check_dependencies()

    print(f"CoreMusic version: Available")
    print(f"NumPy available: {cm.NUMPY_AVAILABLE}")
    print(f"SciPy available: {spu.SCIPY_AVAILABLE}")
    print(f"matplotlib available: {has_matplotlib}\n")

    demo_1_filter_design()
    demo_2_filter_application()
    demo_3_signal_processor()
    demo_4_resampling()
    demo_5_spectral_analysis()
    demo_6_visualization(has_matplotlib)

    print("=" * 80)
    print(" All demos completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
