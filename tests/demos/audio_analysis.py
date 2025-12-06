#!/usr/bin/env python3
"""Demo: Audio Analysis and Feature Extraction

Demonstrates the audio analysis capabilities of CoreMusic including:
- Beat detection and tempo estimation
- Pitch detection and tracking
- Spectral analysis (FFT, centroid, rolloff, peaks)
- MFCC extraction
- Chroma features and key detection
- Audio fingerprinting
- Real-time pitch detection

Requires NumPy and SciPy.
"""

import sys
import time
from pathlib import Path

# Add src to path for demo purposes
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import coremusic as cm

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available - demos will be skipped")

try:
    from scipy import signal

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not available - demos will be skipped")

if NUMPY_AVAILABLE and SCIPY_AVAILABLE:
    from coremusic.audio.analysis import (
        AudioAnalyzer,
        LivePitchDetector,
        BeatInfo,
        PitchInfo,
    )


def demo_beat_detection():
    """Demo 1: Beat detection and tempo estimation."""
    print("\n" + "=" * 70)
    print("DEMO 1: Beat Detection and Tempo Estimation")
    print("=" * 70)

    print("\nAnalyzing audio file for beats and tempo...")
    analyzer = AudioAnalyzer("tests/amen.wav")

    beat_info = analyzer.detect_beats()

    print(f"\nResults:")
    print(f"  Detected tempo: {beat_info.tempo:.1f} BPM")
    print(f"  Number of beats: {len(beat_info.beats)}")
    print(f"  Number of downbeats: {len(beat_info.downbeats)}")
    print(f"  Confidence: {beat_info.confidence:.2f}")

    if len(beat_info.beats) > 0:
        print(f"\n  First 5 beats (seconds):")
        for i, beat_time in enumerate(beat_info.beats[:5], 1):
            print(f"    Beat {i}: {beat_time:.3f}s")

    if len(beat_info.downbeats) > 0:
        print(f"\n  First 3 downbeats (seconds):")
        for i, downbeat_time in enumerate(beat_info.downbeats[:3], 1):
            print(f"    Downbeat {i}: {downbeat_time:.3f}s")


def demo_pitch_detection():
    """Demo 2: Pitch detection and tracking."""
    print("\n" + "=" * 70)
    print("DEMO 2: Pitch Detection and Tracking")
    print("=" * 70)

    print("\nDetecting pitch over time...")
    analyzer = AudioAnalyzer("tests/amen.wav")

    # Analyze first 0.5 seconds
    pitch_track = analyzer.detect_pitch(time_range=(0.0, 0.5))

    print(f"\nResults:")
    print(f"  Number of pitch detections: {len(pitch_track)}")

    if len(pitch_track) > 0:
        print(f"\n  First 5 pitch detections:")
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        for i, pitch_info in enumerate(pitch_track[:5], 1):
            note_name = note_names[pitch_info.midi_note % 12]
            octave = pitch_info.midi_note // 12 - 1
            print(
                f"    {i}. {note_name}{octave}: {pitch_info.frequency:.1f} Hz "
                f"(MIDI {pitch_info.midi_note}, {pitch_info.cents_offset:+.0f} cents)"
            )

        # Calculate average pitch
        avg_freq = np.mean([p.frequency for p in pitch_track])
        print(f"\n  Average frequency: {avg_freq:.1f} Hz")


def demo_spectral_analysis():
    """Demo 3: Spectral analysis."""
    print("\n" + "=" * 70)
    print("DEMO 3: Spectral Analysis")
    print("=" * 70)

    print("\nAnalyzing spectrum at specific time points...")
    analyzer = AudioAnalyzer("tests/amen.wav")

    # Analyze at different time points
    time_points = [0.1, 0.5, 1.0]

    for t in time_points:
        print(f"\n  Time: {t:.1f}s")
        spectrum = analyzer.analyze_spectrum(time=t, window_size=0.05)

        print(f"    Spectral centroid: {spectrum['centroid']:.1f} Hz")
        print(f"    Spectral rolloff: {spectrum['rolloff']:.1f} Hz")
        print(f"    Number of peaks: {len(spectrum['peaks'])}")

        if len(spectrum['peaks']) > 0:
            print(f"    Top 3 spectral peaks:")
            for i, (freq, mag) in enumerate(spectrum["peaks"][:3], 1):
                print(f"      {i}. {freq:.1f} Hz (magnitude: {mag:.2f})")


def demo_mfcc():
    """Demo 4: MFCC extraction."""
    print("\n" + "=" * 70)
    print("DEMO 4: MFCC Extraction")
    print("=" * 70)

    print("\nExtracting Mel-Frequency Cepstral Coefficients...")
    analyzer = AudioAnalyzer("tests/amen.wav")

    # Extract MFCCs
    n_mfcc = 13
    mfcc = analyzer.extract_mfcc(n_mfcc=n_mfcc)

    print(f"\nResults:")
    print(f"  MFCC shape: {mfcc.shape}")
    print(f"  Number of coefficients: {mfcc.shape[0]}")
    print(f"  Number of frames: {mfcc.shape[1]}")

    # Show statistics for first coefficient
    print(f"\n  First MFCC coefficient statistics:")
    print(f"    Mean: {np.mean(mfcc[0]):.3f}")
    print(f"    Std: {np.std(mfcc[0]):.3f}")
    print(f"    Min: {np.min(mfcc[0]):.3f}")
    print(f"    Max: {np.max(mfcc[0]):.3f}")


def demo_key_detection():
    """Demo 5: Key detection."""
    print("\n" + "=" * 70)
    print("DEMO 5: Musical Key Detection")
    print("=" * 70)

    print("\nDetecting musical key...")
    analyzer = AudioAnalyzer("tests/amen.wav")

    key, mode = analyzer.detect_key()

    print(f"\nResults:")
    print(f"  Detected key: {key} {mode}")


def demo_fingerprinting():
    """Demo 6: Audio fingerprinting."""
    print("\n" + "=" * 70)
    print("DEMO 6: Audio Fingerprinting")
    print("=" * 70)

    print("\nGenerating audio fingerprint...")
    analyzer = AudioAnalyzer("tests/amen.wav")

    fingerprint = analyzer.get_audio_fingerprint()

    print(f"\nResults:")
    print(f"  Fingerprint length: {len(fingerprint)} characters")
    print(f"  Fingerprint (first 32 chars): {fingerprint[:32]}...")

    # Verify consistency
    fingerprint2 = analyzer.get_audio_fingerprint()
    is_consistent = fingerprint == fingerprint2

    print(f"  Fingerprint is consistent: {is_consistent}")


def demo_live_pitch_detector():
    """Demo 7: Real-time pitch detection."""
    print("\n" + "=" * 70)
    print("DEMO 7: Real-Time Pitch Detection")
    print("=" * 70)

    print("\nSimulating real-time pitch detection...")
    print("  (Processing 440 Hz sine wave in chunks)")

    # Create detector
    detector = LivePitchDetector(sample_rate=44100, buffer_size=2048)

    # Generate 440 Hz test tone
    sr = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t)

    # Process in chunks
    chunk_size = 512
    detections = []

    print("\n  Processing chunks...")
    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[i : i + chunk_size]
        pitch_info = detector.process(chunk)

        if pitch_info:
            detections.append(pitch_info)
            if len(detections) <= 3:  # Show first 3
                note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
                note_name = note_names[pitch_info.midi_note % 12]
                octave = pitch_info.midi_note // 12 - 1
                print(
                    f"    Detected: {note_name}{octave} ({pitch_info.frequency:.1f} Hz, "
                    f"{pitch_info.cents_offset:+.0f} cents)"
                )

    print(f"\n  Total detections: {len(detections)}")
    if len(detections) > 0:
        avg_freq = np.mean([d.frequency for d in detections])
        print(f"  Average frequency: {avg_freq:.1f} Hz (expected: 440.0 Hz)")


def demo_comprehensive_analysis():
    """Demo 8: Comprehensive analysis workflow."""
    print("\n" + "=" * 70)
    print("DEMO 8: Comprehensive Analysis Workflow")
    print("=" * 70)

    print("\nPerforming comprehensive audio analysis...")
    analyzer = AudioAnalyzer("tests/amen.wav")

    # Beat detection
    print("\n  1. Beat Detection")
    beat_info = analyzer.detect_beats()
    print(f"     Tempo: {beat_info.tempo:.1f} BPM")

    # Key detection
    print("\n  2. Key Detection")
    key, mode = analyzer.detect_key()
    print(f"     Key: {key} {mode}")

    # Spectral analysis
    print("\n  3. Spectral Analysis")
    spectrum = analyzer.analyze_spectrum(time=0.5)
    print(f"     Centroid: {spectrum['centroid']:.1f} Hz")
    print(f"     Rolloff: {spectrum['rolloff']:.1f} Hz")

    # MFCC
    print("\n  4. MFCC Extraction")
    mfcc = analyzer.extract_mfcc()
    print(f"     Shape: {mfcc.shape}")

    # Fingerprint
    print("\n  5. Audio Fingerprinting")
    fingerprint = analyzer.get_audio_fingerprint()
    print(f"     Fingerprint: {fingerprint[:32]}...")

    print("\n  Analysis complete!")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("COREMUSIC AUDIO ANALYSIS DEMO")
    print("=" * 70)
    print("\nThis demo showcases audio analysis capabilities:")
    print("- Beat detection and tempo estimation")
    print("- Pitch detection and tracking")
    print("- Spectral analysis (FFT, features)")
    print("- MFCC extraction")
    print("- Key detection")
    print("- Audio fingerprinting")
    print("- Real-time pitch detection")

    if not NUMPY_AVAILABLE:
        print("\nERROR: NumPy is required for audio analysis.")
        print("Install with: pip install numpy")
        return

    if not SCIPY_AVAILABLE:
        print("\nERROR: SciPy is required for audio analysis.")
        print("Install with: pip install scipy")
        return

    try:
        demo_beat_detection()
        demo_pitch_detection()
        demo_spectral_analysis()
        demo_mfcc()
        demo_key_detection()
        demo_fingerprinting()
        demo_live_pitch_detector()
        demo_comprehensive_analysis()

        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("- AudioAnalyzer provides comprehensive offline analysis")
        print("- Beat detection estimates tempo and locates beats/downbeats")
        print("- Pitch detection tracks fundamental frequency over time")
        print("- Spectral analysis reveals frequency content and features")
        print("- MFCCs provide compact spectral representation")
        print("- Key detection identifies musical tonality")
        print("- Fingerprinting enables audio identification")
        print("- LivePitchDetector enables real-time pitch tracking")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
