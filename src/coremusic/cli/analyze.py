"""Audio analysis commands."""

from __future__ import annotations

import argparse
import math

from ._formatters import format_db, format_duration, output_json
from ._utils import EXIT_SUCCESS, require_file, require_numpy, require_scipy


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register analyze commands."""
    parser = subparsers.add_parser("analyze", help="Audio analysis and feature extraction")
    analyze_sub = parser.add_subparsers(dest="analyze_command", metavar="<subcommand>")

    # analyze peak
    peak_parser = analyze_sub.add_parser("peak", help="Calculate peak amplitude")
    peak_parser.add_argument("file", help="Audio file path")
    peak_parser.add_argument("--db", action="store_true", help="Output in decibels")
    peak_parser.set_defaults(func=cmd_peak)

    # analyze rms
    rms_parser = analyze_sub.add_parser("rms", help="Calculate RMS level")
    rms_parser.add_argument("file", help="Audio file path")
    rms_parser.add_argument("--db", action="store_true", help="Output in decibels")
    rms_parser.set_defaults(func=cmd_rms)

    # analyze levels
    levels_parser = analyze_sub.add_parser("levels", help="Show both peak and RMS levels")
    levels_parser.add_argument("file", help="Audio file path")
    levels_parser.set_defaults(func=cmd_levels)

    # analyze silence
    silence_parser = analyze_sub.add_parser("silence", help="Detect silence regions")
    silence_parser.add_argument("file", help="Audio file path")
    silence_parser.add_argument(
        "--threshold", "-t", type=float, default=-40.0,
        help="Silence threshold in dB (default: -40)"
    )
    silence_parser.add_argument(
        "--min-duration", "-d", type=float, default=0.5,
        help="Minimum silence duration in seconds (default: 0.5)"
    )
    silence_parser.set_defaults(func=cmd_silence)

    # analyze tempo
    tempo_parser = analyze_sub.add_parser("tempo", help="Detect tempo/BPM")
    tempo_parser.add_argument("file", help="Audio file path")
    tempo_parser.set_defaults(func=cmd_tempo)

    # analyze spectrum
    spectrum_parser = analyze_sub.add_parser("spectrum", help="Analyze frequency spectrum")
    spectrum_parser.add_argument("file", help="Audio file path")
    spectrum_parser.add_argument(
        "--time", "-t", type=float, default=None,
        help="Time position in seconds (default: middle of file)"
    )
    spectrum_parser.add_argument(
        "--peaks", "-p", type=int, default=5,
        help="Number of spectral peaks to show (default: 5)"
    )
    spectrum_parser.set_defaults(func=cmd_spectrum)

    # analyze key
    key_parser = analyze_sub.add_parser("key", help="Detect musical key")
    key_parser.add_argument("file", help="Audio file path")
    key_parser.set_defaults(func=cmd_key)

    # analyze mfcc
    mfcc_parser = analyze_sub.add_parser("mfcc", help="Extract MFCC features")
    mfcc_parser.add_argument("file", help="Audio file path")
    mfcc_parser.add_argument(
        "--coefficients", "-n", type=int, default=13,
        help="Number of MFCC coefficients (default: 13)"
    )
    mfcc_parser.add_argument(
        "--time", "-t", type=float, default=None,
        help="Time position in seconds (default: middle of file)"
    )
    mfcc_parser.set_defaults(func=cmd_mfcc)

    # analyze loudness
    loudness_parser = analyze_sub.add_parser("loudness", help="LUFS loudness measurement")
    loudness_parser.add_argument("file", help="Audio file path")
    loudness_parser.set_defaults(func=cmd_loudness)

    # analyze onsets
    onsets_parser = analyze_sub.add_parser("onsets", help="Detect note/transient onsets")
    onsets_parser.add_argument("file", help="Audio file path")
    onsets_parser.add_argument(
        "--threshold", "-t", type=float, default=0.3,
        help="Onset detection threshold (0-1, default: 0.3)"
    )
    onsets_parser.add_argument(
        "--min-gap", "-g", type=float, default=0.05,
        help="Minimum gap between onsets in seconds (default: 0.05)"
    )
    onsets_parser.set_defaults(func=cmd_onsets)

    parser.set_defaults(func=lambda args: parser.print_help() or EXIT_SUCCESS)


def cmd_peak(args: argparse.Namespace) -> int:
    """Calculate peak amplitude."""
    require_numpy()
    require_file(args.file)

    from coremusic.audio.analysis import AudioAnalyzer

    # Static method that takes file path
    peak_value = AudioAnalyzer.get_peak_amplitude(args.file)

    if args.json:
        db_value = 20 * math.log10(peak_value) if peak_value > 0 else float("-inf")
        output_json({
            "peak_linear": peak_value,
            "peak_db": db_value,
        })
    elif args.db:
        print(format_db(peak_value))
    else:
        print(f"{peak_value:.6f}")

    return EXIT_SUCCESS


def cmd_rms(args: argparse.Namespace) -> int:
    """Calculate RMS level."""
    require_numpy()
    require_file(args.file)

    from coremusic.audio.analysis import AudioAnalyzer

    # Static method that takes file path
    rms_value = AudioAnalyzer.calculate_rms(args.file)

    if args.json:
        db_value = 20 * math.log10(rms_value) if rms_value > 0 else float("-inf")
        output_json({
            "rms_linear": rms_value,
            "rms_db": db_value,
        })
    elif args.db:
        print(format_db(rms_value))
    else:
        print(f"{rms_value:.6f}")

    return EXIT_SUCCESS


def cmd_levels(args: argparse.Namespace) -> int:
    """Show both peak and RMS levels."""
    require_numpy()
    require_file(args.file)

    from coremusic.audio.analysis import AudioAnalyzer

    # Static methods that take file path
    peak_value = AudioAnalyzer.get_peak_amplitude(args.file)
    rms_value = AudioAnalyzer.calculate_rms(args.file)

    peak_db = 20 * math.log10(peak_value) if peak_value > 0 else float("-inf")
    rms_db = 20 * math.log10(rms_value) if rms_value > 0 else float("-inf")

    if args.json:
        output_json({
            "peak_linear": peak_value,
            "peak_db": peak_db,
            "rms_linear": rms_value,
            "rms_db": rms_db,
        })
    else:
        print(f"Peak: {peak_db:.1f} dB ({peak_value:.6f})")
        print(f"RMS:  {rms_db:.1f} dB ({rms_value:.6f})")

    return EXIT_SUCCESS


def cmd_silence(args: argparse.Namespace) -> int:
    """Detect silence regions in audio."""
    require_numpy()
    require_file(args.file)

    from coremusic.audio.analysis import AudioAnalyzer

    silence_regions = AudioAnalyzer.detect_silence(
        args.file,
        threshold_db=args.threshold,
        min_duration=args.min_duration,
    )

    if args.json:
        output_json({
            "threshold_db": args.threshold,
            "min_duration": args.min_duration,
            "regions": [
                {"start": start, "end": end, "duration": end - start}
                for start, end in silence_regions
            ],
            "total_regions": len(silence_regions),
        })
    else:
        if not silence_regions:
            print(f"No silence regions found (threshold: {args.threshold} dB, min duration: {args.min_duration}s)")
        else:
            print(f"Found {len(silence_regions)} silence regions:\n")
            for i, (start, end) in enumerate(silence_regions, 1):
                duration = end - start
                print(f"  {i}. {format_duration(start)} - {format_duration(end)} ({duration:.2f}s)")

    return EXIT_SUCCESS


def cmd_tempo(args: argparse.Namespace) -> int:
    """Detect tempo/BPM."""
    require_numpy()
    require_scipy()
    require_file(args.file)

    from coremusic.audio.analysis import AudioAnalyzer

    analyzer = AudioAnalyzer(args.file)
    beat_info = analyzer.detect_beats()

    if args.json:
        output_json({
            "tempo_bpm": beat_info.tempo,
            "confidence": beat_info.confidence,
            "beat_count": len(beat_info.beats),
            "downbeat_count": len(beat_info.downbeats),
        })
    else:
        print(f"Tempo: {beat_info.tempo:.1f} BPM")
        print(f"Confidence: {beat_info.confidence * 100:.0f}%")
        print(f"Beats detected: {len(beat_info.beats)}")
        if beat_info.downbeats:
            print(f"Downbeats: {len(beat_info.downbeats)}")

    return EXIT_SUCCESS


def cmd_spectrum(args: argparse.Namespace) -> int:
    """Analyze frequency spectrum."""
    require_numpy()
    require_scipy()
    require_file(args.file)

    import coremusic as cm
    from coremusic.audio.analysis import AudioAnalyzer

    # Get duration to determine analysis time
    with cm.AudioFile(args.file) as af:
        duration = af.duration

    # Default to middle of file
    analysis_time = args.time if args.time is not None else duration / 2

    analyzer = AudioAnalyzer(args.file)
    spectrum = analyzer.analyze_spectrum(time=analysis_time)

    # Limit peaks to requested number
    peaks = spectrum["peaks"][:args.peaks] if spectrum["peaks"] else []

    if args.json:
        output_json({
            "time": analysis_time,
            "centroid_hz": spectrum["centroid"],
            "rolloff_hz": spectrum["rolloff"],
            "peaks": [
                {"frequency_hz": freq, "magnitude": mag}
                for freq, mag in peaks
            ],
        })
    else:
        print(f"Spectrum analysis at {format_duration(analysis_time)}:\n")
        print(f"  Spectral centroid: {spectrum['centroid']:.1f} Hz")
        print(f"  Spectral rolloff:  {spectrum['rolloff']:.1f} Hz")
        if peaks:
            print(f"\n  Top {len(peaks)} frequency peaks:")
            for i, (freq, mag) in enumerate(peaks, 1):
                print(f"    {i}. {freq:.1f} Hz (magnitude: {mag:.4f})")

    return EXIT_SUCCESS


def cmd_key(args: argparse.Namespace) -> int:
    """Detect musical key."""
    require_numpy()
    require_scipy()
    require_file(args.file)

    from coremusic.audio.analysis import AudioAnalyzer

    analyzer = AudioAnalyzer(args.file)
    key, mode = analyzer.detect_key()

    if args.json:
        output_json({
            "key": key,
            "mode": mode,
            "full_name": f"{key} {mode}",
        })
    else:
        print(f"Key: {key} {mode}")

    return EXIT_SUCCESS


def cmd_mfcc(args: argparse.Namespace) -> int:
    """Extract MFCC features."""
    require_numpy()
    require_scipy()
    require_file(args.file)

    import coremusic as cm
    from coremusic.audio.analysis import AudioAnalyzer

    # Get duration to determine analysis time
    with cm.AudioFile(args.file) as af:
        duration = af.duration

    # Default to middle of file
    analysis_time = args.time if args.time is not None else duration / 2

    analyzer = AudioAnalyzer(args.file)
    # extract_mfcc returns an NDArray (n_mfcc x n_frames)
    mfcc_matrix = analyzer.extract_mfcc(n_mfcc=args.coefficients)

    # Get MFCC values at the closest time frame
    # Calculate which frame corresponds to the analysis time
    hop_length = 512
    sample_rate: float = 44100.0
    if hasattr(analyzer, '_sample_rate') and analyzer._sample_rate is not None:
        sample_rate = analyzer._sample_rate
    frame_index = int(analysis_time * sample_rate / hop_length)
    frame_index = min(frame_index, mfcc_matrix.shape[1] - 1)  # Clamp to valid range

    # Extract MFCC values at that frame
    mfcc_values = mfcc_matrix[:, frame_index]

    if args.json:
        output_json({
            "time": analysis_time,
            "coefficients": args.coefficients,
            "mfcc": [float(v) for v in mfcc_values],
        })
    else:
        print(f"MFCC analysis at {format_duration(analysis_time)}:\n")
        print(f"  Coefficients: {args.coefficients}")
        print()
        print("  MFCC values:")
        for i, val in enumerate(mfcc_values):
            print(f"    C{i}: {val:8.2f}")

    return EXIT_SUCCESS


def cmd_loudness(args: argparse.Namespace) -> int:
    """Measure integrated LUFS loudness."""
    require_numpy()
    require_scipy()
    require_file(args.file)

    import numpy as np

    import coremusic as cm

    # Open audio file and read data
    with cm.AudioFile(args.file) as af:
        fmt = af.format
        sample_rate = fmt.sample_rate
        duration = af.duration
        audio_data = af.read_as_numpy()

    # Convert to float and normalize
    if audio_data.dtype in [np.int16, np.int32]:
        max_val = np.iinfo(audio_data.dtype).max
        audio_float = audio_data.astype(np.float64) / max_val
    else:
        audio_float = audio_data.astype(np.float64)

    # Convert to mono if stereo (average channels)
    if audio_float.ndim == 2:
        # For LUFS, we need to process stereo properly
        # but for simplicity, we'll use mono average first
        audio_mono = np.mean(audio_float, axis=1) if audio_float.shape[1] == 2 else audio_float[:, 0]
    else:
        audio_mono = audio_float

    # K-weighting filter coefficients (simplified version)
    # Stage 1: High-shelf filter
    # Stage 2: High-pass filter
    # For a proper LUFS implementation, these should be exact
    # Here we use a simplified RMS-based approximation

    # Apply K-weighting approximation using a high-pass filter
    from scipy import signal as sp_signal

    # High-pass filter at 38 Hz (simplified K-weighting)
    b, a = sp_signal.butter(2, 38.0 / (sample_rate / 2), btype='high')
    filtered = sp_signal.filtfilt(b, a, audio_mono)

    # Calculate mean square
    mean_square = np.mean(filtered ** 2)

    # Calculate LUFS (approximation)
    # LUFS = -0.691 + 10 * log10(mean_square)
    if mean_square > 0:
        lufs = -0.691 + 10 * np.log10(mean_square)
    else:
        lufs = float("-inf")

    # Also calculate simple RMS for comparison
    rms = np.sqrt(np.mean(audio_mono ** 2))
    rms_db = 20 * np.log10(rms) if rms > 0 else float("-inf")

    # Peak level
    peak = np.max(np.abs(audio_mono))
    peak_db = 20 * np.log10(peak) if peak > 0 else float("-inf")

    # Loudness range (simplified - difference between loud and quiet sections)
    # Divide into blocks and calculate variance
    block_size = int(sample_rate * 0.4)  # 400ms blocks
    num_blocks = len(filtered) // block_size
    if num_blocks > 1:
        block_loudness = []
        for i in range(num_blocks):
            block = filtered[i * block_size:(i + 1) * block_size]
            block_ms = np.mean(block ** 2)
            if block_ms > 0:
                block_loudness.append(-0.691 + 10 * np.log10(block_ms))
        if block_loudness:
            loudness_range = max(block_loudness) - min(block_loudness)
        else:
            loudness_range = 0.0
    else:
        loudness_range = 0.0

    if args.json:
        output_json({
            "integrated_lufs": lufs,
            "loudness_range_lu": loudness_range,
            "peak_db": peak_db,
            "rms_db": rms_db,
            "duration_seconds": duration,
        })
    else:
        print(f"Loudness analysis: {args.file}")
        print()
        print(f"  Integrated LUFS: {lufs:.1f} LUFS")
        print(f"  Loudness Range:  {loudness_range:.1f} LU")
        print(f"  Peak:            {peak_db:.1f} dBFS")
        print(f"  RMS:             {rms_db:.1f} dB")

    return EXIT_SUCCESS


def cmd_onsets(args: argparse.Namespace) -> int:
    """Detect note/transient onsets in audio."""
    require_numpy()
    require_scipy()
    require_file(args.file)

    import numpy as np
    from scipy import signal as sp_signal

    import coremusic as cm

    # Open audio file and read data
    with cm.AudioFile(args.file) as af:
        fmt = af.format
        sample_rate = fmt.sample_rate
        duration = af.duration
        audio_data = af.read_as_numpy()

    # Convert to float and normalize
    if audio_data.dtype in [np.int16, np.int32]:
        max_val = np.iinfo(audio_data.dtype).max
        audio_float = audio_data.astype(np.float32) / max_val
    else:
        audio_float = audio_data.astype(np.float32)

    # Convert to mono
    if audio_float.ndim == 2:
        audio_mono = np.mean(audio_float, axis=1)
    else:
        audio_mono = audio_float

    # Compute spectral flux for onset detection
    hop_length = 512
    n_fft = 2048

    # Compute STFT
    f, t, Zxx = sp_signal.stft(
        audio_mono, sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length
    )

    # Spectral flux (difference between consecutive frames)
    mag = np.abs(Zxx)
    flux = np.sum(np.maximum(0, mag[:, 1:] - mag[:, :-1]), axis=0)

    # Normalize flux
    if np.max(flux) > 0:
        flux_norm = flux / np.max(flux)
    else:
        flux_norm = flux

    # Calculate adaptive threshold
    mean_flux = np.mean(flux_norm)
    std_flux = np.std(flux_norm)
    adaptive_threshold = mean_flux + args.threshold * std_flux

    # Find peaks above threshold with minimum distance
    min_distance = int(args.min_gap * sample_rate / hop_length)
    peak_indices, properties = sp_signal.find_peaks(
        flux_norm,
        height=adaptive_threshold,
        distance=max(1, min_distance)
    )

    # Convert to time
    onset_times = t[1:][peak_indices]
    onset_strengths = properties['peak_heights']

    # Format output
    onsets = [
        {"time": float(t), "strength": float(s)}
        for t, s in zip(onset_times, onset_strengths)
    ]

    if args.json:
        output_json({
            "file": args.file,
            "duration_seconds": duration,
            "onset_count": len(onsets),
            "threshold": args.threshold,
            "min_gap": args.min_gap,
            "onsets": onsets,
        })
    else:
        print(f"Onset detection: {args.file}")
        print(f"Duration: {format_duration(duration)}")
        print(f"Onsets found: {len(onsets)}")
        print()

        if onsets:
            print("Onset times:")
            for i, onset in enumerate(onsets[:20]):  # Limit display
                print(f"  {i+1:3d}. {format_duration(onset['time'])} (strength: {onset['strength']:.3f})")

            if len(onsets) > 20:
                print(f"  ... and {len(onsets) - 20} more")

    return EXIT_SUCCESS
