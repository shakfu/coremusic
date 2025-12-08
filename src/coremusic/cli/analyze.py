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
