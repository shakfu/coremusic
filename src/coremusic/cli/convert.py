"""Audio conversion commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._formatters import output_json
from ._mappings import FORMAT_NAMES, get_format_display, get_format_id
from ._utils import EXIT_SUCCESS, CLIError, print_help_default, require_file


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register convert commands."""
    parser = subparsers.add_parser(
        "convert",
        help="Convert audio files between formats",
    )
    convert_sub = parser.add_subparsers(dest="convert_command", metavar="<subcommand>")

    # convert file (single file conversion)
    file_parser = convert_sub.add_parser(
        "file",
        help="Convert a single audio file",
        description="Convert audio files with format, sample rate, and channel options.",
    )
    file_parser.add_argument("input", help="Input audio file")
    file_parser.add_argument("output", help="Output file path")
    file_parser.add_argument(
        "--format",
        "-f",
        dest="output_format",
        choices=list(FORMAT_NAMES.keys()),
        help="Output data format (e.g., wav, aac, alac)",
    )
    file_parser.add_argument(
        "--rate",
        "-r",
        dest="sample_rate",
        type=int,
        choices=[8000, 11025, 16000, 22050, 44100, 48000, 88200, 96000, 176400, 192000],
        help="Output sample rate in Hz",
    )
    file_parser.add_argument(
        "--channels",
        "-c",
        dest="channels",
        type=int,
        choices=[1, 2],
        help="Output channels (1=mono, 2=stereo)",
    )
    file_parser.add_argument(
        "--bits",
        "-b",
        dest="bit_depth",
        type=int,
        choices=[8, 16, 24, 32],
        help="Output bit depth",
    )
    file_parser.add_argument(
        "--quality",
        "-q",
        dest="quality",
        choices=["min", "low", "medium", "high", "max"],
        default="high",
        help="Conversion quality (default: high)",
    )
    file_parser.set_defaults(func=cmd_convert)

    # convert batch (batch conversion)
    batch_parser = convert_sub.add_parser(
        "batch",
        help="Batch convert multiple audio files",
        description="Convert multiple audio files matching a pattern.",
    )
    batch_parser.add_argument(
        "input_dir",
        help="Input directory containing audio files",
    )
    batch_parser.add_argument(
        "output_dir",
        help="Output directory for converted files",
    )
    batch_parser.add_argument(
        "--pattern",
        "-p",
        default="*.wav",
        help="File pattern to match (default: *.wav)",
    )
    batch_parser.add_argument(
        "--format",
        "-f",
        dest="output_format",
        choices=list(FORMAT_NAMES.keys()),
        default="wav",
        help="Output format (default: wav)",
    )
    batch_parser.add_argument(
        "--rate",
        "-r",
        dest="sample_rate",
        type=int,
        choices=[8000, 11025, 16000, 22050, 44100, 48000, 88200, 96000, 176400, 192000],
        help="Output sample rate in Hz",
    )
    batch_parser.add_argument(
        "--channels",
        "-c",
        dest="channels",
        type=int,
        choices=[1, 2],
        help="Output channels (1=mono, 2=stereo)",
    )
    batch_parser.add_argument(
        "--bits",
        "-b",
        dest="bit_depth",
        type=int,
        choices=[8, 16, 24, 32],
        help="Output bit depth",
    )
    batch_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively",
    )
    batch_parser.set_defaults(func=cmd_batch)

    # convert normalize
    normalize_parser = convert_sub.add_parser(
        "normalize",
        help="Normalize audio to target level",
        description="Normalize audio file to a target peak or RMS level.",
    )
    normalize_parser.add_argument("input", help="Input audio file")
    normalize_parser.add_argument("output", help="Output file path")
    normalize_parser.add_argument(
        "--target",
        "-t",
        type=float,
        default=-1.0,
        help="Target level in dB (default: -1.0 dBFS for peak)",
    )
    normalize_parser.add_argument(
        "--mode",
        "-m",
        choices=["peak", "rms"],
        default="peak",
        help="Normalization mode (default: peak)",
    )
    normalize_parser.set_defaults(func=cmd_normalize)

    # convert trim
    trim_parser = convert_sub.add_parser(
        "trim",
        help="Trim audio to time range",
        description="Extract a portion of an audio file by specifying start and end times.",
    )
    trim_parser.add_argument("input", help="Input audio file")
    trim_parser.add_argument("output", help="Output file path")
    trim_parser.add_argument(
        "--start",
        "-s",
        type=float,
        default=0.0,
        help="Start time in seconds (default: 0.0)",
    )
    trim_parser.add_argument(
        "--end",
        "-e",
        type=float,
        default=None,
        help="End time in seconds (default: end of file)",
    )
    trim_parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="Duration in seconds (alternative to --end)",
    )
    trim_parser.set_defaults(func=cmd_trim)

    parser.set_defaults(func=lambda args: print_help_default(parser))


def _infer_format_from_extension(path: Path) -> str:
    """Infer audio format from file extension."""
    ext = path.suffix.lower()
    extension_map = {
        ".wav": "lpcm",
        ".aif": "lpcm",
        ".aiff": "lpcm",
        ".m4a": "aac ",
        ".aac": "aac ",
        ".mp3": ".mp3",
        ".caf": "caff",
        ".flac": "flac",
    }
    return extension_map.get(ext, "lpcm")


def _get_file_type_for_extension(path: Path) -> int:
    """Resolve an AudioFileTypeID from an output extension.

    Delegates to the shared library resolver so the CLI and Python API agree on
    which formats are writable. Raises:
        CLIError: If the extension names a format coremusic cannot write. This
            replaces the previous behaviour of silently emitting WAV data under
            a mismatched extension.
    """
    from coremusic.audio.utilities import resolve_output_file_type

    try:
        return resolve_output_file_type(str(path))
    except ValueError as e:
        raise CLIError(str(e))


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert audio file."""
    from coremusic.audio import AudioFormat, ExtendedAudioFile
    from coremusic.audio.utilities import convert_audio_file

    input_path = require_file(args.input)
    output_path = Path(args.output)

    # Validate the output container up front so unsupported extensions raise a
    # clear error before any work is done.
    _get_file_type_for_extension(output_path)

    # Only PCM/lossless output is supported; reject a compressed --format override
    # instead of silently ignoring it.
    if getattr(args, "output_format", None):
        if get_format_id(args.output_format) != "lpcm":
            raise CLIError(
                f"Only PCM/lossless output is supported; "
                f"'{args.output_format}' output is not available."
            )

    # Read the source format for building the destination and for reporting.
    with ExtendedAudioFile(str(input_path)) as source:
        source.open()
        source_format = source.file_format

    # Build destination format
    dest_sample_rate = args.sample_rate or source_format.sample_rate
    dest_channels = args.channels or source_format.channels_per_frame
    dest_bits = args.bit_depth or source_format.bits_per_channel

    # Compute a complete PCM ASBD. A zero bytes_per_frame or missing PCM flags
    # produce an invalid description that CoreAudio rejects. AIFF stores
    # big-endian samples, so flag that container accordingly.
    is_aiff = output_path.suffix.lower() in (".aif", ".aiff")
    is_float = bool(source_format.format_flags & 1) and dest_bits >= 32
    # kAudioFormatFlagIsFloat=1, IsBigEndian=2, IsSignedInteger=4, IsPacked=8
    dest_flags = (1 if is_float else 4) | 8
    if is_aiff:
        dest_flags |= 2
    dest_bytes_per_frame = (dest_bits // 8) * dest_channels
    dest_format = AudioFormat(
        sample_rate=float(dest_sample_rate),
        format_id="lpcm",
        format_flags=dest_flags,
        bytes_per_packet=dest_bytes_per_frame,
        frames_per_packet=1,
        bytes_per_frame=dest_bytes_per_frame,
        channels_per_frame=dest_channels,
        bits_per_channel=dest_bits,
    )

    # Delegate the read/convert/write to the shared library helper, which uses
    # the callback-based converter API required for sample-rate changes.
    try:
        convert_audio_file(str(input_path), str(output_path), dest_format)
    except ValueError as e:
        raise CLIError(str(e))
    except Exception as e:
        raise CLIError(f"Conversion failed: {e}")

    # Output result
    if args.json:
        output_json(
            {
                "input": str(input_path.absolute()),
                "output": str(output_path.absolute()),
                "source_format": {
                    "sample_rate": source_format.sample_rate,
                    "channels": source_format.channels_per_frame,
                    "bits": source_format.bits_per_channel,
                    "format": get_format_display(source_format.format_id),
                },
                "dest_format": {
                    "sample_rate": dest_format.sample_rate,
                    "channels": dest_format.channels_per_frame,
                    "bits": dest_format.bits_per_channel,
                    "format": get_format_display(dest_format.format_id),
                },
            }
        )
    else:
        print(f"Converted: {input_path.name} -> {output_path.name}")
        print(
            f"  Format: {get_format_display(source_format.format_id)} -> {get_format_display(dest_format.format_id)}"
        )
        if dest_sample_rate != source_format.sample_rate:
            print(
                f"  Sample rate: {source_format.sample_rate:.0f} Hz -> {dest_sample_rate} Hz"
            )
        if dest_channels != source_format.channels_per_frame:
            ch_name = "Mono" if dest_channels == 1 else "Stereo"
            print(
                f"  Channels: {source_format.channels_per_frame} -> {dest_channels} ({ch_name})"
            )
        if dest_bits != source_format.bits_per_channel:
            print(
                f"  Bit depth: {source_format.bits_per_channel}-bit -> {dest_bits}-bit"
            )

    return EXIT_SUCCESS


def _get_output_extension(format_name: str) -> str:
    """Get file extension for format name."""
    ext_map = {
        "pcm": ".wav",
        "linear-pcm": ".wav",
        "wav": ".wav",
        "aiff": ".aiff",
        "aac": ".m4a",
        "mp3": ".mp3",
        "alac": ".m4a",
        "apple-lossless": ".m4a",
        "flac": ".flac",
    }
    return ext_map.get(format_name, ".wav")


def cmd_batch(args: argparse.Namespace) -> int:
    """Batch convert multiple audio files."""
    from coremusic.audio import AudioFormat, ExtendedAudioFile
    from coremusic.audio.utilities import convert_audio_file, resolve_output_file_type

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise CLIError(f"Input directory not found: {input_dir}")

    if not input_dir.is_dir():
        raise CLIError(f"Not a directory: {input_dir}")

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find matching files
    if args.recursive:
        input_files = list(input_dir.rglob(args.pattern))
    else:
        input_files = list(input_dir.glob(args.pattern))

    if not input_files:
        if not args.json:
            print(f"No files matching '{args.pattern}' found in {input_dir}")
        return EXIT_SUCCESS

    # Get output extension
    output_ext = _get_output_extension(args.output_format)

    # Only PCM/lossless containers are writable; fail fast before processing.
    if get_format_id(args.output_format) != "lpcm":
        raise CLIError(
            f"Only PCM/lossless output is supported; "
            f"'{args.output_format}' output is not available."
        )
    try:
        resolve_output_file_type(f"x{output_ext}")
    except ValueError as e:
        raise CLIError(str(e))

    results = []
    success_count = 0
    error_count = 0
    errors: list[tuple[str, str]] = []
    total_files = len(input_files)
    sorted_files = sorted(input_files)

    for i, input_path in enumerate(sorted_files):
        # Show progress bar (not in JSON mode)
        if not args.json:
            progress = (i + 1) / total_files
            bar_width = 30
            filled = int(bar_width * progress)
            bar = "=" * filled + "-" * (bar_width - filled)
            print(
                f"\r[{bar}] {i + 1}/{total_files} ({int(progress * 100)}%)",
                end="",
                flush=True,
            )

        # Determine output path
        relative_path = input_path.relative_to(input_dir)
        output_path = output_dir / relative_path.with_suffix(output_ext)

        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = {
            "input": str(input_path),
            "output": str(output_path),
            "status": "success",
        }

        try:
            with ExtendedAudioFile(str(input_path)) as source:
                source.open()
                source_format = source.file_format

            # Build a complete PCM destination format (see cmd_convert).
            dest_sample_rate = args.sample_rate or source_format.sample_rate
            dest_channels = args.channels or source_format.channels_per_frame
            dest_bits = args.bit_depth or source_format.bits_per_channel
            is_aiff = output_path.suffix.lower() in (".aif", ".aiff")
            is_float = bool(source_format.format_flags & 1) and dest_bits >= 32
            dest_flags = (1 if is_float else 4) | 8
            if is_aiff:
                dest_flags |= 2
            bytes_per_frame = (dest_bits // 8) * dest_channels
            dest_format = AudioFormat(
                sample_rate=float(dest_sample_rate),
                format_id="lpcm",
                format_flags=dest_flags,
                bytes_per_packet=bytes_per_frame,
                frames_per_packet=1,
                bytes_per_frame=bytes_per_frame,
                channels_per_frame=dest_channels,
                bits_per_channel=dest_bits,
            )

            convert_audio_file(str(input_path), str(output_path), dest_format)
            success_count += 1

        except Exception as e:
            error_count += 1
            result["status"] = "error"
            result["error"] = str(e)
            errors.append((input_path.name, str(e)))

        results.append(result)

    # Clear progress bar line
    if not args.json:
        print("\r" + " " * 60 + "\r", end="")

    if args.json:
        output_json(
            {
                "input_dir": str(input_dir.absolute()),
                "output_dir": str(output_dir.absolute()),
                "pattern": args.pattern,
                "format": args.output_format,
                "total": total_files,
                "success": success_count,
                "errors": error_count,
                "files": results,
            }
        )
    else:
        print(f"Converted {success_count}/{total_files} files")
        if errors:
            print(f"\nErrors ({error_count}):")
            for filename, error_msg in errors:
                print(f"  {filename}: {error_msg}")

    return EXIT_SUCCESS


def cmd_normalize(args: argparse.Namespace) -> int:
    """Normalize audio to target level."""
    import math

    import numpy as np

    from coremusic.audio import AudioFile, ExtendedAudioFile

    from ._utils import require_numpy

    require_numpy()

    input_path = require_file(args.input)
    output_path = Path(args.output)

    # Open source file
    with AudioFile(str(input_path)) as source:
        source_format = source.format
        audio_data = source.read_as_numpy()

        # Convert to float for processing
        if audio_data.dtype in [np.int16, np.int32]:
            max_val = np.iinfo(audio_data.dtype).max
            audio_float = audio_data.astype(np.float32) / max_val
        else:
            audio_float = audio_data.astype(np.float32)

        # Calculate current level
        if args.mode == "peak":
            current_level = np.max(np.abs(audio_float))
        else:  # rms
            current_level = np.sqrt(np.mean(audio_float**2))

        if current_level == 0:
            raise CLIError("Audio file is silent, cannot normalize")

        # Calculate gain
        target_linear = 10 ** (args.target / 20)
        gain = target_linear / current_level

        # Apply gain
        normalized = audio_float * gain

        # Clip to prevent overflow
        normalized = np.clip(normalized, -1.0, 1.0)

        # Convert back to original format
        if audio_data.dtype == np.int16:
            normalized_int = (normalized * 32767).astype(np.int16)
            output_bytes = normalized_int.tobytes()
        elif audio_data.dtype == np.int32:
            normalized_int = (normalized * 2147483647).astype(np.int32)
            output_bytes = normalized_int.tobytes()
        else:
            output_bytes = normalized.tobytes()

    # Write output file
    file_type = _get_file_type_for_extension(output_path)
    try:
        with ExtendedAudioFile.create(
            str(output_path), file_type, source_format
        ) as out_file:
            num_frames = len(output_bytes) // source_format.bytes_per_frame
            out_file.write(num_frames, output_bytes)
    except Exception as e:
        raise CLIError(f"Failed to write output file: {e}")

    # Calculate levels for output
    current_db = 20 * math.log10(current_level) if current_level > 0 else float("-inf")
    gain_db = 20 * math.log10(gain) if gain > 0 else 0

    if args.json:
        output_json(
            {
                "input": str(input_path.absolute()),
                "output": str(output_path.absolute()),
                "mode": args.mode,
                "original_level_db": current_db,
                "target_level_db": args.target,
                "gain_applied_db": gain_db,
            }
        )
    else:
        print(f"Normalized: {input_path.name} -> {output_path.name}")
        print(f"  Mode:     {args.mode}")
        print(f"  Original: {current_db:.1f} dB")
        print(f"  Target:   {args.target:.1f} dB")
        print(f"  Gain:     {gain_db:+.1f} dB")

    return EXIT_SUCCESS


def cmd_trim(args: argparse.Namespace) -> int:
    """Trim audio to time range."""

    from coremusic.audio import AudioFile, ExtendedAudioFile

    from ._utils import require_numpy

    require_numpy()

    input_path = require_file(args.input)
    output_path = Path(args.output)

    # Open source file and read all data
    with AudioFile(str(input_path)) as source:
        source_format = source.format
        duration = source.duration
        sample_rate = source_format.sample_rate
        audio_data = source.read_as_numpy()

    # Calculate start and end frames
    start_frame = int(args.start * sample_rate)

    if args.duration is not None:
        end_frame = start_frame + int(args.duration * sample_rate)
    elif args.end is not None:
        end_frame = int(args.end * sample_rate)
    else:
        end_frame = len(audio_data)

    # Validate range
    total_frames = len(audio_data)
    if start_frame < 0:
        start_frame = 0
    if end_frame > total_frames:
        end_frame = total_frames
    if start_frame >= end_frame:
        raise CLIError(
            f"Invalid time range: start ({args.start}s) >= end ({end_frame / sample_rate:.3f}s)"
        )

    # Extract the trimmed portion
    trimmed_data = audio_data[start_frame:end_frame]
    frames_read = len(trimmed_data)

    # Convert to bytes
    trimmed_bytes = trimmed_data.tobytes()

    # Calculate actual times
    actual_start = start_frame / sample_rate
    actual_end = end_frame / sample_rate
    actual_duration = actual_end - actual_start

    # Write output file
    file_type = _get_file_type_for_extension(output_path)
    try:
        with ExtendedAudioFile.create(
            str(output_path), file_type, source_format
        ) as out_file:
            out_file.write(frames_read, trimmed_bytes)
    except Exception as e:
        raise CLIError(f"Failed to write output file: {e}")

    if args.json:
        output_json(
            {
                "input": str(input_path.absolute()),
                "output": str(output_path.absolute()),
                "original_duration": duration,
                "start_time": actual_start,
                "end_time": actual_end,
                "trimmed_duration": actual_duration,
                "frames_written": frames_read,
            }
        )
    else:
        from ._formatters import format_duration

        print(f"Trimmed: {input_path.name} -> {output_path.name}")
        print(f"  Original: {format_duration(duration)}")
        print(
            f"  Range:    {format_duration(actual_start)} - {format_duration(actual_end)}"
        )
        print(f"  Duration: {format_duration(actual_duration)}")

    return EXIT_SUCCESS
