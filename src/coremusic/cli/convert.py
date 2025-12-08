"""Audio conversion commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._utils import require_file, CLIError, EXIT_SUCCESS
from ._formatters import output_json
from ._mappings import get_format_id, get_format_display, FORMAT_NAMES


def register(subparsers: argparse._SubParsersAction) -> None:
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
        "--format", "-f",
        dest="output_format",
        choices=list(FORMAT_NAMES.keys()),
        help="Output format (e.g., wav, aac, alac, flac)",
    )
    file_parser.add_argument(
        "--rate", "-r",
        dest="sample_rate",
        type=int,
        choices=[8000, 11025, 16000, 22050, 44100, 48000, 88200, 96000, 176400, 192000],
        help="Output sample rate in Hz",
    )
    file_parser.add_argument(
        "--channels", "-c",
        dest="channels",
        type=int,
        choices=[1, 2],
        help="Output channels (1=mono, 2=stereo)",
    )
    file_parser.add_argument(
        "--bits", "-b",
        dest="bit_depth",
        type=int,
        choices=[8, 16, 24, 32],
        help="Output bit depth",
    )
    file_parser.add_argument(
        "--quality", "-q",
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
        "--pattern", "-p",
        default="*.wav",
        help="File pattern to match (default: *.wav)",
    )
    batch_parser.add_argument(
        "--format", "-f",
        dest="output_format",
        choices=list(FORMAT_NAMES.keys()),
        default="wav",
        help="Output format (default: wav)",
    )
    batch_parser.add_argument(
        "--rate", "-r",
        dest="sample_rate",
        type=int,
        choices=[8000, 11025, 16000, 22050, 44100, 48000, 88200, 96000, 176400, 192000],
        help="Output sample rate in Hz",
    )
    batch_parser.add_argument(
        "--channels", "-c",
        dest="channels",
        type=int,
        choices=[1, 2],
        help="Output channels (1=mono, 2=stereo)",
    )
    batch_parser.add_argument(
        "--bits", "-b",
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

    parser.set_defaults(func=lambda args: parser.print_help() or EXIT_SUCCESS)


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
    """Get AudioFileTypeID for file extension."""
    from coremusic.constants import AudioFileType

    ext = path.suffix.lower()
    type_map = {
        ".wav": AudioFileType.WAVE,
        ".aif": AudioFileType.AIFF,
        ".aiff": AudioFileType.AIFF,
        ".m4a": AudioFileType.M4A,
        ".aac": AudioFileType.AAC_ADTS,
        ".caf": AudioFileType.CAF,
    }
    return type_map.get(ext, AudioFileType.WAVE)


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert audio file."""
    import coremusic as cm

    input_path = require_file(args.input)
    output_path = Path(args.output)

    # Open source file using ExtendedAudioFile for reading
    with cm.ExtendedAudioFile(str(input_path)) as source:
        source.open()
        source_format = source.file_format

        # Determine output format
        if args.output_format:
            dest_format_id = get_format_id(args.output_format)
        else:
            dest_format_id = _infer_format_from_extension(output_path)

        # Build destination format
        dest_sample_rate = args.sample_rate or source_format.sample_rate
        dest_channels = args.channels or source_format.channels_per_frame
        dest_bits = args.bit_depth or source_format.bits_per_channel

        # Create destination AudioFormat
        dest_format = cm.AudioFormat(
            sample_rate=float(dest_sample_rate),
            format_id=dest_format_id,
            channels_per_frame=dest_channels,
            bits_per_channel=dest_bits,
        )

        # Get file type for output
        file_type = _get_file_type_for_extension(output_path)

        # Read all source data
        frame_count = source.frame_count
        source_data, frames_read = source.read(frame_count)

        # Check if conversion is needed
        needs_conversion = (
            dest_format.sample_rate != source_format.sample_rate or
            dest_format.channels_per_frame != source_format.channels_per_frame or
            dest_format.bits_per_channel != source_format.bits_per_channel
        )

        if needs_conversion:
            # Use AudioConverter for format conversion
            try:
                with cm.AudioConverter(source_format, dest_format) as converter:
                    converted_data = converter.convert(source_data)
            except Exception as e:
                raise CLIError(f"Conversion failed: {e}")
        else:
            converted_data = source_data

        # Write output file
        try:
            with cm.ExtendedAudioFile.create(str(output_path), file_type, dest_format) as out_file:
                num_frames = len(converted_data) // dest_format.bytes_per_frame
                out_file.write(num_frames, converted_data)
        except Exception as e:
            raise CLIError(f"Failed to write output file: {e}")

    # Output result
    if args.json:
        output_json({
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
        })
    else:
        print(f"Converted: {input_path.name} -> {output_path.name}")
        print(f"  Format: {get_format_display(source_format.format_id)} -> {get_format_display(dest_format.format_id)}")
        if dest_sample_rate != source_format.sample_rate:
            print(f"  Sample rate: {source_format.sample_rate:.0f} Hz -> {dest_sample_rate} Hz")
        if dest_channels != source_format.channels_per_frame:
            ch_name = "Mono" if dest_channels == 1 else "Stereo"
            print(f"  Channels: {source_format.channels_per_frame} -> {dest_channels} ({ch_name})")
        if dest_bits != source_format.bits_per_channel:
            print(f"  Bit depth: {source_format.bits_per_channel}-bit -> {dest_bits}-bit")

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
    import coremusic as cm

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
    dest_format_id = get_format_id(args.output_format)

    results = []
    success_count = 0
    error_count = 0

    if not args.json:
        print(f"Converting {len(input_files)} files...")
        print()

    for input_path in sorted(input_files):
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
            with cm.ExtendedAudioFile(str(input_path)) as source:
                source.open()
                source_format = source.file_format

                # Build destination format
                dest_sample_rate = args.sample_rate or source_format.sample_rate
                dest_channels = args.channels or source_format.channels_per_frame
                dest_bits = args.bit_depth or source_format.bits_per_channel

                dest_format = cm.AudioFormat(
                    sample_rate=float(dest_sample_rate),
                    format_id=dest_format_id,
                    channels_per_frame=dest_channels,
                    bits_per_channel=dest_bits,
                )

                file_type = _get_file_type_for_extension(output_path)
                frame_count = source.frame_count
                source_data, _ = source.read(frame_count)

                # Check if conversion is needed
                needs_conversion = (
                    dest_format.sample_rate != source_format.sample_rate or
                    dest_format.channels_per_frame != source_format.channels_per_frame or
                    dest_format.bits_per_channel != source_format.bits_per_channel
                )

                if needs_conversion:
                    with cm.AudioConverter(source_format, dest_format) as converter:
                        converted_data = converter.convert(source_data)
                else:
                    converted_data = source_data

                with cm.ExtendedAudioFile.create(str(output_path), file_type, dest_format) as out_file:
                    num_frames = len(converted_data) // dest_format.bytes_per_frame
                    out_file.write(num_frames, converted_data)

            success_count += 1
            if not args.json:
                print(f"  OK: {input_path.name} -> {output_path.name}")

        except Exception as e:
            error_count += 1
            result["status"] = "error"
            result["error"] = str(e)
            if not args.json:
                print(f"  FAIL: {input_path.name} - {e}")

        results.append(result)

    if args.json:
        output_json({
            "input_dir": str(input_dir.absolute()),
            "output_dir": str(output_dir.absolute()),
            "pattern": args.pattern,
            "format": args.output_format,
            "total": len(input_files),
            "success": success_count,
            "errors": error_count,
            "files": results,
        })
    else:
        print()
        print(f"Converted {success_count}/{len(input_files)} files")
        if error_count > 0:
            print(f"Errors: {error_count}")

    return EXIT_SUCCESS
