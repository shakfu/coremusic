"""Audio file commands."""

from __future__ import annotations

import argparse

from ._utils import require_file, EXIT_SUCCESS
from ._formatters import (
    format_duration,
    format_bytes,
    format_sample_rate,
    output_json,
)
from ._mappings import get_format_display, get_channel_display


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register audio commands."""
    parser = subparsers.add_parser("audio", help="Audio file operations")
    audio_sub = parser.add_subparsers(dest="audio_command", metavar="<subcommand>")

    # audio info
    info_parser = audio_sub.add_parser("info", help="Display audio file information")
    info_parser.add_argument("file", help="Audio file path")
    info_parser.set_defaults(func=cmd_info)

    # audio duration
    dur_parser = audio_sub.add_parser("duration", help="Get audio file duration")
    dur_parser.add_argument("file", help="Audio file path")
    dur_parser.add_argument(
        "--format", dest="fmt", choices=["seconds", "mm:ss", "samples"],
        default="seconds", help="Output format (default: seconds)"
    )
    dur_parser.set_defaults(func=cmd_duration)

    # audio metadata
    meta_parser = audio_sub.add_parser("metadata", help="Show audio file metadata/tags")
    meta_parser.add_argument("file", help="Audio file path")
    meta_parser.set_defaults(func=cmd_metadata)

    parser.set_defaults(func=lambda args: parser.print_help() or EXIT_SUCCESS)


def cmd_info(args: argparse.Namespace) -> int:
    """Display comprehensive audio file information."""
    import coremusic as cm

    path = require_file(args.file)

    with cm.AudioFile(str(path)) as audio_file:
        fmt = audio_file.format
        duration = audio_file.duration
        total_frames = int(duration * fmt.sample_rate)

        if args.json:
            data = {
                "file": str(path.absolute()),
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "format": {
                    "format_id": fmt.format_id,
                    "format_name": get_format_display(fmt.format_id),
                    "sample_rate": fmt.sample_rate,
                    "channels": fmt.channels_per_frame,
                    "bits_per_channel": fmt.bits_per_channel,
                    "bytes_per_frame": fmt.bytes_per_frame,
                    "format_flags": fmt.format_flags,
                },
                "duration_seconds": duration,
                "total_frames": total_frames,
            }
            output_json(data)
        else:
            print(f"File:        {path.name}")
            print(f"Path:        {path.absolute()}")
            print(f"Size:        {format_bytes(path.stat().st_size)}")
            print()
            print(f"Format:      {get_format_display(fmt.format_id)}")
            print(f"Sample Rate: {format_sample_rate(fmt.sample_rate)}")
            print(f"Channels:    {fmt.channels_per_frame} ({get_channel_display(fmt.channels_per_frame)})")
            print(f"Bit Depth:   {fmt.bits_per_channel}-bit")
            print()
            print(f"Duration:    {format_duration(duration)} ({duration:.3f}s)")
            print(f"Frames:      {total_frames:,}")

    return EXIT_SUCCESS


def cmd_duration(args: argparse.Namespace) -> int:
    """Get audio file duration."""
    import coremusic as cm

    require_file(args.file)

    with cm.AudioFile(args.file) as audio_file:
        dur = audio_file.duration
        sample_rate = audio_file.format.sample_rate

        if args.json:
            output_json({
                "duration_seconds": dur,
                "duration_samples": int(dur * sample_rate),
                "sample_rate": sample_rate,
            })
        elif args.fmt == "mm:ss":
            print(format_duration(dur))
        elif args.fmt == "samples":
            print(int(dur * sample_rate))
        else:
            print(f"{dur:.6f}")

    return EXIT_SUCCESS


def cmd_metadata(args: argparse.Namespace) -> int:
    """Show audio file metadata/tags."""
    import coremusic as cm
    from typing import Any, Dict

    path = require_file(args.file)

    metadata: Dict[str, Any] = {}

    with cm.AudioFile(str(path)) as audio_file:
        fmt = audio_file.format

        # Basic format info
        format_info = {
            "format_id": fmt.format_id,
            "format_name": get_format_display(fmt.format_id),
            "sample_rate": fmt.sample_rate,
            "channels": fmt.channels_per_frame,
            "bits_per_channel": fmt.bits_per_channel,
        }
        metadata["format"] = format_info
        duration = audio_file.duration
        metadata["duration_seconds"] = duration
        metadata["total_frames"] = int(duration * fmt.sample_rate)

        # Try to get info dictionary (may not be available for all formats)
        # Note: For some properties, audio_file_get_property might return a dict
        try:
            if hasattr(cm, 'AudioFileProperty'):
                import coremusic.capi as capi
                info_dict_prop = cm.AudioFileProperty.INFO_DICTIONARY
                info_data = capi.audio_file_get_property(audio_file.object_id, info_dict_prop)
                if info_data and isinstance(info_data, dict):  # type: ignore[unreachable]
                    metadata["tags"] = info_data  # type: ignore[unreachable]
        except Exception:
            # Info dictionary not available for this format
            pass

        # File info
        metadata["file"] = {
            "name": path.name,
            "path": str(path.absolute()),
            "size_bytes": path.stat().st_size,
        }

    if args.json:
        output_json(metadata)
    else:
        fmt_info = metadata["format"]
        print(f"File: {path.name}")
        print(f"Path: {path.absolute()}")
        print(f"Size: {format_bytes(path.stat().st_size)}")
        print()
        print(f"Format:      {get_format_display(str(fmt_info['format_id']))}")
        print(f"Sample Rate: {format_sample_rate(float(fmt_info['sample_rate']))}")
        print(f"Channels:    {fmt_info['channels']} ({get_channel_display(int(fmt_info['channels']))})")
        print(f"Bit Depth:   {fmt_info['bits_per_channel']}-bit")
        print()
        print(f"Duration:    {format_duration(float(metadata['duration_seconds']))}")
        print(f"Frames:      {metadata['total_frames']:,}")

        if "tags" in metadata and metadata["tags"]:
            print()
            print("Tags:")
            for key, value in metadata["tags"].items():
                print(f"  {key}: {value}")

    return EXIT_SUCCESS
