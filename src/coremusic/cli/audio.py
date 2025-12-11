"""Audio file commands."""

from __future__ import annotations

import argparse

from ._formatters import (format_bytes, format_duration, format_sample_rate,
                          output_json)
from ._mappings import get_channel_display, get_format_display
from ._utils import EXIT_SUCCESS, require_file


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

    # audio play
    play_parser = audio_sub.add_parser("play", help="Play audio file to default output")
    play_parser.add_argument("file", help="Audio file path")
    play_parser.add_argument("--loop", "-l", action="store_true",
                             help="Loop playback")
    play_parser.set_defaults(func=cmd_play)

    # audio record
    record_parser = audio_sub.add_parser("record", help="Record audio from input device")
    record_parser.add_argument("-o", "--output", required=True,
                               help="Output audio file path (WAV format)")
    record_parser.add_argument("--duration", "-d", type=float, default=None,
                               help="Recording duration in seconds (default: until Ctrl+C, max 300s)")
    record_parser.add_argument("--device", type=int, default=None,
                               help="Input device index (default: system default)")
    record_parser.add_argument("--sample-rate", "-r", type=int, default=44100,
                               help="Sample rate in Hz (default: 44100)")
    record_parser.add_argument("--channels", "-c", type=int, default=2,
                               choices=[1, 2], help="Number of channels (default: 2)")
    record_parser.set_defaults(func=cmd_record)

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
    from typing import Any, Dict

    import coremusic as cm

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


def cmd_play(args: argparse.Namespace) -> int:
    """Play audio file to default output device."""
    import signal
    import time

    import coremusic as cm
    from coremusic.capi import AudioPlayer

    from ._utils import CLIError

    path = require_file(args.file)

    # Get file info for display
    try:
        with cm.AudioFile(str(path)) as audio_file:
            duration = audio_file.duration
            fmt = audio_file.format
    except Exception as e:
        raise CLIError(f"Failed to open audio file: {e}")

    # Create and set up audio player
    try:
        player = AudioPlayer()
        player.load_file(str(path))
        player.setup_output()
    except Exception as e:
        raise CLIError(f"Failed to initialize audio player: {e}")

    # Set looping if requested
    player.set_looping(args.loop)

    # Stop flag for Ctrl+C
    stop_requested = False

    def signal_handler(sig: int, frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    original_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        if args.json:
            # Start playback silently for JSON output
            player.start()

            while player.is_playing() and not stop_requested:
                time.sleep(0.1)

            player.stop()

            output_json({
                "file": str(path.absolute()),
                "duration": duration,
                "sample_rate": fmt.sample_rate,
                "channels": fmt.channels_per_frame,
                "looped": args.loop,
                "stopped": stop_requested,
            })
        else:
            print(f"Playing: {path.name}")
            print(f"Duration: {format_duration(duration)}")
            print(f"Format:  {get_format_display(fmt.format_id)} {fmt.sample_rate:.0f}Hz {fmt.channels_per_frame}ch")
            if args.loop:
                print("Looping: Yes")
            print("Press Ctrl+C to stop...\n")

            player.start()

            # Show progress while playing
            try:
                while player.is_playing() and not stop_requested:
                    progress = player.get_progress()
                    elapsed = progress * duration
                    bar_width = 40
                    filled = int(bar_width * progress)
                    bar = "=" * filled + "-" * (bar_width - filled)
                    print(f"\r[{bar}] {format_duration(elapsed)} / {format_duration(duration)}", end="", flush=True)
                    time.sleep(0.1)
            except KeyboardInterrupt:
                stop_requested = True

            player.stop()

            if stop_requested:
                print("\n\nStopped.")
            else:
                print("\n\nFinished.")

    finally:
        signal.signal(signal.SIGINT, original_handler)

    return EXIT_SUCCESS


def cmd_record(args: argparse.Namespace) -> int:
    """Record audio from input device to file."""
    import signal
    from pathlib import Path

    from coremusic.capi import AudioRecorder

    from ._utils import CLIError

    output_path = Path(args.output)

    # Ensure parent directory exists
    if not output_path.parent.exists():
        raise CLIError(f"Output directory does not exist: {output_path.parent}")

    # Determine recording duration
    max_duration = 300.0  # Maximum 5 minutes
    if args.duration is not None:
        duration = min(args.duration, max_duration)
    else:
        duration = max_duration  # Default to max, will stop on Ctrl+C

    # Create recorder
    try:
        recorder = AudioRecorder(
            sample_rate=args.sample_rate,
            channels=args.channels
        )
        recorder.setup_input(duration)
    except Exception as e:
        raise CLIError(f"Failed to initialize audio recorder: {e}")

    # Stop flag for Ctrl+C
    stop_requested = False

    def signal_handler(sig: int, frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    original_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        if args.json:
            # Start recording silently for JSON output
            recorder.start()

            while recorder.is_recording() and not stop_requested:
                recorder.run_loop(0.1)

            recorder.stop()
            recorder.save_to_file(str(output_path))

            output_json({
                "output": str(output_path.absolute()),
                "duration": recorder.get_recorded_duration(),
                "sample_rate": recorder.sample_rate,
                "channels": recorder.channels,
                "frames": recorder.get_recorded_frames(),
                "stopped": stop_requested,
            })
        else:
            print(f"Recording to: {output_path.name}")
            print(f"Format:       WAV {args.sample_rate}Hz {args.channels}ch")
            if args.duration:
                print(f"Duration:     {format_duration(duration)} (max)")
            else:
                print(f"Duration:     Until Ctrl+C (max {format_duration(max_duration)})")
            print("Press Ctrl+C to stop...\n")

            recorder.start()

            # Show progress while recording
            # Must call run_loop() to process audio queue callbacks
            try:
                while recorder.is_recording() and not stop_requested:
                    # Run the CFRunLoop to receive audio data
                    recorder.run_loop(0.1)
                    recorded = recorder.get_recorded_duration()
                    progress = recorder.get_progress()
                    bar_width = 40
                    filled = int(bar_width * progress)
                    bar = "=" * filled + "-" * (bar_width - filled)
                    print(f"\r[{bar}] {format_duration(recorded)} / {format_duration(duration)}", end="", flush=True)
            except KeyboardInterrupt:
                stop_requested = True

            recorder.stop()

            # Save the recording
            try:
                recorder.save_to_file(str(output_path))
                recorded_duration = recorder.get_recorded_duration()

                if stop_requested:
                    print(f"\n\nStopped. Saved {format_duration(recorded_duration)} to {output_path.name}")
                else:
                    print(f"\n\nFinished. Saved {format_duration(recorded_duration)} to {output_path.name}")

                # Check for silent recording (permissions issue)
                if not recorder.has_audio_content():
                    print("\nWarning: Recording appears to be silent.")
                    print("This usually means microphone permission was not granted.")
                    print("Grant permission in: System Preferences > Security & Privacy > Privacy > Microphone")
            except Exception as e:
                raise CLIError(f"Failed to save recording: {e}")

    finally:
        signal.signal(signal.SIGINT, original_handler)

    return EXIT_SUCCESS
