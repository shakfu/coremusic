"""Main CLI application using argparse."""

from __future__ import annotations

import argparse
import sys

VERSION = "0.1.11"


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="coremusic",
        description="CoreMusic - Python bindings for Apple CoreAudio.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {VERSION}"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # Register command modules
    from . import analyze, audio, convert, devices, midi, plugins, sequence

    audio.register(subparsers)
    devices.register(subparsers)
    plugins.register(subparsers)
    analyze.register(subparsers)
    convert.register(subparsers)
    midi.register(subparsers)
    sequence.register(subparsers)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Execute the command handler
    try:
        result = args.func(args)
        return int(result) if result is not None else 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        from ._utils import CLIError

        if isinstance(e, CLIError):
            print(f"Error: {e}", file=sys.stderr)
            return e.exit_code
        raise


if __name__ == "__main__":
    sys.exit(main())
