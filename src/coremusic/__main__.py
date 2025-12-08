"""Entry point for python3 -m coremusic."""

import sys


def _main() -> int:
    # Import here to avoid circular import with coremusic package
    from coremusic.cli import main
    return main()


if __name__ == "__main__":
    sys.exit(_main())
