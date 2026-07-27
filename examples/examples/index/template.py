#!/usr/bin/env python3
"""Skeleton for a standalone example script."""

import sys

# The template is shown in the docs as the shape a contributed example takes;
# running it here with no arguments exercises the usage path.
sys.argv = [sys.argv[0], "audio.wav"]


# --8<-- [start:example]
#!/usr/bin/env python3
"""
Example: [Example Name]

Description: [What this example demonstrates]

Usage: python example_name.py [arguments]
"""

import sys

from coremusic.audio import AudioFile
from coremusic.exceptions import AudioFileError


def main():
    """Main function."""
    # Argument parsing
    if len(sys.argv) < 2:
        print("Usage: python example_name.py <audio_file>")
        sys.exit(1)

    filepath = sys.argv[1]

    # Example implementation
    try:
        with AudioFile(filepath) as audio:
            print(f"{filepath}: {audio.duration:.2f}s")

    except AudioFileError as e:
        print(f"Audio file error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
# --8<-- [end:example]
