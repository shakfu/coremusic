#!/usr/bin/env python3
"""Check Format Support."""

# --8<-- [start:example]
from coremusic.audio import AudioFile
from coremusic.exceptions import AudioFileError

def is_format_supported(filepath):
    """Check if audio format is supported."""
    try:
        with AudioFile(filepath) as audio:
            # Try to read format
            fmt = audio.format

            # Try to read data
            data, count = audio.read_packets(0, 1)

            return True, f"Supported: {fmt.format_id}"

    except AudioFileError as e:
        return False, f"Not supported: {e}"
    except Exception as e:
        return False, f"Error: {e}"

# Usage
supported, message = is_format_supported("audio.wav")
print(message)
# --8<-- [end:example]
