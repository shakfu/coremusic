#!/usr/bin/env python3
"""Graceful Error Recovery."""

# --8<-- [start:example]
from pathlib import Path

from coremusic.audio import AudioFile
from coremusic.exceptions import AudioFileError


def safe_audio_operation(filepath):
    """Perform audio operation with comprehensive error handling."""
    # Pre-check
    if not Path(filepath).exists():
        return None, "File not found"

    try:
        with AudioFile(filepath) as audio:
            data, count = audio.read_packets(0, audio.packet_count)
            return data, None

    except AudioFileError as e:
        return None, f"Audio error: {e}"
    except MemoryError:
        return None, "File too large for memory"
    except Exception as e:
        return None, f"Unexpected error: {e}"


# Usage
data, error = safe_audio_operation("audio.wav")
if error:
    print(f"Failed: {error}")
else:
    print(f"Read {len(data)} bytes")
# --8<-- [end:example]
