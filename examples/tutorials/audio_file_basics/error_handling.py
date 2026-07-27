#!/usr/bin/env python3
"""Opening and validating a file defensively."""

# --8<-- [start:safe-open]
from pathlib import Path

from coremusic.audio import AudioFile
from coremusic.exceptions import AudioFileError


def safe_open_audio_file(filepath):
    """Safely open audio file with error handling."""
    # Check if file exists
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        audio = AudioFile(filepath)
        audio.open()
        return audio
    except AudioFileError as e:
        raise RuntimeError(f"Failed to open audio file: {e}")


# Use with error handling
try:
    audio = safe_open_audio_file("audio.wav")
    try:
        # Work with file
        print(f"Duration: {audio.duration}")
    finally:
        audio.close()
except FileNotFoundError as e:
    print(f"Error: {e}")
except RuntimeError as e:
    print(f"Error: {e}")
# --8<-- [end:safe-open]

# --8<-- [start:validate]
from coremusic.audio import AudioFile


def validate_audio_file(filepath):
    """Validate audio file can be opened and read."""
    try:
        with AudioFile(filepath) as audio:
            # Try to read first packet
            data, count = audio.read_packets(0, 1)

            if count == 0:
                return False, "File contains no audio data"

            # Check basic format validity
            fmt = audio.format
            if fmt.sample_rate <= 0:
                return False, "Invalid sample rate"

            if fmt.channels_per_frame <= 0:
                return False, "Invalid channel count"

            return True, "Valid audio file"

    except Exception as e:
        return False, f"Validation failed: {e}"


# Validate file
is_valid, message = validate_audio_file("audio.wav")
print(f"Valid: {is_valid}, Message: {message}")
# --8<-- [end:validate]
