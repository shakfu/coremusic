#!/usr/bin/env python3
"""Validate Audio File."""

# --8<-- [start:example]
from pathlib import Path

from coremusic.audio import AudioFile
from coremusic.exceptions import AudioFileError


def validate_audio_file(filepath, min_duration=0.1, max_duration=3600):
    """Validate audio file meets requirements."""
    errors = []

    # Check file exists
    if not Path(filepath).exists():
        errors.append(f"File not found: {filepath}")
        return False, errors

    try:
        with AudioFile(filepath) as audio:
            fmt = audio.format

            # Check duration
            if audio.duration < min_duration:
                errors.append(f"Duration too short: {audio.duration}s")

            if audio.duration > max_duration:
                errors.append(f"Duration too long: {audio.duration}s")

            # Check sample rate
            if fmt.sample_rate < 8000 or fmt.sample_rate > 192000:
                errors.append(f"Invalid sample rate: {fmt.sample_rate}Hz")

            # Check channels
            if fmt.channels_per_frame < 1 or fmt.channels_per_frame > 32:
                errors.append(f"Invalid channel count: {fmt.channels_per_frame}")

            # Check bit depth
            if fmt.bits_per_channel not in [8, 16, 24, 32]:
                errors.append(f"Unsupported bit depth: {fmt.bits_per_channel}")

            # Try to read first frame
            try:
                data, count = audio.read_packets(0, 1)
                if count == 0:
                    errors.append("File contains no audio data")
            except Exception as e:
                errors.append(f"Cannot read audio data: {e}")

    except AudioFileError as e:
        errors.append(f"Audio file error: {e}")
    except Exception as e:
        errors.append(f"Unexpected error: {e}")

    return len(errors) == 0, errors

# Usage
is_valid, errors = validate_audio_file("audio.wav")
if not is_valid:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
# --8<-- [end:example]
