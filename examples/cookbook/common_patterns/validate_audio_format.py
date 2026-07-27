#!/usr/bin/env python3
"""Format Detection and Validation."""

# --8<-- [start:example]
from coremusic.audio import AudioFile

def validate_audio_format(filepath, required_format=None):
    """Validate audio file format."""
    with AudioFile(filepath) as audio:
        fmt = audio.format

        # Basic validation
        if fmt.sample_rate <= 0:
            raise ValueError("Invalid sample rate")
        if fmt.channels_per_frame <= 0:
            raise ValueError("Invalid channel count")

        # Check against required format
        if required_format:
            if fmt.sample_rate != required_format.sample_rate:
                raise ValueError(
                    f"Sample rate mismatch: {fmt.sample_rate} != {required_format.sample_rate}"
                )
            if fmt.channels_per_frame != required_format.channels_per_frame:
                raise ValueError(
                    f"Channel mismatch: {fmt.channels_per_frame} != {required_format.channels_per_frame}"
                )

        return fmt
# --8<-- [end:example]
