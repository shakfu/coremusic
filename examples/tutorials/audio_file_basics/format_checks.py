#!/usr/bin/env python3
"""Classifying a file by its format."""

# --8<-- [start:detect]
from coremusic.audio import AudioFile


def detect_audio_format(filepath):
    """Detect and classify audio format."""
    with AudioFile(filepath) as audio:
        fmt = audio.format

        if fmt.format_id == "lpcm":
            return "Linear PCM (uncompressed)"
        elif fmt.format_id == "aac ":
            return "AAC (compressed)"
        elif fmt.format_id == ".mp3":
            return "MP3 (compressed)"
        elif fmt.format_id == "alac":
            return "Apple Lossless (compressed)"
        else:
            return f"Unknown format: {fmt.format_id}"


print(detect_audio_format("audio.wav"))  # Linear PCM (uncompressed)
# --8<-- [end:detect]

# --8<-- [start:properties]
from coremusic.audio import AudioFile


def check_format_properties(filepath):
    """Check various format properties."""
    with AudioFile(filepath) as audio:
        fmt = audio.format

        # AudioFormat answers the common questions itself
        print(f"PCM: {fmt.is_pcm}")
        print(f"Stereo: {fmt.is_stereo}")

        # Check if CD quality (44.1kHz, 16-bit stereo)
        is_cd_quality = (
            fmt.sample_rate == 44100.0 and fmt.bits_per_channel == 16 and fmt.is_stereo
        )
        print(f"CD Quality: {is_cd_quality}")


check_format_properties("audio.wav")
# --8<-- [end:properties]
