#!/usr/bin/env python3
"""Audio Format Presets."""

# --8<-- [start:example]
from coremusic.audio import AudioFormat


class AudioFormats:
    """Common audio format presets."""

    CD_QUALITY = AudioFormat(
        sample_rate=44100.0,
        format_id='lpcm',
        format_flags=0x0C,
        channels_per_frame=2,
        bits_per_channel=16,
        bytes_per_frame=4,
        frames_per_packet=1,
        bytes_per_packet=4
    )

    DVD_QUALITY = AudioFormat(
        sample_rate=48000.0,
        format_id='lpcm',
        format_flags=0x0C,
        channels_per_frame=2,
        bits_per_channel=24,
        bytes_per_frame=6,
        frames_per_packet=1,
        bytes_per_packet=6
    )

    HIRES_AUDIO = AudioFormat(
        sample_rate=96000.0,
        format_id='lpcm',
        format_flags=0x0C,
        channels_per_frame=2,
        bits_per_channel=24,
        bytes_per_frame=6,
        frames_per_packet=1,
        bytes_per_packet=6
    )

    FLOAT32_STEREO = AudioFormat(
        sample_rate=44100.0,
        format_id='lpcm',
        format_flags=0x09,  # Float, packed
        channels_per_frame=2,
        bits_per_channel=32,
        bytes_per_frame=8,
        frames_per_packet=1,
        bytes_per_packet=8
    )

# Usage
format = AudioFormats.CD_QUALITY
# --8<-- [end:example]
