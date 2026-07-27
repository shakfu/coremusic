#!/usr/bin/env python3
"""Audio Format Conversion."""

# --8<-- [start:example]
from coremusic.audio import AudioConverter, AudioFile

def convert_audio_format(input_path, output_path, target_format):
    """Convert audio file to different format."""
    # Open input file
    with AudioFile(input_path) as input_audio:
        # Create converter
        converter = AudioConverter(input_audio.format, target_format)

        # Read and convert
        data, count = input_audio.read_packets(0, input_audio.packet_count)
        converted_data = converter.convert(data, count)

        # Write to output file
        # (implementation depends on output requirements)
# --8<-- [end:example]
