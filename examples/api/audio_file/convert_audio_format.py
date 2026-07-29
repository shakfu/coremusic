#!/usr/bin/env python3
"""Audio Format Conversion."""

# --8<-- [start:example]
from coremusic import capi
from coremusic.audio import AudioConverter, AudioFile, AudioFormat, ExtendedAudioFile


def convert_audio_format(input_path, output_path, target_format):
    """Convert audio file to different format."""
    # Open input file
    with AudioFile(input_path) as input_audio:
        # Create converter
        converter = AudioConverter(input_audio.format, target_format)

        # Read and convert. convert() handles depth and channel changes; a
        # change of sample rate needs convert_with_callback().
        data, count = input_audio.read_packets(0, input_audio.packet_count)
        converted = converter.convert(data)

    # Write to output file
    with ExtendedAudioFile.create(
        output_path, capi.fourchar_to_int("WAVE"), target_format
    ) as output_audio:
        output_audio.write(len(converted) // target_format.bytes_per_frame, converted)


convert_audio_format(
    "audio.wav",
    "converted.wav",
    AudioFormat.pcm(44100.0, channels=2, bits=32, is_float=True),
)
# --8<-- [end:example]
