#!/usr/bin/env python3
"""Convert any input to a standard PCM format."""

# --8<-- [start:example]
from coremusic import capi
from coremusic.audio import AudioFormat, ExtendedAudioFile


def convert_to_standard_format(input_path, output_path):
    """Convert any audio to standard PCM format."""
    # Standard format: 44.1kHz, 16-bit, stereo PCM
    target_format = AudioFormat.pcm(sample_rate=44100.0, channels=2, bits=16)

    with ExtendedAudioFile(input_path) as input_file:
        # Set client format for automatic conversion
        input_file.client_format = target_format

        with ExtendedAudioFile.create(
            output_path, capi.fourchar_to_int("WAVE"), target_format
        ) as output_file:
            # Copy with automatic conversion
            chunk_size = 8192
            while True:
                data, count = input_file.read(chunk_size)
                if count == 0:
                    break
                output_file.write(count, data)


convert_to_standard_format("audio.wav", "standard.wav")
# --8<-- [end:example]
