#!/usr/bin/env python3
"""Change a file's sample rate."""

# --8<-- [start:example]
from coremusic.audio import AudioFormat, convert_audio_file


def resample_audio(input_path, output_path, target_sample_rate=48000.0):
    """Resample audio with automatic conversion"""
    out_format = AudioFormat.pcm(
        sample_rate=target_sample_rate,
        channels=2,
        bits=16,
    )
    convert_audio_file(input_path, output_path, out_format)


resample_audio("input.wav", "resampled.wav", target_sample_rate=48000.0)
# --8<-- [end:example]

# --8<-- [start:manual]
from coremusic import capi
from coremusic.audio import AudioFormat, ExtendedAudioFile


def resample_manually(input_path, output_path, target_sample_rate=48000.0):
    """Resample by setting a client format and copying block by block.

    ExtendedAudioFile converts to the client format as it reads, so the copy
    loop below is doing the resampling.
    """
    with ExtendedAudioFile(input_path) as input_file:
        in_format = input_file.file_format

        out_format = AudioFormat.pcm(
            sample_rate=target_sample_rate,
            channels=in_format.channels_per_frame,
            bits=in_format.bits_per_channel,
        )

        # Set client format for automatic conversion
        input_file.client_format = out_format

        with ExtendedAudioFile.create(
            output_path, capi.fourchar_to_int("WAVE"), out_format
        ) as output_file:
            chunk_size = 8192
            while True:
                data, count = input_file.read(chunk_size)
                if count == 0:
                    break
                output_file.write(count, data)


resample_manually("input.wav", "resampled_manual.wav")
# --8<-- [end:manual]
