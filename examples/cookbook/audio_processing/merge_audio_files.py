#!/usr/bin/env python3
"""Concatenate files end to end."""

# --8<-- [start:example]
import coremusic.capi as capi
from coremusic.audio import AudioFile, AudioFormat, ExtendedAudioFile


def merge_audio_files(input_files, output_path):
    """Merge multiple audio files sequentially.

    Every input must share the first file's format; convert first if they do
    not.
    """
    with AudioFile(str(input_files[0])) as first_file:
        source_format = first_file.format

    out_format = AudioFormat.pcm(
        source_format.sample_rate,
        channels=source_format.channels_per_frame,
        bits=source_format.bits_per_channel,
    )

    with ExtendedAudioFile.create(
        output_path, capi.fourchar_to_int('WAVE'), out_format
    ) as output_file:
        for input_path in input_files:
            with AudioFile(str(input_path)) as input_file:
                data, count = input_file.read_packets(0, input_file.packet_count)
                output_file.write(count, data)


# Usage
files = ["audio.wav", "input.wav", "drums.wav"]
merge_audio_files(files, "complete.wav")
# --8<-- [end:example]
