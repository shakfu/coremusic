#!/usr/bin/env python3
"""Multiple Resources Pattern."""

# --8<-- [start:example]
from contextlib import ExitStack

from coremusic import capi
from coremusic.audio import AudioFile, AudioFormat, ExtendedAudioFile


def process_multiple_files(input_files, output_path):
    """Process multiple input files safely.

    ExitStack closes whatever was opened, however the block is left, without
    nesting one `with` per file.
    """
    with AudioFile(input_files[0]) as first:
        out_format = AudioFormat.pcm(
            first.format.sample_rate,
            channels=first.format.channels_per_frame,
            bits=first.format.bits_per_channel,
        )

    with ExitStack() as stack:
        # Open all input files
        inputs = [stack.enter_context(AudioFile(f)) for f in input_files]

        # Open output file
        output = stack.enter_context(
            ExtendedAudioFile.create(
                output_path, capi.fourchar_to_int('WAVE'), out_format
            )
        )

        # Process all files
        for input_file in inputs:
            data, count = input_file.read_packets(0, input_file.packet_count)
            output.write(count, data)
    # All files automatically closed


process_multiple_files(["audio.wav", "input.wav"], "joined.wav")
# --8<-- [end:example]
