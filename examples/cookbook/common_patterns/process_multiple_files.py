#!/usr/bin/env python3
"""Multiple Resources Pattern."""

# --8<-- [start:example]
from coremusic.audio import AudioFile, ExtendedAudioFile

from contextlib import ExitStack

def process_multiple_files(input_files, output_path):
    """Process multiple input files safely."""
    with ExitStack() as stack:
        # Open all input files
        inputs = [
            stack.enter_context(AudioFile(f))
            for f in input_files
        ]

        # Open output file
        output = stack.enter_context(
            ExtendedAudioFile.create(output_path, ...)
        )

        # Process all files
        for input_file in inputs:
            data, count = input_file.read_packets(0, input_file.packet_count)
            output.write(count, data)
    # All files automatically closed
# --8<-- [end:example]
