#!/usr/bin/env python3
"""Chunked Processing."""

# --8<-- [start:example]
import coremusic.capi as capi
from coremusic.audio import AudioFile, ExtendedAudioFile

import numpy as np

def process_large_file(input_path, output_path, chunk_size=8192):
    """Process large audio file efficiently"""
    with AudioFile(input_path) as input_file:
        format = input_file.format

        with ExtendedAudioFile.create(
            output_path,
            capi.fourchar_to_int('WAVE'),
            format
        ) as output_file:
            total_frames = input_file.packet_count
            processed = 0

            while processed < total_frames:
                # Read chunk
                remaining = min(chunk_size, total_frames - processed)
                data, count = input_file.read_packets(0, remaining)

                # Process
                samples = np.frombuffer(data, dtype=np.float32)
                samples *= 0.8  # Example processing

                # Write
                output_file.write(count, samples.tobytes())
                processed += count

                # Progress
                progress = (processed / total_frames) * 100
                print(f"Progress: {progress:.1f}%", end='\r')
# --8<-- [end:example]
