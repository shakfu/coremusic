#!/usr/bin/env python3
"""Chunked Processing."""

# --8<-- [start:example]
import numpy as np

from coremusic import capi
from coremusic.audio import AudioFile, AudioFormat, ExtendedAudioFile


def process_large_file(input_path, output_path, chunk_size=8192):
    """Process a large audio file without loading all of it."""
    with AudioFile(input_path) as input_file:
        source_format = input_file.format
        total_packets = input_file.packet_count

        # Work in float internally, and write what we actually produced
        out_format = AudioFormat.pcm(
            source_format.sample_rate,
            channels=source_format.channels_per_frame,
            bits=32,
            is_float=True,
        )

        with ExtendedAudioFile.create(
            output_path, capi.fourchar_to_int("WAVE"), out_format
        ) as output_file:
            processed = 0

            while processed < total_packets:
                # Read the next chunk - note the offset, not a fixed 0
                remaining = min(chunk_size, total_packets - processed)
                data, count = input_file.read_packets(processed, remaining)
                if count == 0:
                    break

                # Process
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                samples /= 32768.0
                samples *= 0.8  # Example processing

                # Write
                output_file.write(count, samples.tobytes())
                processed += count

                # Progress
                print(f"Progress: {processed / total_packets * 100:.1f}%", end="\r")

    print()


process_large_file("audio.wav", "processed_large.wav")
# --8<-- [end:example]
