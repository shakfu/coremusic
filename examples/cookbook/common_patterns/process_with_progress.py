#!/usr/bin/env python3
"""Progress Tracking."""

def process_chunk(chunk):
    pass


# --8<-- [start:example]
from coremusic.audio import AudioFile

def process_with_progress(filepath, callback=None):
    """Process audio with progress callback."""
    with AudioFile(filepath) as audio:
        total = audio.packet_count
        processed = 0
        chunk_size = 4096

        while processed < total:
            data, count = audio.read_packets(processed, chunk_size)
            if count == 0:
                break

            # Process data
            process_chunk(data)

            processed += count

            # Report progress
            if callback:
                progress = processed / total
                callback(progress)

# Usage with progress bar
def show_progress(progress):
    bar_length = 40
    filled = int(bar_length * progress)
    bar = '=' * filled + '-' * (bar_length - filled)
    print(f'\r[{bar}] {progress:.1%}', end='')

process_with_progress("audio.wav", callback=show_progress)
print()  # New line after progress bar
# --8<-- [end:example]
