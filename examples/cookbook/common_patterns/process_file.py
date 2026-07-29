#!/usr/bin/env python3
"""Parallel Processing."""

# --8<-- [start:example]
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from coremusic.audio import AudioFile


def process_file(filepath):
    """Process a single audio file."""
    with AudioFile(filepath) as audio:
        # Your processing logic
        return {"path": str(filepath), "duration": audio.duration}


def process_batch(filepaths, max_workers=4):
    """Process multiple files in parallel."""
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, fp): fp for fp in filepaths}

        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                filepath = futures[future]
                print(f"Error processing {filepath}: {e}")

    return results


# Usage
wav_files = list(Path("audio_dir").glob("*.wav"))
results = process_batch(wav_files)
# --8<-- [end:example]
