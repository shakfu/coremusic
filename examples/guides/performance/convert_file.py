#!/usr/bin/env python3
"""Parallel File Processing."""

# --8<-- [start:example]
from coremusic.audio import AudioFile

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def convert_file(input_path):
    """Convert single file"""
    output_path = input_path.with_suffix('.mp3')

    with AudioFile(str(input_path)) as audio:
        format = audio.format
        # Conversion logic...

    return output_path

def batch_convert(input_dir, num_workers=4):
    """Convert all files in directory"""
    files = list(Path(input_dir).glob("*.wav"))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(convert_file, files)

    return list(results)

# Convert 100 files using 4 cores
results = batch_convert("audio_files/", num_workers=4)
# --8<-- [end:example]
