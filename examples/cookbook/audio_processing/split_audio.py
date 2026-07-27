#!/usr/bin/env python3
"""Cut a file into pieces."""

# --8<-- [start:fixed]
from pathlib import Path

from coremusic.audio import AudioFile, trim_audio


def split_audio(input_path, output_dir, chunk_duration=1.0):
    """Split audio file into fixed-duration chunks"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with AudioFile(input_path) as audio:
        duration = audio.duration

    index = 0
    start = 0.0
    while start < duration:
        end = min(start + chunk_duration, duration)
        trim_audio(input_path, str(output_dir / f"chunk_{index:03d}.wav"), start, end)
        start = end
        index += 1

    print(f"Wrote {index} chunks to {output_dir}")


split_audio("input.wav", "chunks", chunk_duration=1.0)
# --8<-- [end:fixed]

# --8<-- [start:onsets]
from coremusic.audio import AudioSlicer

# To cut on musical boundaries instead of a fixed grid, let AudioSlicer find
# the onsets and export what it found.
slicer = AudioSlicer("input.wav", method="onset")
slices = slicer.detect_slices()
print(f"Detected {len(slices)} slices")

paths = slicer.export_slices("slices", name_template="slice_{index:03d}.wav")
print(f"Exported {len(paths)} files")
# --8<-- [end:onsets]
