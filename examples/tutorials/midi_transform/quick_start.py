#!/usr/bin/env python3
"""Quick Start."""

# --8<-- [start:example]
from coremusic.midi.transform import Humanize, Pipeline, Quantize, Transpose
from coremusic.midi.utilities import MIDISequence

# Load MIDI file
seq = MIDISequence.load("input.mid")

# Create transformation pipeline
pipeline = Pipeline(
    [
        Transpose(semitones=5),  # Up a perfect fourth
        Quantize(grid=0.125, strength=0.8),  # Quantize to 16th notes
        Humanize(timing=0.01, velocity=5),  # Add human feel
    ]
)

# Apply and save
transformed = pipeline.apply(seq)
transformed.save("output.mid")
# --8<-- [end:example]
