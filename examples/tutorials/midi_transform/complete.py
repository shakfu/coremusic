#!/usr/bin/env python3
"""Complete Example."""

# --8<-- [start:example]
from coremusic.midi.transform import (
    Harmonize,
    Humanize,
    Pipeline,
    Quantize,
    Transpose,
    VelocityCurve,
    VelocityScale,
)
from coremusic.midi.utilities import MIDISequence

# Load source MIDI
original = MIDISequence.load("input.mid")
print(f"Loaded: {len(original.tracks)} tracks, {original.duration:.2f}s")

# Create processing pipeline
pipeline = Pipeline([
    # Fix timing
    Quantize(grid=0.125, strength=0.7),

    # Transpose to different key
    Transpose(5),  # Up a fourth

    # Shape dynamics
    VelocityCurve(curve='soft'),
    VelocityScale(min_vel=50, max_vel=110),

    # Add expression
    Humanize(timing=0.015, velocity=8, seed=42),
])

# Apply transformations
processed = pipeline.apply(original)

# Save result
processed.save("processed.mid")
print("Saved processed file")

# Create harmony version
harmony_pipeline = Pipeline([
    Harmonize([4, 7]),          # Add thirds and fifths
    VelocityScale(factor=0.7),  # Reduce volume
])
harmony = harmony_pipeline.apply(original)
harmony.save("harmony.mid")
# --8<-- [end:example]
