#!/usr/bin/env python3
"""Pipeline Basics."""

from coremusic.midi.utilities import MIDISequence

sequence = MIDISequence.load("input.mid")

# --8<-- [start:creating]
from coremusic.midi.transform import Pipeline, Transpose, VelocityScale

# Create with list of transformers
pipeline = Pipeline([
    Transpose(5),
    VelocityScale(factor=0.8),
])

# Or build incrementally
pipeline = Pipeline()
pipeline.add(Transpose(5))
pipeline.add(VelocityScale(factor=0.8))

# Apply to sequence
result = pipeline.apply(sequence)

# Pipelines are callable
result = pipeline(sequence)
# --8<-- [end:creating]

# --8<-- [start:individual]
from coremusic.midi.transform import Reverse, Transpose

# Direct transform call
transposed = Transpose(12).transform(sequence)

# Transformers are callable
reversed_seq = Reverse()(sequence)
# --8<-- [end:individual]
