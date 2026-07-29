#!/usr/bin/env python3
"""Velocity Transformers."""

from coremusic.midi.utilities import MIDISequence

sequence = MIDISequence.load("input.mid")

# --8<-- [start:scale]
from coremusic.midi.transform import VelocityScale

# Scale by factor
quieter = VelocityScale(factor=0.5).transform(sequence)
louder = VelocityScale(factor=1.5).transform(sequence)

# Compress to range
compressed = VelocityScale(min_vel=40, max_vel=100).transform(sequence)
# --8<-- [end:scale]

# --8<-- [start:curve]
from coremusic.midi.transform import VelocityCurve

# Built-in curves
soft = VelocityCurve(curve="soft").transform(sequence)  # Softer dynamics
hard = VelocityCurve(curve="hard").transform(sequence)  # Harder dynamics
log = VelocityCurve(curve="log").transform(sequence)  # Logarithmic
exp = VelocityCurve(curve="exp").transform(sequence)  # Exponential

# Custom curve function (input/output 0.0-1.0)
custom = VelocityCurve(curve=lambda x: x**0.7).transform(sequence)
# --8<-- [end:curve]

# --8<-- [start:humanize]
from coremusic.midi.transform import Humanize

# Add subtle variation
humanized = Humanize(
    timing=0.01,  # +/- 10ms timing variation
    velocity=5,  # +/- 5 velocity variation
).transform(sequence)

# Reproducible with seed
reproducible = Humanize(timing=0.02, velocity=10, seed=42).transform(sequence)
# --8<-- [end:humanize]
