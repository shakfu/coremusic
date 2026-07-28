#!/usr/bin/env python3
"""Convenience Functions."""

from coremusic.midi.utilities import MIDISequence

sequence = MIDISequence.load("input.mid")

# --8<-- [start:example]
from coremusic.midi.transform import (
    humanize,
    quantize,
    reverse,
    scale_velocity,
    transpose,
)

# Quick transformations
result = transpose(sequence, 5)
result = quantize(sequence, 0.125)
result = humanize(sequence, timing=0.01, velocity=5)
result = reverse(sequence)
result = scale_velocity(sequence, factor=0.8)

# Chain them
result = humanize(quantize(transpose(sequence, 12), 0.25), timing=0.01)
# --8<-- [end:example]
