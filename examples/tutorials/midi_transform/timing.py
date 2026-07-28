#!/usr/bin/env python3
"""Time Transformers."""

from coremusic.midi.utilities import MIDISequence

sequence = MIDISequence.load("input.mid")

# --8<-- [start:quantize]
from coremusic.midi.transform import Quantize

# Full quantize to 16th notes (0.125s at 120 BPM)
quantized = Quantize(grid=0.125, strength=1.0).transform(sequence)

# Partial quantize (preserves some groove)
soft_quant = Quantize(grid=0.125, strength=0.5).transform(sequence)

# Add swing feel
swing = Quantize(grid=0.125, swing=0.3).transform(sequence)
# --8<-- [end:quantize]

# --8<-- [start:stretch]
from coremusic.midi.transform import TimeStretch

# Double the tempo (half the time)
faster = TimeStretch(0.5).transform(sequence)

# Half the tempo (double the time)
slower = TimeStretch(2.0).transform(sequence)
# --8<-- [end:stretch]

# --8<-- [start:shift]
from coremusic.midi.transform import TimeShift

# Delay by 1 second
delayed = TimeShift(1.0).transform(sequence)

# Shift earlier (with clamping at 0)
earlier = TimeShift(-0.5).transform(sequence)
# --8<-- [end:shift]

# --8<-- [start:reverse]
from coremusic.midi.transform import Reverse

# Reverse note order, preserving durations
reversed_seq = Reverse().transform(sequence)
# --8<-- [end:reverse]
