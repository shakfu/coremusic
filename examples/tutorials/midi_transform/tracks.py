#!/usr/bin/env python3
"""Track Transformers."""

from coremusic.midi.utilities import MIDISequence

sequence = MIDISequence.load("input.mid")

# --8<-- [start:channel]
from coremusic.midi.transform import ChannelRemap

# Move melody from channel 0 to channel 1
remapped = ChannelRemap({0: 1}).transform(sequence)

# Move to drums channel
drums = ChannelRemap({0: 9}).transform(sequence)
# --8<-- [end:channel]

# --8<-- [start:merge]
from coremusic.midi.transform import TrackMerge

merged = TrackMerge(name="Combined").transform(sequence)
# --8<-- [end:merge]

# --8<-- [start:arpeggio]
from coremusic.midi.transform import Arpeggiate

# Arpeggiate upward
arp_up = Arpeggiate(
    pattern='up',
    note_duration=0.1
).transform(sequence)

# Available patterns: 'up', 'down', 'up_down', 'down_up', 'random'
arp_down = Arpeggiate(pattern='down', note_duration=0.1).transform(sequence)
arp_random = Arpeggiate(
    pattern='random', note_duration=0.1, seed=42
).transform(sequence)
# --8<-- [end:arpeggio]
