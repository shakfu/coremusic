#!/usr/bin/env python3
"""Importing from the midi subpackage."""

# --8<-- [start:objects]
from coremusic.midi import (
    MIDIClient,
    MIDIEndpoint,
    MIDIInputPort,
    MIDIOutputPort,
    MusicPlayer,
    MusicSequence,
    MusicTrack,
    get_destinations,
    get_sources,
)

client = MIDIClient("MyApp")
client.dispose()
# --8<-- [end:objects]

# --8<-- [start:files]
from coremusic.midi import MIDIEvent, MIDISequence, MIDITrack

sequence = MIDISequence.load("song.mid")
print(f"{sequence.duration:.2f} beats")
# --8<-- [end:files]

# --8<-- [start:transform]
from coremusic.midi import Humanize, Pipeline, Quantize, Transpose, transpose

# Class-based pipeline, or the one-call functions
pipeline = Pipeline([Transpose(semitones=5), Quantize(grid=0.25)])
transposed = transpose(sequence, 5)
# --8<-- [end:transform]

# --8<-- [start:link]
from coremusic.midi.link import LinkMIDIClock, LinkMIDISequencer

# The submodule is also reachable from the package
from coremusic.midi import link
# --8<-- [end:link]
