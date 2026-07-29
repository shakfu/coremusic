#!/usr/bin/env python3
"""Filter Transformers."""

from coremusic.midi.utilities import MIDISequence

sequence = MIDISequence.load("input.mid")

# --8<-- [start:notes]
from coremusic.midi.transform import NoteFilter

# Keep only bass notes (MIDI 24-48)
bass = NoteFilter(min_note=24, max_note=48).transform(sequence)

# Keep only loud notes
loud = NoteFilter(min_velocity=80).transform(sequence)

# Keep specific channels
channel_0 = NoteFilter(channels={0}).transform(sequence)

# Remove matching notes (invert filter)
no_bass = NoteFilter(min_note=24, max_note=48, invert=True).transform(sequence)
# --8<-- [end:notes]

# --8<-- [start:events]
from coremusic.midi.transform import EventTypeFilter
from coremusic.midi.utilities import MIDIStatus

# Keep only note events
notes_only = EventTypeFilter(keep=[MIDIStatus.NOTE_ON, MIDIStatus.NOTE_OFF]).transform(
    sequence
)

# Remove control changes
no_cc = EventTypeFilter(remove=[MIDIStatus.CONTROL_CHANGE]).transform(sequence)
# --8<-- [end:events]
