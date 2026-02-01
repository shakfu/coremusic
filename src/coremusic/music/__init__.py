#!/usr/bin/env python3
"""CoreMusic music theory package.

This package provides music theory foundations for working with
notes, intervals, scales, chords, and rhythm.

Example:
    >>> from coremusic.music import theory
    >>>
    >>> # Create a C major scale
    >>> scale = theory.Scale(theory.Note('C', 4), theory.ScaleType.MAJOR)
    >>> notes = scale.get_notes()
    >>>
    >>> # Create a chord
    >>> chord = theory.Chord(theory.Note('C', 4), theory.ChordType.MAJOR_7)
    >>>
    >>> # Work with time signatures and rhythm
    >>> ts = theory.TimeSignature(4, 4)
    >>> duration = theory.Duration(theory.NoteValue.QUARTER, dots=1)
"""

from . import theory
from .theory import (
    CIRCLE_OF_FIFTHS,
    COMMON_PATTERNS,
    KEY_SIGNATURES,
    Chord,
    ChordType,
    Duration,
    Interval,
    IntervalQuality,
    MeterType,
    Mode,
    Note,
    NoteValue,
    RhythmPattern,
    Scale,
    ScaleType,
    TimeSignature,
    midi_to_note_name,
    note_name_to_midi,
)

__all__ = [
    # Submodule
    "theory",
    # Theory classes - Pitch
    "Note",
    "Interval",
    "IntervalQuality",
    "Scale",
    "ScaleType",
    "Chord",
    "ChordType",
    "Mode",
    # Theory classes - Rhythm
    "NoteValue",
    "MeterType",
    "TimeSignature",
    "Duration",
    "RhythmPattern",
    "COMMON_PATTERNS",
    # Constants
    "KEY_SIGNATURES",
    "CIRCLE_OF_FIFTHS",
    # Functions
    "note_name_to_midi",
    "midi_to_note_name",
]
