#!/usr/bin/env python3
"""CoreMusic music theory package.

This package provides music theory foundations for working with
notes, intervals, scales, and chords.

Example:
    >>> from coremusic.music import theory
    >>>
    >>> # Create a C major scale
    >>> scale = theory.Scale(theory.Note('C', 4), theory.ScaleType.MAJOR)
    >>> notes = scale.get_notes()
    >>>
    >>> # Create a chord
    >>> chord = theory.Chord(theory.Note('C', 4), theory.ChordType.MAJOR_7)
"""

from . import theory
from .theory import (
    CIRCLE_OF_FIFTHS,
    KEY_SIGNATURES,
    Chord,
    ChordType,
    Interval,
    IntervalQuality,
    Mode,
    Note,
    Scale,
    ScaleType,
    midi_to_note_name,
    note_name_to_midi,
)

__all__ = [
    # Submodule
    "theory",
    # Theory classes
    "Note",
    "Interval",
    "IntervalQuality",
    "Scale",
    "ScaleType",
    "Chord",
    "ChordType",
    "Mode",
    "KEY_SIGNATURES",
    "CIRCLE_OF_FIFTHS",
    "note_name_to_midi",
    "midi_to_note_name",
]
