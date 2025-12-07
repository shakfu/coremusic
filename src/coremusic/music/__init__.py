#!/usr/bin/env python3
"""CoreMusic music theory and generative music package.

This package provides music theory foundations and generative algorithms
for creating MIDI-based compositions.

Submodules:
- theory: Music theory primitives (notes, intervals, scales, chords)
- generative: Generative music algorithms (arpeggiator, euclidean, markov, etc.)

Example:
    >>> from coremusic.music import theory, generative
    >>>
    >>> # Create a C major scale
    >>> scale = theory.Scale(theory.Note('C', 4), theory.ScaleType.MAJOR)
    >>> notes = scale.get_notes()
    >>>
    >>> # Create an arpeggiator
    >>> arp = generative.Arpeggiator(
    ...     chord=theory.Chord(theory.Note('C', 4), theory.ChordType.MAJOR_7),
    ...     pattern=generative.ArpPattern.UP_DOWN,
    ...     note_duration=0.25
    ... )
    >>> events = arp.generate(num_cycles=2)
"""

from . import theory
from . import generative

# Re-export commonly used classes
from .theory import (
    Note,
    Interval,
    IntervalQuality,
    Scale,
    ScaleType,
    Chord,
    ChordType,
    Mode,
    KEY_SIGNATURES,
    CIRCLE_OF_FIFTHS,
    note_name_to_midi,
    midi_to_note_name,
)

from .generative import (
    # Base classes
    Generator,
    GeneratorConfig,
    # Arpeggiator
    Arpeggiator,
    ArpPattern,
    ArpConfig,
    # Euclidean
    EuclideanGenerator,
    EuclideanConfig,
    # Markov
    MarkovGenerator,
    MarkovConfig,
    # Probabilistic
    ProbabilisticGenerator,
    ProbabilisticConfig,
    # Sequence
    SequenceGenerator,
    SequenceConfig,
    # Melody
    MelodyGenerator,
    MelodyConfig,
    # Polyrhythm
    PolyrhythmGenerator,
    PolyrhythmConfig,
)

__all__ = [
    # Submodules
    "theory",
    "generative",
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
    # Generator base
    "Generator",
    "GeneratorConfig",
    # Arpeggiator
    "Arpeggiator",
    "ArpPattern",
    "ArpConfig",
    # Euclidean
    "EuclideanGenerator",
    "EuclideanConfig",
    # Markov
    "MarkovGenerator",
    "MarkovConfig",
    # Probabilistic
    "ProbabilisticGenerator",
    "ProbabilisticConfig",
    # Sequence
    "SequenceGenerator",
    "SequenceConfig",
    # Melody
    "MelodyGenerator",
    "MelodyConfig",
    # Polyrhythm
    "PolyrhythmGenerator",
    "PolyrhythmConfig",
]
