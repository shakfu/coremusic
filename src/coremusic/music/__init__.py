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
from . import markov
from . import bayes

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
    # Bit Shift Register
    BitShiftRegister,
    BitShiftRegisterGenerator,
    BitShiftRegisterConfig,
)

from .markov import (
    # Markov Chain Analysis
    MarkovChain,
    ChainConfig,
    ModelingMode,
    RhythmMode,
    NoteData,
    TransitionEdge,
    # MIDI Analysis and Generation
    MIDIMarkovAnalyzer,
    MIDIMarkovGenerator,
    # Utility Functions
    analyze_and_generate,
    merge_chains,
    chain_statistics,
)

from .bayes import (
    # Bayesian Network Analysis
    NetworkMode,
    StructureMode,
    NoteObservation,
    NetworkConfig,
    CPT,
    BayesianNetwork,
    # MIDI Analysis and Generation
    MIDIBayesAnalyzer,
    MIDIBayesGenerator,
    # Utility Functions
    analyze_and_generate as bayes_analyze_and_generate,
    merge_networks,
    network_statistics,
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
    # Bit Shift Register
    "BitShiftRegister",
    "BitShiftRegisterGenerator",
    "BitShiftRegisterConfig",
    # Markov Chain Analysis
    "markov",
    "MarkovChain",
    "ChainConfig",
    "ModelingMode",
    "RhythmMode",
    "NoteData",
    "TransitionEdge",
    "MIDIMarkovAnalyzer",
    "MIDIMarkovGenerator",
    "analyze_and_generate",
    "merge_chains",
    "chain_statistics",
    # Bayesian Network Analysis
    "bayes",
    "NetworkMode",
    "StructureMode",
    "NoteObservation",
    "NetworkConfig",
    "CPT",
    "BayesianNetwork",
    "MIDIBayesAnalyzer",
    "MIDIBayesGenerator",
    "bayes_analyze_and_generate",
    "merge_networks",
    "network_statistics",
]
