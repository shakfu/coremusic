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

from . import bayes, generative, markov, neural, theory
from .bayes import (CPT, BayesianNetwork, MIDIBayesAnalyzer,
                    MIDIBayesGenerator, NetworkConfig, NetworkMode,
                    NoteObservation, StructureMode)
from .bayes import \
    analyze_and_generate as \
    bayes_analyze_and_generate  # Bayesian Network Analysis; MIDI Analysis and Generation; Utility Functions
from .bayes import merge_networks, network_statistics
from .generative import (  # Base classes; Arpeggiator; Euclidean; Markov; Probabilistic; Sequence; Melody; Polyrhythm; Bit Shift Register
    ArpConfig, Arpeggiator, ArpPattern, BitShiftRegister,
    BitShiftRegisterConfig, BitShiftRegisterGenerator, EuclideanConfig,
    EuclideanGenerator, Generator, GeneratorConfig, MarkovConfig,
    MarkovGenerator, MelodyConfig, MelodyGenerator, PolyrhythmConfig,
    PolyrhythmGenerator, ProbabilisticConfig, ProbabilisticGenerator,
    SequenceConfig, SequenceGenerator)
from .markov import (  # Markov Chain Analysis; MIDI Analysis and Generation; Utility Functions
    ChainConfig, MarkovChain, MIDIMarkovAnalyzer, MIDIMarkovGenerator,
    ModelingMode, NoteData, RhythmMode, TransitionEdge, analyze_and_generate,
    chain_statistics, merge_chains)
from .neural import (  # Encoders; Dataset; Model functions; Training; Generation
    BaseEncoder, Callback, EarlyStopping, GreedySampling,
    LearningRateScheduler, MIDIDataset, ModelCheckpoint, ModelFactory,
    MusicGenerator, NoteEncoder, NucleusSampling, ProgressLogger,
    SamplingStrategy, TemperatureSampling, TopKSampling, Trainer,
    TrainingConfig, create_gru_model, create_lstm_model, create_mlp_model,
    create_rnn_model)
# Re-export commonly used classes
from .theory import (CIRCLE_OF_FIFTHS, KEY_SIGNATURES, Chord, ChordType,
                     Interval, IntervalQuality, Mode, Note, Scale, ScaleType,
                     midi_to_note_name, note_name_to_midi)

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
    # Neural Network Module
    "neural",
    "BaseEncoder",
    "NoteEncoder",
    "MIDIDataset",
    "create_mlp_model",
    "create_rnn_model",
    "create_lstm_model",
    "create_gru_model",
    "ModelFactory",
    # Neural - Training
    "TrainingConfig",
    "Trainer",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "ProgressLogger",
    "LearningRateScheduler",
    # Neural - Generation
    "SamplingStrategy",
    "GreedySampling",
    "TemperatureSampling",
    "TopKSampling",
    "NucleusSampling",
    "MusicGenerator",
]
