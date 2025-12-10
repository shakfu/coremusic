# Generative music examples - experimental
from .generative import (
    ArpConfig,
    Arpeggiator,
    ArpPattern,
    BitShiftRegister,
    BitShiftRegisterConfig,
    BitShiftRegisterGenerator,
    EuclideanConfig,
    EuclideanGenerator,
    Generator,
    GeneratorConfig,
    MarkovConfig,
    MarkovGenerator,
    MelodyConfig,
    MelodyGenerator,
    PolyrhythmConfig,
    PolyrhythmGenerator,
    ProbabilisticConfig,
    ProbabilisticGenerator,
    SequenceConfig,
    SequenceGenerator,
)
from .markov import (
    ChainConfig,
    MarkovChain,
    MIDIMarkovAnalyzer,
    MIDIMarkovGenerator,
    ModelingMode,
    NoteData,
    RhythmMode,
    TransitionEdge,
    analyze_and_generate,
    chain_statistics,
    merge_chains,
)
from .bayes import (
    BayesianNetwork,
    CPT,
    MIDIBayesAnalyzer,
    MIDIBayesGenerator,
    NetworkConfig,
    NetworkMode,
    NoteObservation,
    StructureMode,
    analyze_and_generate as bayes_analyze_and_generate,
    merge_networks,
    network_statistics,
)

__all__ = [
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
    # Markov Generator
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
    # Bayesian Network
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
