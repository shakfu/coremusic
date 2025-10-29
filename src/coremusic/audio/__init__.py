#!/usr/bin/env python3
"""CoreMusic audio package.

This package contains audio-related modules:
- async_io: Asynchronous I/O classes for audio operations
- streaming: Real-time audio streaming and processing
- analysis: Audio analysis and feature extraction
- slicing: Audio slicing and recombination
- visualization: Audio visualization and plotting
- utilities: High-level audio processing utilities
"""

# Import async I/O classes
from .async_io import *

# Import streaming classes
from .streaming import *

# Import analysis classes
from .analysis import *

# Import slicing classes
from .slicing import *

# Import visualization classes
from .visualization import *

# Import utilities
from .utilities import *

__all__ = [
    # Async I/O
    "AsyncAudioFile",
    "AsyncAudioQueue",
    # Streaming
    "AudioInputStream",
    "AudioOutputStream",
    "AudioProcessor",
    "StreamGraph",
    "StreamNode",
    "create_loopback",
    # Analysis
    "AudioAnalyzer",
    "LivePitchDetector",
    "BeatInfo",
    "PitchInfo",
    # Slicing
    "Slice",
    "AudioSlicer",
    "SliceCollection",
    "SliceRecombinator",
    "SliceMethod",
    "RecombineMethod",
    # Visualization
    "WaveformPlotter",
    "SpectrogramPlotter",
    "FrequencySpectrumPlotter",
    "MATPLOTLIB_AVAILABLE",
    "NUMPY_AVAILABLE",
    # Utilities
    "AudioEffectsChain",
    "AudioFormatPresets",
    "batch_convert",
    "batch_process_files",
    "batch_process_parallel",
    "convert_audio_file",
    "create_simple_effect_chain",
    "find_audio_unit_by_name",
    "get_audiounit_names",
    "list_available_audio_units",
    "parse_audio_stream_basic_description",
    "trim_audio",
]
