#!/usr/bin/env python3
"""CoreMusic audio package.

This package contains audio-related modules:
- async_io: Asynchronous I/O classes for audio operations
- streaming: Real-time audio streaming and processing
- analysis: Audio analysis and feature extraction
- slicing: Audio slicing and recombination
- visualization: Audio visualization and plotting
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
]
