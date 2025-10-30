#!/usr/bin/env python3
"""CoreMusic audio package.

This package contains audio-related modules:
- async_io: Asynchronous I/O classes for audio operations
- streaming: Real-time audio streaming and processing
- analysis: Audio analysis and feature extraction
- slicing: Audio slicing and recombination
- visualization: Audio visualization and plotting
- utilities: High-level audio processing utilities
- mmap_file: Memory-mapped file access for performance
- buffer_pool: Buffer pooling for efficient memory management
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

# Import performance modules
from .mmap_file import MMapAudioFile
from .buffer_pool import (
    BufferPool,
    PooledBuffer,
    BufferPoolStats,
    get_global_pool,
    reset_global_pool,
)

# Import Cython-optimized operations from capi
try:
    from ..capi import (
        normalize_audio,
        apply_gain,
        calculate_rms,
        calculate_peak,
        normalize_audio_float32,
        apply_gain_float32,
        mix_audio_float32,
        convert_float32_to_int16,
        convert_int16_to_float32,
        stereo_to_mono_float32,
        mono_to_stereo_float32,
        apply_fade_in_float32,
        apply_fade_out_float32,
        calculate_rms_float32,
        calculate_peak_float32,
    )
    CYTHON_OPS_AVAILABLE = True
except ImportError:
    CYTHON_OPS_AVAILABLE = False

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
    # Performance
    "MMapAudioFile",
    "BufferPool",
    "PooledBuffer",
    "BufferPoolStats",
    "get_global_pool",
    "reset_global_pool",
    # Cython ops (if available)
    "CYTHON_OPS_AVAILABLE",
]

# Add Cython ops to __all__ if available
if CYTHON_OPS_AVAILABLE:
    __all__.extend([
        "normalize_audio",
        "apply_gain",
        "calculate_rms",
        "calculate_peak",
        "normalize_audio_float32",
        "apply_gain_float32",
        "mix_audio_float32",
        "convert_float32_to_int16",
        "convert_int16_to_float32",
        "stereo_to_mono_float32",
        "mono_to_stereo_float32",
        "apply_fade_in_float32",
        "apply_fade_out_float32",
        "calculate_rms_float32",
        "calculate_peak_float32",
    ])
