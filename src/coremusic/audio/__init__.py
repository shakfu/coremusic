#!/usr/bin/env python3
"""CoreMusic audio package.

This package contains audio-related modules:
- core: Audio file and format classes (AudioFile, AudioFormat, AudioQueue, etc.)
- units: AudioUnit classes (AudioComponent, AudioComponentDescription, AudioUnit)
- graph: AUGraph classes
- devices: Audio device classes (AudioDevice, AudioDeviceManager)
- clock: Audio clock classes (AudioClock, ClockTimeFormat)
- async_io: Asynchronous I/O classes for audio operations
- streaming: Real-time audio streaming and processing
- analysis: Audio analysis and feature extraction
- slicing: Audio slicing and recombination
- visualization: Audio visualization and plotting
- utilities: High-level audio processing utilities
- mmap_file: Memory-mapped file access for performance
- buffer_pool: Buffer pooling for efficient memory management
"""

# Import analysis classes
from .analysis import AudioAnalyzer, BeatInfo, LivePitchDetector, PitchInfo

# Import async I/O classes (re-exported)
from .async_io import AsyncAudioFile as AsyncAudioFile
from .async_io import AsyncAudioQueue as AsyncAudioQueue
from .async_io import create_output_queue_async as create_output_queue_async
from .async_io import open_audio_file_async as open_audio_file_async
from .buffer_pool import (
    BufferPool,
    BufferPoolStats,
    PooledBuffer,
    get_global_pool,
    reset_global_pool,
)

# Import domain object classes
from .clock import AudioClock, ClockTimeFormat
from .core import (
    AudioBuffer,
    AudioConverter,
    AudioFile,
    AudioFileStream,
    AudioFormat,
    AudioQueue,
    ExtendedAudioFile,
)
from .devices import AudioDevice, AudioDeviceManager
from .graph import AUGraph

# Import performance modules
from .mmap_file import MMapAudioFile

# Import slicing classes
from .slicing import (
    AudioSlicer,
    RecombineMethod,
    Slice,
    SliceCollection,
    SliceMethod,
    SliceRecombinator,
)

# Import streaming classes
from .streaming import (
    AudioInputStream,
    AudioOutputStream,
    AudioProcessor,
    StreamGraph,
    StreamNode,
    create_loopback,
)
from .units import AudioComponent, AudioComponentDescription, AudioUnit

# Import utilities
from .utilities import (
    AudioEffectsChain,
    AudioFormatPresets,
    batch_convert,
    batch_process_files,
    batch_process_parallel,
    convert_audio_file,
    create_simple_effect_chain,
    find_audio_unit_by_name,
    get_audiounit_names,
    list_available_audio_units,
    parse_audio_stream_basic_description,
    trim_audio,
)

# Import visualization classes
from .visualization import (
    MATPLOTLIB_AVAILABLE,
    NUMPY_AVAILABLE,
    FrequencySpectrumPlotter,
    SpectrogramPlotter,
    WaveformPlotter,
)

# Import Cython-optimized operations from capi (re-exported when available)
# These are defined in capi.pyx and may not be visible to type checkers
try:
    from ..capi import apply_fade_in_float32 as apply_fade_in_float32
    from ..capi import apply_fade_out_float32 as apply_fade_out_float32
    from ..capi import apply_gain as apply_gain
    from ..capi import apply_gain_float32 as apply_gain_float32
    from ..capi import calculate_peak as calculate_peak
    from ..capi import calculate_peak_float32 as calculate_peak_float32
    from ..capi import calculate_rms as calculate_rms
    from ..capi import calculate_rms_float32 as calculate_rms_float32
    from ..capi import convert_float32_to_int16 as convert_float32_to_int16
    from ..capi import convert_int16_to_float32 as convert_int16_to_float32
    from ..capi import mix_audio_float32 as mix_audio_float32
    from ..capi import mono_to_stereo_float32 as mono_to_stereo_float32
    from ..capi import normalize_audio as normalize_audio
    from ..capi import normalize_audio_float32 as normalize_audio_float32
    from ..capi import stereo_to_mono_float32 as stereo_to_mono_float32

    CYTHON_OPS_AVAILABLE = True
except ImportError:
    CYTHON_OPS_AVAILABLE = False

__all__ = [
    # Core audio objects
    "AudioFile",
    "AudioFormat",
    "AudioQueue",
    "AudioBuffer",
    "AudioConverter",
    "AudioFileStream",
    "ExtendedAudioFile",
    # AudioUnit
    "AudioComponent",
    "AudioComponentDescription",
    "AudioUnit",
    # AUGraph
    "AUGraph",
    # Devices
    "AudioDevice",
    "AudioDeviceManager",
    # Clock
    "AudioClock",
    "ClockTimeFormat",
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
    __all__.extend(
        [
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
        ]
    )
