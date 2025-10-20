# Import functional API and CoreAudioObject Base class
from .capi import *

# Import pure Python OO classes
from .objects import (
    # Exception hierarchy
    CoreAudioError,
    AudioFileError,
    AudioQueueError,
    AudioUnitError,
    AudioConverterError,
    MIDIError,
    MusicPlayerError,
    AudioDeviceError,
    AUGraphError,

    # Audio formats and data structures
    AudioFormat,

    # Audio File Framework
    AudioFile,
    AudioFileStream,
    ExtendedAudioFile,

    # AudioConverter Framework
    AudioConverter,

    # Audio Queue Framework
    AudioBuffer,
    AudioQueue,

    # Audio Component & AudioUnit Framework
    AudioComponentDescription,
    AudioComponent,
    AudioUnit,

    # MIDI Framework
    MIDIClient,
    MIDIPort,
    MIDIInputPort,
    MIDIOutputPort,

    # Audio Device & Hardware
    AudioDevice,
    AudioDeviceManager,

    # AUGraph Framework
    AUGraph,
)

# NumPy availability flag
try:
    from .objects import NUMPY_AVAILABLE
except ImportError:
    NUMPY_AVAILABLE = False

# SciPy availability flag and utilities
try:
    from .scipy_utils import (
        SCIPY_AVAILABLE,
        # Filter design
        design_butterworth_filter,
        design_chebyshev_filter,
        # Filter application
        apply_filter,
        apply_lowpass_filter,
        apply_highpass_filter,
        apply_bandpass_filter,
        # Resampling
        resample_audio,
        # Spectral analysis
        compute_spectrum,
        compute_fft,
        compute_spectrogram,
        # High-level processor
        AudioSignalProcessor,
    )
except ImportError:
    SCIPY_AVAILABLE = False
    # Provide None placeholders for type checking
    design_butterworth_filter = None  # type: ignore
    design_chebyshev_filter = None  # type: ignore
    apply_filter = None  # type: ignore
    apply_lowpass_filter = None  # type: ignore
    apply_highpass_filter = None  # type: ignore
    apply_bandpass_filter = None  # type: ignore
    resample_audio = None  # type: ignore
    compute_spectrum = None  # type: ignore
    compute_fft = None  # type: ignore
    compute_spectrogram = None  # type: ignore
    AudioSignalProcessor = None  # type: ignore

# Import async I/O classes
from .async_io import (
    AsyncAudioFile,
    AsyncAudioQueue,
    open_audio_file_async,
    create_output_queue_async,
)

# Import high-level utilities
from .utilities import (
    AudioAnalyzer,
    AudioFormatPresets,
    batch_convert,
    convert_audio_file,
    trim_audio,
    AudioEffectsChain,
    create_simple_effect_chain,
    find_audio_unit_by_name,
    list_available_audio_units,
    get_audiounit_names,
)
