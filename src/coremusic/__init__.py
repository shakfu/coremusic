#!/usr/bin/env python3
"""CoreMusic: Python bindings for Apple CoreAudio, CoreMIDI, and Ableton Link.

This package provides Python bindings for Apple's CoreAudio and CoreMIDI ecosystems,
plus Ableton Link tempo synchronization, exposing the APIs through Python.

The primary interface is the object-oriented API with automatic resource
management and context manager support. This is itself built-up from the low-level
functional C API which is available via the `capi` submodule for advanced use cases.

Basic Usage
-----------
::

    import coremusic as cm

    # Read an audio file
    with cm.AudioFile("audio.wav") as audio:
        print(f"Duration: {audio.duration:.2f}s")
        print(f"Sample rate: {audio.format.sample_rate}Hz")
        data, count = audio.read_packets(0, 1024)

    # Use constants (preferred over capi getter functions)
    cm.AudioFileProperty.DATA_FORMAT
    cm.AudioFormatID.LINEAR_PCM

Async/Await Support
-------------------
CoreMusic provides async versions of audio classes for non-blocking I/O::

    import asyncio
    import coremusic as cm

    async def process_audio():
        # Async file reading with chunk streaming
        async with cm.AsyncAudioFile("large_file.wav") as audio:
            print(f"Duration: {audio.duration:.2f}s")
            async for chunk in audio.read_chunks_async(chunk_size=4096):
                await process_chunk(chunk)

        # Async AudioQueue playback
        format = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2)
        queue = await cm.AsyncAudioQueue.new_output_async(format)
        await queue.start_async()
        await asyncio.sleep(1.0)
        await queue.stop_async()

    asyncio.run(process_audio())

NumPy Integration (Optional)
----------------------------
NumPy is an optional dependency. When installed, additional functionality is available::

    import coremusic as cm

    # Check if NumPy is available
    if cm.NUMPY_AVAILABLE:
        import numpy as np

        # AudioFormat can convert to NumPy dtype
        with cm.AudioFile("audio.wav") as audio:
            dtype = audio.format.to_numpy_dtype()

        # Memory-mapped files support zero-copy NumPy arrays
        with cm.MMapAudioFile("large.wav") as mmap:
            audio_np = mmap.read_as_numpy(start_frame=0, num_frames=44100)

To install with NumPy support, NumPy must be installed separately::

    pip install numpy

Module Organization
-------------------
- ``coremusic`` - Object-oriented API (primary interface)
- ``coremusic.capi`` - Low-level functional C API
- ``coremusic.constants`` - Enum classes for CoreAudio constants
- ``coremusic.audio`` - Audio processing utilities (async_io, streaming, analysis)
- ``coremusic.midi`` - MIDI utilities and Link integration
- ``coremusic.music`` - Music theory (notes, scales, chords)
- ``coremusic.link`` - Ableton Link tempo synchronization
- ``coremusic.utils.scipy`` - SciPy integration (optional)
"""

# Import subpackages
from . import audio, buffer_utils, constants, link, midi, music, os_status, utils

# Import async I/O classes from new location
from .audio.async_io import (
    AsyncAudioFile,
    AsyncAudioQueue,
    create_output_queue_async,
    open_audio_file_async,
)

# Import AudioUnit hosting
from .audio.audiounit_host import (
    AudioFormatConverter,
    AudioUnitChain,
    AudioUnitHost,
    AudioUnitParameter,
    AudioUnitPlugin,
    AudioUnitPreset,
    PluginAudioFormat,
    PresetManager,
)

# Import high-level utilities (re-exported)
from .audio.utilities import AudioEffectsChain as AudioEffectsChain
from .audio.utilities import AudioFormatPresets as AudioFormatPresets
from .audio.utilities import batch_convert as batch_convert
from .audio.utilities import batch_process_files as batch_process_files
from .audio.utilities import batch_process_parallel as batch_process_parallel
from .audio.utilities import convert_audio_file as convert_audio_file
from .audio.utilities import create_simple_effect_chain as create_simple_effect_chain
from .audio.utilities import find_audio_unit_by_name as find_audio_unit_by_name
from .audio.utilities import get_audiounit_names as get_audiounit_names
from .audio.utilities import list_available_audio_units as list_available_audio_units
from .audio.utilities import (
    parse_audio_stream_basic_description as parse_audio_stream_basic_description,
)
from .audio.utilities import trim_audio as trim_audio
from .buffer_utils import (
    AudioStreamBasicDescription,
    calculate_buffer_size,
    optimal_buffer_size,
    pack_audio_buffer,
    unpack_audio_buffer,
)

# Import commonly-used constants for convenience
from .constants import (
    # Audio File
    AudioFilePermission,
    AudioFileProperty,
    AudioFileType,
    # Audio Format
    AudioFormatID,
    LinearPCMFormatFlag,
    # Audio Converter
    AudioConverterProperty,
    AudioConverterQuality,
    # Extended Audio File
    ExtendedAudioFileProperty,
    # AudioUnit
    AudioUnitElement,
    AudioUnitParameterUnit,
    AudioUnitProperty,
    AudioUnitRenderActionFlags,
    AudioUnitScope,
    # Audio Queue
    AudioQueueParameter,
    AudioQueueProperty,
    # Audio Object/Device
    AudioDeviceProperty,
    AudioObjectProperty,
    # MIDI
    MIDIControlChange,
    MIDIObjectProperty,
    MIDIStatus,
)

# Import Link + MIDI integration from new location
from .midi import link as link_midi

# Import object-oriented API (primary interface)
from .objects import (
    NUMPY_AVAILABLE,
    AudioBuffer,
    AudioClock,
    AudioComponent,
    AudioComponentDescription,
    AudioConverter,
    AudioConverterError,
    AudioDevice,
    AudioDeviceError,
    AudioDeviceManager,
    AudioFile,
    AudioFileError,
    AudioFileStream,
    AudioFormat,
    AudioPlayer,
    AudioQueue,
    AudioQueueError,
    AudioUnit,
    AudioUnitError,
    AUGraph,
    AUGraphError,
    ClockTimeFormat,
    CoreAudioError,
    CoreAudioObject,
    ExtendedAudioFile,
    MIDIClient,
    MIDIError,
    MIDIInputPort,
    MIDIOutputPort,
    MIDIPort,
    MusicPlayer,
    MusicPlayerError,
    MusicSequence,
    MusicTrack,
)
from .os_status import (
    check_os_status,
    check_return_status,
    format_os_status_error,
    handle_exceptions,
    raises_on_error,
)

__all__ = [
    # Base class
    "CoreAudioObject",
    # Exception hierarchy
    "CoreAudioError",
    "AudioFileError",
    "AudioQueueError",
    "AudioUnitError",
    "AudioConverterError",
    "MIDIError",
    "MusicPlayerError",
    "AudioDeviceError",
    "AUGraphError",
    # Audio formats and data structures
    "AudioFormat",
    # Audio File Framework
    "AudioFile",
    "AudioFileStream",
    "ExtendedAudioFile",
    # AudioConverter Framework
    "AudioConverter",
    # Audio Queue Framework
    "AudioBuffer",
    "AudioQueue",
    # Audio Component & AudioUnit Framework
    "AudioComponentDescription",
    "AudioComponent",
    "AudioUnit",
    # MIDI Framework
    "MIDIClient",
    "MIDIPort",
    "MIDIInputPort",
    "MIDIOutputPort",
    # MusicPlayer Framework
    "MusicPlayer",
    "MusicSequence",
    "MusicTrack",
    # Audio Device & Hardware
    "AudioDevice",
    "AudioDeviceManager",
    # AUGraph Framework
    "AUGraph",
    # CoreAudioClock - Synchronization and Timing
    "AudioClock",
    "ClockTimeFormat",
    # Audio Player
    "AudioPlayer",
    # Async I/O Support
    "AsyncAudioFile",
    "AsyncAudioQueue",
    "open_audio_file_async",
    "create_output_queue_async",
    # Ableton Link - Tempo Synchronization
    "link",  # link module containing LinkSession, SessionState, Clock
    "link_midi",  # Link + MIDI integration (LinkMIDIClock, LinkMIDISequencer)
    # AudioUnit Plugin Hosting
    "AudioUnitHost",
    "AudioUnitPlugin",
    "AudioUnitParameter",
    "AudioUnitPreset",
    "PluginAudioFormat",
    "AudioFormatConverter",
    "AudioUnitChain",
    "PresetManager",
    # NumPy availability flag
    "NUMPY_AVAILABLE",
    # Error handling decorators
    "check_os_status",
    "check_return_status",
    "raises_on_error",
    "handle_exceptions",
    "format_os_status_error",
    # Buffer management
    "buffer_utils",  # buffer_utils module
    "AudioStreamBasicDescription",
    "pack_audio_buffer",
    "unpack_audio_buffer",
    "calculate_buffer_size",
    "optimal_buffer_size",
    # Constants module and enum classes (preferred over capi getter functions)
    "constants",  # constants module with all enum classes
    "AudioFileProperty",
    "AudioFileType",
    "AudioFilePermission",
    "AudioFormatID",
    "LinearPCMFormatFlag",
    "AudioConverterProperty",
    "AudioConverterQuality",
    "ExtendedAudioFileProperty",
    "AudioUnitProperty",
    "AudioUnitScope",
    "AudioUnitElement",
    "AudioUnitRenderActionFlags",
    "AudioUnitParameterUnit",
    "AudioQueueProperty",
    "AudioQueueParameter",
    "AudioObjectProperty",
    "AudioDeviceProperty",
    "MIDIStatus",
    "MIDIControlChange",
    "MIDIObjectProperty",
    # Music theory
    "music",  # music module with theory submodule
    # Submodules (also exported for attribute access)
    "audio",  # audio subpackage
    "midi",  # midi subpackage
    "utils",  # utils subpackage
    "os_status",  # os_status module
]
