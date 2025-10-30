Import Guide
============

**Version:** 0.1.8

Complete guide to importing modules and classes from CoreMusic.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Start
-----------

Most Common Imports
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Main package - Object-Oriented API
   import coremusic as cm

   # All high-level classes available directly
   player = cm.AudioPlayer()
   file = cm.AudioFile("audio.wav")
   sequence = cm.MusicSequence()

   # Functional C API (for performance)
   import coremusic.capi as capi
   file_id = capi.audio_file_open_url("audio.wav")

   # NumPy integration
   import numpy as np
   import coremusic as cm

Hierarchical Package Structure
-------------------------------

CoreMusic uses a hierarchical package structure for better organization::

   coremusic/
   ├── __init__.py          # Main package (OO API)
   ├── capi.pyx            # Functional C API
   ├── objects.py          # OO wrappers
   ├── constants.py        # Constants and enums
   ├── os_status.py        # Error handling
   │
   ├── audio/              # Audio subpackage
   │   ├── __init__.py
   │   ├── async_io.py     # Async audio I/O
   │   ├── utilities.py    # Audio utilities
   │   ├── slicing.py      # Audio slicing
   │   ├── analysis.py     # Audio analysis
   │   ├── visualization.py # Audio visualization
   │   └── audiounit_host.py # AudioUnit hosting
   │
   ├── midi/               # MIDI subpackage
   │   ├── __init__.py
   │   ├── link.py         # Link + MIDI integration
   │   └── utilities.py    # MIDI utilities
   │
   ├── utils/              # Utilities subpackage
   │   ├── __init__.py
   │   ├── scipy.py        # SciPy integration
   │   └── fourcc.py       # FourCC utilities
   │
   ├── daw.py              # DAW building blocks
   └── link.py             # Ableton Link

Import Patterns
---------------

Main Package Imports
^^^^^^^^^^^^^^^^^^^^

**Object-Oriented API (Recommended):**

.. code-block:: python

   import coremusic as cm

   # Audio file operations
   audio = cm.AudioFile("audio.wav")
   ext_audio = cm.ExtendedAudioFile("audio.mp3")

   # Audio processing
   queue = cm.AudioQueue.create_output(format)
   unit = cm.AudioUnit.default_output()
   converter = cm.AudioConverter()

   # MIDI and music
   player = cm.MusicPlayer()
   sequence = cm.MusicSequence()
   track = sequence.new_track()

   # Hardware
   device = cm.AudioDevice.get_default_output_device()
   client = cm.MIDIClient("MyApp")

   # Utilities
   player = cm.AudioPlayer("song.wav")

**Functional C API (For Performance):**

.. code-block:: python

   import coremusic.capi as capi

   # Direct C function calls
   file_id = capi.audio_file_open_url("audio.wav")
   format = capi.audio_file_get_property(file_id, property_id)
   data, count = capi.audio_file_read_packets(file_id, 0, 1024)
   capi.audio_file_close(file_id)

   # Constants
   property_id = capi.get_audio_file_property_data_format()
   format_id = capi.fourchar_to_int('lpcm')

Audio Subpackage Imports
^^^^^^^^^^^^^^^^^^^^^^^^^

**Async I/O:**

.. code-block:: python

   # New hierarchical import (recommended)
   from coremusic.audio import AsyncAudioFile, AsyncAudioQueue

   # Or
   from coremusic.audio.async_io import AsyncAudioFile

   # Backward compatible (still works)
   from coremusic import AsyncAudioFile

**Audio Analysis:**

.. code-block:: python

   from coremusic.audio.analysis import (
       AudioAnalyzer,
       LivePitchDetector,
       BeatInfo,
       PitchInfo
   )

   # Usage
   analyzer = AudioAnalyzer("song.wav")
   beats = analyzer.detect_beats()

**Audio Slicing:**

.. code-block:: python

   from coremusic.audio.slicing import AudioSlicer

   slicer = AudioSlicer("audio.wav")
   slice_data = slicer.slice_time_range(0.0, 10.0)

**Audio Visualization:**

.. code-block:: python

   from coremusic.audio.visualization import (
       WaveformPlotter,
       SpectrogramPlotter,
       FrequencySpectrumPlotter
   )

   plotter = WaveformPlotter("audio.wav")
   plotter.plot()

**AudioUnit Hosting:**

.. code-block:: python

   from coremusic.audio.audiounit_host import (
       AudioUnitHost,
       AudioUnitPlugin,
       AudioUnitParameter,
       AudioUnitPreset,
       AudioUnitChain,
       PresetManager
   )

   host = AudioUnitHost()
   plugin = host.load_plugin("AUReverb")

MIDI Subpackage Imports
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from coremusic.midi import (
       MIDISequence,
       MIDITrack,
       MIDINote,
       load_midi_file,
       save_midi_file,
       create_midi_message,
       parse_midi_message
   )

   # Load MIDI file
   midi = load_midi_file("song.mid")
   print(f"Tempo: {midi.tempo} BPM")

**Link + MIDI Integration:**

.. code-block:: python

   from coremusic.midi.link import (
       LinkMIDIClock,
       LinkMIDISequencer
   )

   # Or backward compatible
   from coremusic import link_midi
   clock = link_midi.LinkMIDIClock()

Utils Subpackage Imports
^^^^^^^^^^^^^^^^^^^^^^^^^

**SciPy Integration:**

.. code-block:: python

   from coremusic.utils.scipy import (
       audio_to_scipy,
       scipy_to_audio,
       apply_scipy_filter,
       resample_scipy
   )

   # Or
   import coremusic.utils.scipy as spu
   audio_array = spu.audio_to_scipy("audio.wav")

**FourCC Utilities:**

.. code-block:: python

   from coremusic.utils.fourcc import (
       fourcc_to_string,
       string_to_fourcc,
       is_valid_fourcc
   )

   # These are also available from capi
   from coremusic import capi
   fourcc = capi.fourchar_to_int('lpcm')
   string = capi.int_to_fourchar(fourcc)

DAW Subpackage Imports
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from coremusic.daw import (
       Timeline,
       Track,
       Clip,
       MIDIClip,
       MIDINote,
       TimelineMarker,
       TimeRange,
       AutomationLane
   )

   # Create timeline
   timeline = Timeline(tempo=120.0, sample_rate=44100.0)
   track = Track(name="Audio")
   timeline.add_track(track)

Link Subpackage Imports
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from coremusic import link

   # Create Link session
   session = link.LinkSession()
   session.enable(True)

   # Access clock
   clock = link.Clock()
   micros = clock.micros()

   # Session state
   state = session.capture_app_session_state()
   tempo = state.tempo

Backward Compatibility
----------------------

CoreMusic maintains full backward compatibility with pre-0.1.8 import patterns.

Old Import Pattern (Still Works)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # These old imports still work
   from coremusic import AsyncAudioFile      # ✅ Still works
   from coremusic import link_midi           # ✅ Still works
   from coremusic import AudioAnalyzer       # ✅ Still works

New Import Pattern (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # New hierarchical imports (preferred)
   from coremusic.audio import AsyncAudioFile
   from coremusic.midi import link as link_midi
   from coremusic.audio.analysis import AudioAnalyzer

Why Use New Imports?
^^^^^^^^^^^^^^^^^^^^^

1. **Clearer organization** - Know where functionality lives
2. **Better IDE support** - More accurate autocompletion
3. **Faster imports** - Only load what you need
4. **Future-proof** - Aligned with package structure

Complete Import Reference
--------------------------

Core Classes (coremusic.*)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   # Base
   cm.CoreAudioObject

   # Exceptions
   cm.CoreAudioError
   cm.AudioFileError
   cm.AudioQueueError
   cm.AudioUnitError
   cm.AudioConverterError
   cm.MIDIError
   cm.MusicPlayerError
   cm.AudioDeviceError
   cm.AUGraphError

   # Audio Format
   cm.AudioFormat

   # Audio File Framework
   cm.AudioFile
   cm.AudioFileStream
   cm.ExtendedAudioFile

   # Audio Converter
   cm.AudioConverter

   # Audio Queue
   cm.AudioBuffer
   cm.AudioQueue

   # AudioUnit Framework
   cm.AudioComponentDescription
   cm.AudioComponent
   cm.AudioUnit

   # MIDI Framework
   cm.MIDIClient
   cm.MIDIPort
   cm.MIDIInputPort
   cm.MIDIOutputPort

   # Music Player Framework
   cm.MusicPlayer
   cm.MusicSequence
   cm.MusicTrack

   # Audio Device
   cm.AudioDevice
   cm.AudioDeviceManager

   # AUGraph
   cm.AUGraph

   # Audio Clock
   cm.AudioClock
   cm.ClockTimeFormat

   # Audio Player
   cm.AudioPlayer

Functional API (coremusic.capi.*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic.capi as capi

   # All C functions available via capi module
   # Examples:
   capi.audio_file_open_url()
   capi.audio_queue_new_output()
   capi.audio_unit_initialize()
   capi.midi_client_create()
   capi.new_music_player()

   # FourCC utilities
   capi.fourchar_to_int()
   capi.int_to_fourchar()

   # Constants (get_* functions)
   capi.get_audio_file_property_data_format()
   capi.get_audio_unit_property_stream_format()
   # ... hundreds of constants

Audio Subpackage (coremusic.audio.*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from coremusic.audio import (
       # Async I/O
       AsyncAudioFile,
       AsyncAudioQueue,

       # Utilities
       load_audio_file_async,
       create_output_queue_async,
       resample_audio,
       convert_audio_format,

       # Analysis
       AudioAnalyzer,
       LivePitchDetector,
       BeatInfo,
       PitchInfo,

       # Slicing
       AudioSlicer,

       # Visualization
       WaveformPlotter,
       SpectrogramPlotter,
       FrequencySpectrumPlotter,

       # AudioUnit Hosting
       AudioUnitHost,
       AudioUnitPlugin,
       AudioUnitParameter,
       AudioUnitPreset,
       AudioUnitChain,
       PresetManager,
       PluginAudioFormat,
       AudioFormatConverter,
   )

MIDI Subpackage (coremusic.midi.*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from coremusic.midi import (
       # MIDI Utilities
       MIDISequence,
       MIDITrack,
       MIDINote,
       load_midi_file,
       save_midi_file,
       create_midi_message,
       parse_midi_message,
       midi_note_to_name,
       midi_name_to_note,
       transpose_notes,

       # Link Integration
       link,  # Link + MIDI submodule
   )

   from coremusic.midi.link import (
       LinkMIDIClock,
       LinkMIDISequencer,
   )

Utils Subpackage (coremusic.utils.*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from coremusic.utils import (
       # SciPy integration
       scipy,

       # FourCC utilities
       fourcc,
   )

   from coremusic.utils.scipy import (
       audio_to_scipy,
       scipy_to_audio,
       apply_scipy_filter,
       resample_scipy,
   )

   from coremusic.utils.fourcc import (
       fourcc_to_string,
       string_to_fourcc,
       is_valid_fourcc,
   )

DAW Subpackage (coremusic.daw.*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from coremusic.daw import (
       Timeline,
       Track,
       Clip,
       MIDIClip,
       MIDINote,
       TimelineMarker,
       TimeRange,
       AutomationLane,
       AudioUnitPlugin,  # Re-exported from audiounit_host
   )

Link Subpackage (coremusic.link.*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from coremusic import link

   # Link classes
   link.LinkSession
   link.SessionState
   link.Clock

   # Or direct import
   from coremusic.link import LinkSession, SessionState, Clock

Import Best Practices
----------------------

✅ DO: Use Hierarchical Imports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Good: Clear and explicit
   from coremusic.audio.analysis import AudioAnalyzer
   from coremusic.midi import load_midi_file
   from coremusic.daw import Timeline

   # Also good: Import submodule
   from coremusic import audio
   analyzer = audio.analysis.AudioAnalyzer("song.wav")

✅ DO: Use Aliases for Convenience
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Good: Short, clear aliases
   import coremusic as cm
   import coremusic.capi as capi
   import coremusic.utils.scipy as spu
   from coremusic.audio import analysis as audio_analysis

✅ DO: Import Only What You Need
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Good: Specific imports
   from coremusic import AudioFile, AudioQueue
   from coremusic.audio.analysis import AudioAnalyzer

   # Avoid: Importing everything
   # from coremusic import *  # Don't do this

❌ DON'T: Use Wildcard Imports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Bad: Pollutes namespace
   from coremusic import *
   from coremusic.audio import *

   # Good: Be explicit
   import coremusic as cm
   from coremusic.audio import AsyncAudioFile

✅ DO: Group Imports Logically
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Standard library
   import time
   from pathlib import Path

   # Third-party
   import numpy as np

   # CoreMusic - main package
   import coremusic as cm

   # CoreMusic - subpackages
   from coremusic.audio import AsyncAudioFile
   from coremusic.midi import load_midi_file
   from coremusic.daw import Timeline

Import Examples by Use Case
----------------------------

Basic Audio File Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   import numpy as np

   with cm.AudioFile("audio.wav") as audio:
       data, count = audio.read(audio.frame_count)
       samples = np.frombuffer(data, dtype=np.float32)

Real-Time Audio
^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   from coremusic.audio import AsyncAudioQueue

   # Async approach
   queue = AsyncAudioQueue.create_output(format)
   await queue.start_async()

MIDI Composition
^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   from coremusic.midi import MIDINote

   player = cm.MusicPlayer()
   sequence = cm.MusicSequence()
   track = sequence.new_track()
   track.add_midi_note(0.0, 0, 60, 100)

Audio Analysis
^^^^^^^^^^^^^^

.. code-block:: python

   from coremusic.audio.analysis import AudioAnalyzer

   analyzer = AudioAnalyzer("song.wav")
   analyzer.load_audio()
   beats = analyzer.detect_beats()
   pitch = analyzer.detect_pitch()

DAW Development
^^^^^^^^^^^^^^^

.. code-block:: python

   from coremusic.daw import Timeline, Track, Clip

   timeline = Timeline(tempo=120.0)
   track = Track(name="Audio")
   clip = Clip(file_path="audio.wav", start_time=0.0)
   track.add_clip(clip)
   timeline.add_track(track)

Link Synchronization
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from coremusic import link

   session = link.LinkSession()
   session.enable(True)
   state = session.capture_app_session_state()
   tempo = state.tempo

Type Hints and IDE Support
---------------------------

CoreMusic includes comprehensive type hints for better IDE support:

.. code-block:: python

   import coremusic as cm
   from typing import Optional, List

   # IDE will show type hints
   audio: cm.AudioFile = cm.AudioFile("audio.wav")
   format: cm.AudioFormat = audio.format
   duration: float = audio.duration

   # Function signatures include type hints
   def process_audio(
       file_path: str,
       output_path: str,
       gain: float = 1.0
   ) -> Optional[cm.AudioFile]:
       """Process audio with type hints"""
       pass

Troubleshooting Imports
------------------------

ModuleNotFoundError
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Error: ModuleNotFoundError: No module named 'coremusic'

   # Solution: Install coremusic
   # pip install coremusic
   # or
   # uv pip install coremusic

ImportError for Submodule
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Error: ImportError: cannot import name 'AsyncAudioFile'

   # Check if using correct import path
   from coremusic.audio import AsyncAudioFile  # ✅ Correct
   from coremusic.audio.async_io import AsyncAudioFile  # ✅ Also correct
   from coremusic import AsyncAudioFile  # ✅ Backward compatible

Circular Import
^^^^^^^^^^^^^^^

.. code-block:: python

   # If you get circular import errors, try:

   # Instead of:
   import coremusic
   from coremusic import AudioFile

   # Do:
   import coremusic as cm
   audio = cm.AudioFile("audio.wav")

Quick Reference Card
--------------------

.. code-block:: python

   # === Core Package ===
   import coremusic as cm                    # Main OO API
   import coremusic.capi as capi             # Functional C API

   # === Audio Subpackage ===
   from coremusic.audio import (
       AsyncAudioFile,                       # Async file I/O
       AudioAnalyzer,                        # Audio analysis
       AudioSlicer,                          # Audio slicing
       WaveformPlotter,                      # Visualization
       AudioUnitHost,                        # AudioUnit hosting
   )

   # === MIDI Subpackage ===
   from coremusic.midi import (
       load_midi_file,                       # MIDI file utilities
       MIDISequence,                         # MIDI data structures
       link,                                 # Link + MIDI
   )

   # === Utils Subpackage ===
   from coremusic.utils import scipy, fourcc

   # === DAW Subpackage ===
   from coremusic.daw import Timeline, Track, Clip

   # === Link Subpackage ===
   from coremusic import link
   session = link.LinkSession()

See Also
--------

- Use ``help(cm.AudioFile)`` in Python for API reference
- See ``tests/demos/`` directory for examples
- See full documentation at :doc:`/index`

.. note::
   **Migration from Old Imports:**
   All old import patterns continue to work. Update at your convenience!
