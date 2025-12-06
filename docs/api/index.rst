API Reference
=============

Complete API reference for coremusic. The package provides both functional and object-oriented APIs.

.. note::
   The object-oriented API is recommended for new applications due to automatic resource management and Pythonic interfaces.

.. toctree::
   :maxdepth: 2
   :caption: Quick Start

   quickstart

Core Modules
------------

.. toctree::
   :maxdepth: 2

   audio_file

Object-Oriented API
-------------------

High-level Pythonic wrappers with automatic resource management.

AudioFile Class
^^^^^^^^^^^^^^^

.. autoclass:: coremusic.AudioFile
   :members:
   :undoc-members:
   :show-inheritance:

AudioFormat Class
^^^^^^^^^^^^^^^^^

.. autoclass:: coremusic.AudioFormat
   :members:
   :undoc-members:
   :show-inheritance:

AudioUnit Class
^^^^^^^^^^^^^^^

.. autoclass:: coremusic.AudioUnit
   :members:
   :undoc-members:
   :show-inheritance:

AudioQueue Class
^^^^^^^^^^^^^^^^

.. autoclass:: coremusic.AudioQueue
   :members:
   :undoc-members:
   :show-inheritance:

AudioConverter Class
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: coremusic.AudioConverter
   :members:
   :undoc-members:
   :show-inheritance:

MIDIClient Class
^^^^^^^^^^^^^^^^

.. autoclass:: coremusic.MIDIClient
   :members:
   :undoc-members:
   :show-inheritance:

AudioClock Class
^^^^^^^^^^^^^^^^

.. autoclass:: coremusic.AudioClock
   :members:
   :undoc-members:
   :show-inheritance:

ClockTimeFormat
^^^^^^^^^^^^^^^

.. autoclass:: coremusic.ClockTimeFormat
   :members:
   :undoc-members:
   :show-inheritance:

AudioUnit Plugin Hosting
^^^^^^^^^^^^^^^^^^^^^^^^^

AudioUnitHost Class
"""""""""""""""""""

.. autoclass:: coremusic.AudioUnitHost
   :members:
   :undoc-members:
   :show-inheritance:

AudioUnitPlugin Class
"""""""""""""""""""""

.. autoclass:: coremusic.AudioUnitPlugin
   :members:
   :undoc-members:
   :show-inheritance:

AudioUnitParameter Class
"""""""""""""""""""""""""

.. autoclass:: coremusic.AudioUnitParameter
   :members:
   :undoc-members:
   :show-inheritance:

AudioUnitPreset Class
"""""""""""""""""""""

.. autoclass:: coremusic.AudioUnitPreset
   :members:
   :undoc-members:
   :show-inheritance:

PluginAudioFormat Class
"""""""""""""""""""""""

.. autoclass:: coremusic.PluginAudioFormat
   :members:
   :undoc-members:
   :show-inheritance:

AudioFormatConverter Class
"""""""""""""""""""""""""""

.. autoclass:: coremusic.AudioFormatConverter
   :members:
   :undoc-members:
   :show-inheritance:

PresetManager Class
"""""""""""""""""""

.. autoclass:: coremusic.PresetManager
   :members:
   :undoc-members:
   :show-inheritance:

AudioUnitChain Class
""""""""""""""""""""

.. autoclass:: coremusic.AudioUnitChain
   :members:
   :undoc-members:
   :show-inheritance:

Functional API
--------------

Low-level C-style functions are available through the ``coremusic.capi`` module
for advanced use cases requiring direct access to CoreAudio frameworks.

.. note::
   The object-oriented API is recommended for most use cases. The functional
   API in ``coremusic.capi`` provides low-level access when needed.

For direct access to low-level functions::

    import coremusic.capi as capi

    # Low-level audio file operations
    file_id = capi.audio_file_open_url("audio.wav")
    # ... operations ...
    capi.audio_file_close(file_id)

    # Low-level clock operations
    clock_id = capi.ca_clock_new()
    capi.ca_clock_start(clock_id)
    # ... operations ...
    capi.ca_clock_dispose(clock_id)

Error Handling
--------------

coremusic provides exception classes for different CoreAudio subsystems:

.. autoexception:: coremusic.CoreAudioError
   :members:
   :show-inheritance:

.. autoexception:: coremusic.AudioFileError
   :members:
   :show-inheritance:

.. autoexception:: coremusic.AudioUnitError
   :members:
   :show-inheritance:

.. autoexception:: coremusic.AudioQueueError
   :members:
   :show-inheritance:

.. autoexception:: coremusic.AudioConverterError
   :members:
   :show-inheritance:

.. autoexception:: coremusic.MIDIError
   :members:
   :show-inheritance:

.. autoexception:: coremusic.MusicPlayerError
   :members:
   :show-inheritance:

.. autoexception:: coremusic.AudioDeviceError
   :members:
   :show-inheritance:

.. autoexception:: coremusic.AUGraphError
   :members:
   :show-inheritance:

Utility Functions
-----------------

Utility functions are available through ``coremusic.capi`` for FourCC conversion
and other low-level operations::

    import coremusic.capi as capi

    # Convert FourCC string to integer
    format_int = capi.fourchar_to_int('lpcm')

    # Convert integer back to FourCC string
    format_str = capi.int_to_fourchar(format_int)
