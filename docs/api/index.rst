API Reference
=============

Complete API reference for coremusic. The package provides both functional and object-oriented APIs.

.. note::
   The object-oriented API is recommended for new applications due to automatic resource management and Pythonic interfaces.

Core Modules
------------

.. toctree::
   :maxdepth: 2

   audio_file
   audio_unit
   audio_queue
   audio_converter
   midi
   utilities

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

Functional API
--------------

Low-level C-style functions providing direct access to CoreAudio frameworks.

Audio File Functions
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: coremusic.audio_file_open_url
.. autofunction:: coremusic.audio_file_close
.. autofunction:: coremusic.audio_file_read_packets
.. autofunction:: coremusic.audio_file_get_property

AudioUnit Functions
^^^^^^^^^^^^^^^^^^^

.. autofunction:: coremusic.audio_component_find_next
.. autofunction:: coremusic.audio_component_instance_new
.. autofunction:: coremusic.audio_component_instance_dispose
.. autofunction:: coremusic.audio_unit_initialize
.. autofunction:: coremusic.audio_unit_uninitialize
.. autofunction:: coremusic.audio_unit_set_property
.. autofunction:: coremusic.audio_unit_get_property

AudioQueue Functions
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: coremusic.audio_queue_new_output
.. autofunction:: coremusic.audio_queue_dispose
.. autofunction:: coremusic.audio_queue_start
.. autofunction:: coremusic.audio_queue_stop
.. autofunction:: coremusic.audio_queue_allocate_buffer

MIDI Functions
^^^^^^^^^^^^^^

.. autofunction:: coremusic.midi_client_create
.. autofunction:: coremusic.midi_client_dispose
.. autofunction:: coremusic.midi_input_port_create
.. autofunction:: coremusic.midi_output_port_create
.. autofunction:: coremusic.midi_send

Constants and Enumerations
---------------------------

AudioFormat Constants
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: coremusic.get_audio_format_linear_pcm
.. autofunction:: coremusic.get_audio_format_mpeg4aac
.. autofunction:: coremusic.get_audio_format_apple_lossless

AudioUnit Types
^^^^^^^^^^^^^^^

.. autofunction:: coremusic.get_audio_unit_type_output
.. autofunction:: coremusic.get_audio_unit_type_effect
.. autofunction:: coremusic.get_audio_unit_type_generator

Error Handling
--------------

Exception Classes
^^^^^^^^^^^^^^^^^

.. autoexception:: coremusic.AudioFileError
.. autoexception:: coremusic.AudioUnitError
.. autoexception:: coremusic.AudioQueueError
.. autoexception:: coremusic.MIDIError

Utility Functions
-----------------

FourCC Conversion
^^^^^^^^^^^^^^^^^

.. autofunction:: coremusic.fourchar_to_int
.. autofunction:: coremusic.int_to_fourchar

Type Conversion
^^^^^^^^^^^^^^^

.. autofunction:: coremusic.asbd_to_dict
.. autofunction:: coremusic.dict_to_asbd
