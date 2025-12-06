API Quickstart
==============

A rapid introduction to coremusic's most commonly used APIs.

Import Patterns
---------------

.. code-block:: python

   # Main package - object-oriented API (recommended)
   import coremusic as cm

   # Low-level functional API
   import coremusic.capi as capi

   # Constants (preferred over capi getter functions)
   from coremusic.constants import AudioFileProperty, AudioFormatID

   # Optional integrations
   import coremusic.utils.scipy as spu  # SciPy integration (requires scipy)

Audio File Operations
---------------------

Read Audio File
^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   # Context manager (recommended)
   with cm.AudioFile("audio.wav") as audio:
       print(f"Duration: {audio.duration:.2f}s")
       print(f"Sample rate: {audio.format.sample_rate}Hz")
       data, count = audio.read_packets(0, 1024)

Get Audio Format
^^^^^^^^^^^^^^^^

.. code-block:: python

   with cm.AudioFile("audio.wav") as audio:
       fmt = audio.format
       print(f"Format ID: {fmt.format_id}")           # 'lpcm'
       print(f"Sample rate: {fmt.sample_rate}")       # 44100.0
       print(f"Channels: {fmt.channels_per_frame}")   # 2
       print(f"Bits: {fmt.bits_per_channel}")         # 16

Extended Audio File (Format Conversion)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   with cm.ExtendedAudioFile("input.mp3") as ext_audio:
       # Set client format for automatic conversion
       ext_audio.client_format = cm.AudioFormat(
           sample_rate=48000.0,
           format_id='lpcm',
           channels_per_frame=2
       )
       # Read converted data
       data, count = ext_audio.read(8192)

AudioUnit Operations
--------------------

Create Default Output
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   with cm.AudioUnit.default_output() as unit:
       # Set format
       format = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2)
       unit.set_stream_format(format)

       # Start audio processing
       unit.start()
       # ... audio flows ...
       unit.stop()

Find and Create AudioUnit
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Create component description
   desc = cm.AudioComponentDescription(
       component_type='aufx',      # Effect
       component_subtype='dely',   # Delay
       component_manufacturer='appl'
   )

   # Find component
   component = cm.AudioComponent.find_first(desc)
   if component:
       print(f"Found: {component.name}")

       # Create instance
       with component.create_instance() as unit:
           unit.initialize()
           # ... use the unit ...

MIDI Operations
---------------

List MIDI Devices
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic.capi as capi

   # Count devices
   num_devices = capi.midi_get_number_of_devices()
   num_sources = capi.midi_get_number_of_sources()
   num_destinations = capi.midi_get_number_of_destinations()

   print(f"Devices: {num_devices}")
   print(f"Sources: {num_sources}")
   print(f"Destinations: {num_destinations}")

Create MIDI Client
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   client = cm.MIDIClient("My App")
   try:
       # Create ports
       output_port = client.create_output_port("Output")
       input_port = client.create_input_port("Input")

       # Use ports...
   finally:
       client.dispose()

Audio Queue Operations
----------------------

Create Output Queue
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   format = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2)

   with cm.AudioQueue.new_output(format) as queue:
       # Allocate buffer
       buffer = queue.allocate_buffer(4096)

       # Start playback
       queue.start()

       # ... fill buffer and enqueue ...

       queue.stop()

Constants Usage
---------------

Using Enum Constants
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from coremusic import (
       AudioFileProperty,
       AudioFormatID,
       AudioUnitProperty,
       AudioUnitScope,
   )

   # Audio file properties
   prop_id = AudioFileProperty.DATA_FORMAT          # 1684434292
   prop_id = AudioFileProperty.ESTIMATED_DURATION   # 1701082482

   # Audio format IDs
   fmt_id = AudioFormatID.LINEAR_PCM   # 1819304813
   fmt_id = AudioFormatID.AAC          # 1633772320

   # AudioUnit properties
   au_prop = AudioUnitProperty.STREAM_FORMAT    # 8
   au_prop = AudioUnitProperty.SAMPLE_RATE      # 2

   # AudioUnit scopes
   scope = AudioUnitScope.INPUT   # 1
   scope = AudioUnitScope.OUTPUT  # 2

Constants in API Calls
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   import coremusic.capi as capi

   # Use constant enum in functional API
   file_id = capi.audio_file_open_url("audio.wav")
   format_data = capi.audio_file_get_property(
       file_id,
       int(cm.AudioFileProperty.DATA_FORMAT)  # Convert to int
   )
   capi.audio_file_close(file_id)

Async Operations
----------------

Async File Reading
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import asyncio
   import coremusic as cm

   async def read_async():
       async with cm.AsyncAudioFile("audio.wav") as audio:
           print(f"Duration: {audio.duration:.2f}s")

           # Stream chunks
           async for chunk in audio.read_chunks_async(chunk_size=4096):
               # Process chunk
               pass

   asyncio.run(read_async())

Error Handling
--------------

Exception Hierarchy
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   try:
       with cm.AudioFile("missing.wav") as audio:
           pass
   except cm.AudioFileError as e:
       print(f"Audio file error: {e}")
   except cm.CoreAudioError as e:
       print(f"CoreAudio error: {e}")

   # Specific exception types:
   # - cm.AudioFileError
   # - cm.AudioQueueError
   # - cm.AudioUnitError
   # - cm.AudioConverterError
   # - cm.MIDIError
   # - cm.MusicPlayerError
   # - cm.AudioDeviceError
   # - cm.AUGraphError

NumPy Integration
-----------------

Check Availability
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   if cm.NUMPY_AVAILABLE:
       import numpy as np

       with cm.AudioFile("audio.wav") as audio:
           # Get NumPy dtype
           dtype = audio.format.to_numpy_dtype()

           # Read and convert
           data, count = audio.read_packets(0, 1024)
           samples = np.frombuffer(data, dtype=dtype)

Memory-Mapped Files
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from coremusic.audio import MMapAudioFile

   with MMapAudioFile("large.wav") as mmap:
       # Fast random access
       chunk = mmap[1000:2000]  # Read frames 1000-2000

       # Zero-copy NumPy array
       audio_np = mmap.read_as_numpy(start_frame=0, num_frames=44100)

Quick Reference Table
---------------------

.. list-table:: Common Classes
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Purpose
   * - ``cm.AudioFile``
     - Read audio files (WAV, AIFF, MP3, etc.)
   * - ``cm.ExtendedAudioFile``
     - Read with format conversion
   * - ``cm.AudioFormat``
     - Audio format description
   * - ``cm.AudioUnit``
     - Audio processing unit
   * - ``cm.AudioQueue``
     - Audio playback/recording queue
   * - ``cm.AudioConverter``
     - Convert between formats
   * - ``cm.MIDIClient``
     - MIDI client connection
   * - ``cm.AudioPlayer``
     - High-level audio playback
   * - ``cm.AsyncAudioFile``
     - Async file operations
   * - ``cm.AsyncAudioQueue``
     - Async queue operations

See Also
--------

- :doc:`index` - Full API reference
- :doc:`../getting_started` - Complete getting started guide
- :doc:`../tutorials/index` - Step-by-step tutorials
