Audio File Operations
=====================

The audio file module provides functionality for reading and writing audio files in various formats.

Object-Oriented API
-------------------

AudioFile Class
^^^^^^^^^^^^^^^

The ``AudioFile`` class provides high-level audio file operations with automatic resource management.

.. code-block:: python

   import coremusic as cm

   # Context manager usage (recommended)
   with cm.AudioFile("audio.wav") as audio:
       print(f"Duration: {audio.duration:.2f}s")
       data, count = audio.read_packets(0, 1000)

   # Explicit management
   audio = cm.AudioFile("audio.wav")
   audio.open()
   try:
       data = audio.read_packets(0, 1000)
   finally:
       audio.close()

Class Reference
^^^^^^^^^^^^^^^

.. autoclass:: coremusic.AudioFile
   :members:
   :undoc-members:
   :special-members: __init__, __enter__, __exit__
   :show-inheritance:
   :noindex:

AudioFormat Class
^^^^^^^^^^^^^^^^^

The ``AudioFormat`` class represents audio stream format information.

.. code-block:: python

   import coremusic as cm

   # Access format from audio file
   with cm.AudioFile("audio.wav") as audio:
       fmt = audio.format
       print(f"Sample rate: {fmt.sample_rate}Hz")
       print(f"Channels: {fmt.channels_per_frame}")
       print(f"Bit depth: {fmt.bits_per_channel}")

   # Create custom format
   format = cm.AudioFormat(
       sample_rate=44100.0,
       format_id='lpcm',
       channels_per_frame=2,
       bits_per_channel=16
   )

Class Reference
^^^^^^^^^^^^^^^

.. autoclass:: coremusic.AudioFormat
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Functional API
--------------

The functional API provides direct access to CoreAudio file operations through
the ``coremusic.capi`` module.

.. note::
   The object-oriented ``AudioFile`` API is recommended for most use cases.
   Use the functional API only when you need fine-grained control.

Opening and Closing Files
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Example:**

.. code-block:: python

   import coremusic.capi as capi

   # Open audio file
   file_id = capi.audio_file_open_url("audio.wav")
   try:
       # Use file...
       pass
   finally:
       capi.audio_file_close(file_id)

Reading Audio Data
^^^^^^^^^^^^^^^^^^

**Example:**

.. code-block:: python

   import coremusic.capi as capi

   file_id = capi.audio_file_open_url("audio.wav")
   try:
       # Read 1000 packets starting from packet 0
       data, packets_read = capi.audio_file_read_packets(file_id, 0, 1000)
       print(f"Read {packets_read} packets, {len(data)} bytes")
   finally:
       capi.audio_file_close(file_id)

File Properties
^^^^^^^^^^^^^^^

**Example:**

.. code-block:: python

   import coremusic.capi as capi

   file_id = capi.audio_file_open_url("audio.wav")
   try:
       # Get audio format
       format_data = capi.audio_file_get_property(
           file_id,
           capi.get_audio_file_property_data_format()
       )
       print(f"Format: {format_data}")
   finally:
       capi.audio_file_close(file_id)

Supported Formats
-----------------

coremusic supports all audio formats supported by CoreAudio, including:

Common Formats
^^^^^^^^^^^^^^

- **WAV** (Waveform Audio File Format)
- **AIFF** (Audio Interchange File Format)
- **MP3** (MPEG-1 Audio Layer 3)
- **AAC** (Advanced Audio Coding)
- **ALAC** (Apple Lossless Audio Codec)
- **FLAC** (Free Lossless Audio Codec)

Format IDs
^^^^^^^^^^

Common format IDs (FourCC codes):

- ``'lpcm'`` - Linear PCM (uncompressed)
- ``'aac '`` - AAC
- ``'.mp3'`` - MP3
- ``'alac'`` - Apple Lossless
- ``'flac'`` - FLAC

Format Flags
^^^^^^^^^^^^

For Linear PCM, common format flags include:

- Float vs Integer
- Big Endian vs Little Endian
- Packed vs Aligned
- Signed vs Unsigned

Use the provided constant functions to get appropriate flags:

.. code-block:: python

   import coremusic.capi as capi

   # Get standard format flags
   flags = capi.get_audio_format_flag_is_float() | \\
           capi.get_audio_format_flag_is_packed()

Examples
--------

Read Entire Audio File
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def read_audio_file(filepath):
       """Read entire audio file into memory."""
       with cm.AudioFile(filepath) as audio:
           # Get total frame count
           total_frames = audio.frame_count

           # Read all data
           data, count = audio.read_packets(0, total_frames)

           return {
               'data': data,
               'sample_rate': audio.format.sample_rate,
               'channels': audio.format.channels_per_frame,
               'format': audio.format.format_id
           }

   # Use the function
   audio_data = read_audio_file("audio.wav")
   print(f"Loaded {len(audio_data['data'])} bytes")

Process Audio in Chunks
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def process_audio_chunks(filepath, chunk_size=1024):
       """Process audio file in chunks."""
       with cm.AudioFile(filepath) as audio:
           total_frames = audio.frame_count
           current_frame = 0

           while current_frame < total_frames:
               # Calculate chunk size
               frames_to_read = min(chunk_size, total_frames - current_frame)

               # Read chunk
               data, count = audio.read_packets(current_frame, frames_to_read)

               # Process chunk
               process_audio_data(data)

               current_frame += count

   def process_audio_data(data):
       """Process audio data chunk."""
       # Your processing logic here
       pass

Audio Format Conversion
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def convert_audio_format(input_path, output_path, target_format):
       """Convert audio file to different format."""
       # Open input file
       with cm.AudioFile(input_path) as input_audio:
           # Create converter
           converter = cm.AudioConverter(input_audio.format, target_format)

           # Read and convert
           data, count = input_audio.read_packets(0, input_audio.frame_count)
           converted_data = converter.convert(data, count)

           # Write to output file
           # (implementation depends on output requirements)

See Also
--------

- :doc:`../cookbook/file_operations` - Common file operation recipes
