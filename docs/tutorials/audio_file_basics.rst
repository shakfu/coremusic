Audio File Basics
=================

This tutorial covers the fundamentals of working with audio files using coremusic.

Prerequisites
-------------

- coremusic installed and built
- Basic Python knowledge
- An audio file to work with (WAV, AIFF, or MP3)

Opening and Reading Files
--------------------------

Using the Object-Oriented API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The recommended approach uses the ``AudioFile`` class with context managers:

.. code-block:: python

   import coremusic as cm

   # Open with context manager (automatic cleanup)
   with cm.AudioFile("audio.wav") as audio:
       # File is automatically opened
       print(f"Opened: {audio.path}")

   # File is automatically closed here

**Advantages:**

- Automatic resource cleanup
- Exception-safe
- Pythonic and readable

Using the Functional API
^^^^^^^^^^^^^^^^^^^^^^^^^

For more control, use the functional API:

.. code-block:: python

   import coremusic as cm

   # Open file manually
   file_id = cm.audio_file_open_url("audio.wav")
   try:
       # Work with file
       print(f"Opened file: {file_id}")
   finally:
       # Always close, even on error
       cm.audio_file_close(file_id)

Getting File Information
-------------------------

Duration and Format
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   with cm.AudioFile("audio.wav") as audio:
       # Basic information
       print(f"Duration: {audio.duration:.2f} seconds")
       print(f"Total frames: {audio.frame_count}")

       # Format details
       fmt = audio.format
       print(f"Sample rate: {fmt.sample_rate} Hz")
       print(f"Channels: {fmt.channels_per_frame}")
       print(f"Bit depth: {fmt.bits_per_channel}")
       print(f"Format ID: {fmt.format_id}")

**Output example:**

.. code-block:: text

   Duration: 2.74 seconds
   Total frames: 120960
   Sample rate: 44100.0 Hz
   Channels: 2
   Bit depth: 16
   Format ID: lpcm

Detailed Format Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def display_audio_format(fmt):
       """Display detailed format information."""
       print(f"Format Information:")
       print(f"  Sample Rate: {fmt.sample_rate} Hz")
       print(f"  Format ID: {fmt.format_id}")
       print(f"  Channels: {fmt.channels_per_frame}")
       print(f"  Bits/Channel: {fmt.bits_per_channel}")
       print(f"  Bytes/Frame: {fmt.bytes_per_frame}")
       print(f"  Bytes/Packet: {fmt.bytes_per_packet}")
       print(f"  Frames/Packet: {fmt.frames_per_packet}")
       print(f"  Format Flags: 0x{fmt.format_flags:08X}")

   with cm.AudioFile("audio.wav") as audio:
       display_audio_format(audio.format)

Reading Audio Data
------------------

Reading Packets
^^^^^^^^^^^^^^^

Audio data is read in packets (frames):

.. code-block:: python

   import coremusic as cm

   with cm.AudioFile("audio.wav") as audio:
       # Read first 1000 packets
       data, packets_read = audio.read_packets(0, 1000)

       print(f"Read {packets_read} packets")
       print(f"Data size: {len(data)} bytes")

**Parameters:**

- ``start_packet``: Starting packet number (0-indexed)
- ``num_packets``: Number of packets to read

**Returns:**

- ``data``: Raw audio data as bytes
- ``packets_read``: Actual number of packets read

Reading in Chunks
^^^^^^^^^^^^^^^^^

For large files, read in chunks to manage memory:

.. code-block:: python

   import coremusic as cm

   def read_file_in_chunks(filepath, chunk_size=4096):
       """Read audio file in chunks."""
       with cm.AudioFile(filepath) as audio:
           total_frames = audio.frame_count
           current_frame = 0

           while current_frame < total_frames:
               # Calculate remaining frames
               remaining = total_frames - current_frame
               to_read = min(chunk_size, remaining)

               # Read chunk
               data, count = audio.read_packets(current_frame, to_read)

               # Process chunk
               yield data

               current_frame += count

   # Use the generator
   for chunk in read_file_in_chunks("large_audio.wav"):
       process_chunk(chunk)

Reading Entire File
^^^^^^^^^^^^^^^^^^^

For smaller files, read everything at once:

.. code-block:: python

   import coremusic as cm

   def load_audio_file(filepath):
       """Load entire audio file into memory."""
       with cm.AudioFile(filepath) as audio:
           # Read all frames
           data, count = audio.read_packets(0, audio.frame_count)

           return {
               'data': data,
               'sample_rate': audio.format.sample_rate,
               'channels': audio.format.channels_per_frame,
               'bits_per_channel': audio.format.bits_per_channel,
               'duration': audio.duration
           }

   # Load and use
   audio_data = load_audio_file("audio.wav")
   print(f"Loaded {len(audio_data['data'])} bytes")

Working with Different Formats
-------------------------------

Detecting Format Type
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def detect_audio_format(filepath):
       """Detect and classify audio format."""
       with cm.AudioFile(filepath) as audio:
           fmt = audio.format

           if fmt.format_id == 'lpcm':
               return 'Linear PCM (uncompressed)'
           elif fmt.format_id == 'aac ':
               return 'AAC (compressed)'
           elif fmt.format_id == '.mp3':
               return 'MP3 (compressed)'
           elif fmt.format_id == 'alac':
               return 'Apple Lossless (compressed)'
           else:
               return f'Unknown format: {fmt.format_id}'

   print(detect_audio_format("audio.wav"))  # Linear PCM (uncompressed)

Checking Format Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def check_format_properties(filepath):
       """Check various format properties."""
       with cm.AudioFile(filepath) as audio:
           fmt = audio.format

           # Check if PCM
           is_pcm = fmt.format_id == 'lpcm'

           # Check if stereo
           is_stereo = fmt.channels_per_frame == 2

           # Check if CD quality (44.1kHz, 16-bit stereo)
           is_cd_quality = (
               fmt.sample_rate == 44100.0 and
               fmt.bits_per_channel == 16 and
               fmt.channels_per_frame == 2
           )

           print(f"PCM: {is_pcm}")
           print(f"Stereo: {is_stereo}")
           print(f"CD Quality: {is_cd_quality}")

Error Handling
--------------

Handling File Errors
^^^^^^^^^^^^^^^^^^^^

Always handle potential errors:

.. code-block:: python

   import coremusic as cm
   from pathlib import Path

   def safe_open_audio_file(filepath):
       """Safely open audio file with error handling."""
       # Check if file exists
       if not Path(filepath).exists():
           raise FileNotFoundError(f"File not found: {filepath}")

       try:
           audio = cm.AudioFile(filepath)
           audio.open()
           return audio
       except cm.AudioFileError as e:
           raise RuntimeError(f"Failed to open audio file: {e}")
       except Exception as e:
           raise RuntimeError(f"Unexpected error: {e}")

   # Use with error handling
   try:
       audio = safe_open_audio_file("audio.wav")
       try:
           # Work with file
           print(f"Duration: {audio.duration}")
       finally:
           audio.close()
   except FileNotFoundError as e:
       print(f"Error: {e}")
   except RuntimeError as e:
       print(f"Error: {e}")

Validating Audio Files
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def validate_audio_file(filepath):
       """Validate audio file can be opened and read."""
       try:
           with cm.AudioFile(filepath) as audio:
               # Try to read first packet
               data, count = audio.read_packets(0, 1)

               if count == 0:
                   return False, "File contains no audio data"

               # Check basic format validity
               fmt = audio.format
               if fmt.sample_rate <= 0:
                   return False, "Invalid sample rate"

               if fmt.channels_per_frame <= 0:
                   return False, "Invalid channel count"

               return True, "Valid audio file"

       except Exception as e:
           return False, f"Validation failed: {e}"

   # Validate file
   is_valid, message = validate_audio_file("audio.wav")
   print(f"Valid: {is_valid}, Message: {message}")

Complete Example
----------------

Audio File Inspector
^^^^^^^^^^^^^^^^^^^^

A complete tool that inspects audio files:

.. code-block:: python

   import coremusic as cm
   import sys
   from pathlib import Path

   def format_bytes(num_bytes):
       """Format bytes as human-readable string."""
       for unit in ['B', 'KB', 'MB', 'GB']:
           if num_bytes < 1024.0:
               return f"{num_bytes:.2f} {unit}"
           num_bytes /= 1024.0
       return f"{num_bytes:.2f} TB"

   def inspect_audio_file(filepath):
       """Comprehensive audio file inspection."""
       # Check file exists
       path = Path(filepath)
       if not path.exists():
           print(f"Error: File not found: {filepath}")
           return

       print(f"Inspecting: {filepath}")
       print(f"File size: {format_bytes(path.stat().st_size)}")
       print()

       try:
           with cm.AudioFile(filepath) as audio:
               # Format information
               fmt = audio.format
               print("Format Information:")
               print(f"  Format ID: {fmt.format_id}")
               print(f"  Sample Rate: {fmt.sample_rate} Hz")
               print(f"  Channels: {fmt.channels_per_frame}")
               print(f"  Bit Depth: {fmt.bits_per_channel}")
               print(f"  Bytes/Frame: {fmt.bytes_per_frame}")
               print()

               # Duration information
               print("Duration Information:")
               print(f"  Total Frames: {audio.frame_count:,}")
               print(f"  Duration: {audio.duration:.2f} seconds")
               print(f"  Duration: {audio.duration / 60:.2f} minutes")
               print()

               # Quality classification
               print("Classification:")
               if fmt.sample_rate == 44100 and fmt.bits_per_channel == 16:
                   quality = "CD Quality"
               elif fmt.sample_rate >= 96000:
                   quality = "Hi-Res Audio"
               elif fmt.sample_rate >= 48000:
                   quality = "Professional Audio"
               else:
                   quality = "Standard Audio"
               print(f"  Quality: {quality}")

               channel_type = {
                   1: "Mono",
                   2: "Stereo",
                   4: "Quadraphonic",
                   6: "5.1 Surround",
                   8: "7.1 Surround"
               }.get(fmt.channels_per_frame, f"{fmt.channels_per_frame}-channel")
               print(f"  Channel Type: {channel_type}")

               # Calculate bitrate
               bitrate = (fmt.sample_rate * fmt.bytes_per_frame * 8) / 1000
               print(f"  Bitrate: {bitrate:.0f} kbps")

       except cm.AudioFileError as e:
           print(f"Error opening file: {e}")
       except Exception as e:
           print(f"Unexpected error: {e}")

   if __name__ == "__main__":
       if len(sys.argv) < 2:
           print("Usage: python inspect_audio.py <audio_file>")
           sys.exit(1)

       inspect_audio_file(sys.argv[1])

Save as ``inspect_audio.py`` and run:

.. code-block:: bash

   python inspect_audio.py audio.wav

**Example output:**

.. code-block:: text

   Inspecting: audio.wav
   File size: 529.03 KB

   Format Information:
     Format ID: lpcm
     Sample Rate: 44100.0 Hz
     Channels: 2
     Bit Depth: 16
     Bytes/Frame: 4

   Duration Information:
     Total Frames: 120,960
     Duration: 2.74 seconds
     Duration: 0.05 minutes

   Classification:
     Quality: CD Quality
     Channel Type: Stereo
     Bitrate: 1411 kbps

Next Steps
----------

Now that you understand audio file basics, explore:

- :doc:`../cookbook/file_operations` - Common file operation recipes

See Also
--------

- :doc:`../api/audio_file` - Complete AudioFile API reference
