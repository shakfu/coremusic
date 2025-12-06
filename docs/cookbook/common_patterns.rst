Common Patterns
===============

Essential patterns for effective coremusic usage.

Resource Management
-------------------

Context Manager Pattern
^^^^^^^^^^^^^^^^^^^^^^^

Always use context managers for automatic cleanup:

.. code-block:: python

   import coremusic as cm

   # Good: automatic cleanup
   with cm.AudioFile("audio.wav") as audio:
       data, count = audio.read_packets(0, 1024)
   # File automatically closed

   # Good: nested context managers
   with cm.AudioFile("input.wav") as input_file:
       with cm.ExtendedAudioFile.create("output.wav", ...) as output_file:
           # Process...
           pass

   # Avoid: manual management (error-prone)
   audio = cm.AudioFile("audio.wav")
   audio.open()
   # If exception here, file never closes!
   audio.close()

Multiple Resources Pattern
^^^^^^^^^^^^^^^^^^^^^^^^^^

Handle multiple resources safely:

.. code-block:: python

   from contextlib import ExitStack

   def process_multiple_files(input_files, output_path):
       """Process multiple input files safely."""
       with ExitStack() as stack:
           # Open all input files
           inputs = [
               stack.enter_context(cm.AudioFile(f))
               for f in input_files
           ]

           # Open output file
           output = stack.enter_context(
               cm.ExtendedAudioFile.create(output_path, ...)
           )

           # Process all files
           for input_file in inputs:
               data, count = input_file.read_packets(0, input_file.frame_count)
               output.write(count, data)
       # All files automatically closed

Error Handling
--------------

Graceful Error Recovery
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   from pathlib import Path

   def safe_audio_operation(filepath):
       """Perform audio operation with comprehensive error handling."""
       # Pre-check
       if not Path(filepath).exists():
           return None, "File not found"

       try:
           with cm.AudioFile(filepath) as audio:
               data, count = audio.read_packets(0, audio.frame_count)
               return data, None

       except cm.AudioFileError as e:
           return None, f"Audio error: {e}"
       except MemoryError:
           return None, "File too large for memory"
       except Exception as e:
           return None, f"Unexpected error: {e}"

   # Usage
   data, error = safe_audio_operation("audio.wav")
   if error:
       print(f"Failed: {error}")
   else:
       print(f"Read {len(data)} bytes")

Retry Pattern
^^^^^^^^^^^^^

.. code-block:: python

   import time
   import coremusic as cm

   def retry_operation(func, max_retries=3, delay=0.5):
       """Retry an operation with exponential backoff."""
       last_error = None

       for attempt in range(max_retries):
           try:
               return func()
           except cm.CoreAudioError as e:
               last_error = e
               if attempt < max_retries - 1:
                   time.sleep(delay * (2 ** attempt))

       raise last_error

   # Usage
   def read_file():
       with cm.AudioFile("audio.wav") as audio:
           return audio.read_packets(0, 1024)

   data, count = retry_operation(read_file)

Format Handling
---------------

Format Detection and Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def validate_audio_format(filepath, required_format=None):
       """Validate audio file format."""
       with cm.AudioFile(filepath) as audio:
           fmt = audio.format

           # Basic validation
           if fmt.sample_rate <= 0:
               raise ValueError("Invalid sample rate")
           if fmt.channels_per_frame <= 0:
               raise ValueError("Invalid channel count")

           # Check against required format
           if required_format:
               if fmt.sample_rate != required_format.sample_rate:
                   raise ValueError(
                       f"Sample rate mismatch: {fmt.sample_rate} != {required_format.sample_rate}"
                   )
               if fmt.channels_per_frame != required_format.channels_per_frame:
                   raise ValueError(
                       f"Channel mismatch: {fmt.channels_per_frame} != {required_format.channels_per_frame}"
                   )

           return fmt

Format Conversion Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def convert_to_standard_format(input_path, output_path):
       """Convert any audio to standard PCM format."""
       # Standard format: 44.1kHz, 16-bit, stereo PCM
       target_format = cm.AudioFormat(
           sample_rate=44100.0,
           format_id='lpcm',
           format_flags=0x0C,  # Signed integer, packed
           channels_per_frame=2,
           bits_per_channel=16,
           bytes_per_frame=4,
           frames_per_packet=1,
           bytes_per_packet=4
       )

       with cm.ExtendedAudioFile(input_path) as input_file:
           # Set client format for automatic conversion
           input_file.client_format = target_format

           with cm.ExtendedAudioFile.create(
               output_path,
               cm.capi.fourchar_to_int('WAVE'),
               target_format
           ) as output_file:
               # Copy with automatic conversion
               chunk_size = 8192
               while True:
                   data, count = input_file.read(chunk_size)
                   if count == 0:
                       break
                   output_file.write(count, data)

Streaming Patterns
------------------

Generator-Based Streaming
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def stream_audio(filepath, chunk_size=4096):
       """Stream audio data as a generator."""
       with cm.AudioFile(filepath) as audio:
           total_frames = audio.frame_count
           current = 0

           while current < total_frames:
               to_read = min(chunk_size, total_frames - current)
               data, count = audio.read_packets(current, to_read)

               if count == 0:
                   break

               yield data
               current += count

   # Usage
   for chunk in stream_audio("large_file.wav"):
       process_chunk(chunk)

Progress Tracking
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def process_with_progress(filepath, callback=None):
       """Process audio with progress callback."""
       with cm.AudioFile(filepath) as audio:
           total = audio.frame_count
           processed = 0
           chunk_size = 4096

           while processed < total:
               data, count = audio.read_packets(processed, chunk_size)
               if count == 0:
                   break

               # Process data
               process_chunk(data)

               processed += count

               # Report progress
               if callback:
                   progress = processed / total
                   callback(progress)

   # Usage with progress bar
   def show_progress(progress):
       bar_length = 40
       filled = int(bar_length * progress)
       bar = '=' * filled + '-' * (bar_length - filled)
       print(f'\r[{bar}] {progress:.1%}', end='')

   process_with_progress("audio.wav", callback=show_progress)
   print()  # New line after progress bar

Caching Patterns
----------------

Simple Cache
^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   from functools import lru_cache

   @lru_cache(maxsize=100)
   def get_audio_info(filepath):
       """Get cached audio file information."""
       with cm.AudioFile(filepath) as audio:
           return {
               'duration': audio.duration,
               'sample_rate': audio.format.sample_rate,
               'channels': audio.format.channels_per_frame,
               'frame_count': audio.frame_count
           }

   # First call: reads file
   info1 = get_audio_info("audio.wav")

   # Second call: returns cached result
   info2 = get_audio_info("audio.wav")

File Hash Cache
^^^^^^^^^^^^^^^

.. code-block:: python

   import hashlib
   from pathlib import Path
   import coremusic as cm

   class AudioCache:
       """Cache audio data by file hash."""

       def __init__(self):
           self._cache = {}

       def _get_file_hash(self, filepath):
           """Get MD5 hash of file."""
           hasher = hashlib.md5()
           with open(filepath, 'rb') as f:
               for chunk in iter(lambda: f.read(8192), b''):
                   hasher.update(chunk)
           return hasher.hexdigest()

       def get_data(self, filepath):
           """Get cached audio data or load from file."""
           file_hash = self._get_file_hash(filepath)

           if file_hash not in self._cache:
               with cm.AudioFile(filepath) as audio:
                   data, count = audio.read_packets(0, audio.frame_count)
                   self._cache[file_hash] = data

           return self._cache[file_hash]

   # Usage
   cache = AudioCache()
   data = cache.get_data("audio.wav")

Batch Processing
----------------

Parallel Processing
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   from concurrent.futures import ThreadPoolExecutor
   from pathlib import Path

   def process_file(filepath):
       """Process a single audio file."""
       with cm.AudioFile(filepath) as audio:
           # Your processing logic
           return {
               'path': str(filepath),
               'duration': audio.duration
           }

   def process_batch(filepaths, max_workers=4):
       """Process multiple files in parallel."""
       results = []

       with ThreadPoolExecutor(max_workers=max_workers) as executor:
           futures = {executor.submit(process_file, fp): fp for fp in filepaths}

           for future in futures:
               try:
                   result = future.result()
                   results.append(result)
               except Exception as e:
                   filepath = futures[future]
                   print(f"Error processing {filepath}: {e}")

       return results

   # Usage
   wav_files = list(Path("audio_dir").glob("*.wav"))
   results = process_batch(wav_files)

Sequential with Logging
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   import logging
   from pathlib import Path

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   def batch_process_sequential(input_dir, output_dir, processor_func):
       """Process all audio files in directory sequentially."""
       input_path = Path(input_dir)
       output_path = Path(output_dir)
       output_path.mkdir(parents=True, exist_ok=True)

       audio_files = list(input_path.glob("*.wav"))
       total = len(audio_files)

       logger.info(f"Processing {total} files...")

       for i, input_file in enumerate(audio_files, 1):
           output_file = output_path / input_file.name

           try:
               processor_func(str(input_file), str(output_file))
               logger.info(f"[{i}/{total}] Processed: {input_file.name}")
           except Exception as e:
               logger.error(f"[{i}/{total}] Failed: {input_file.name} - {e}")

       logger.info("Batch processing complete")

Configuration Patterns
----------------------

Audio Format Presets
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   class AudioFormats:
       """Common audio format presets."""

       CD_QUALITY = cm.AudioFormat(
           sample_rate=44100.0,
           format_id='lpcm',
           format_flags=0x0C,
           channels_per_frame=2,
           bits_per_channel=16,
           bytes_per_frame=4,
           frames_per_packet=1,
           bytes_per_packet=4
       )

       DVD_QUALITY = cm.AudioFormat(
           sample_rate=48000.0,
           format_id='lpcm',
           format_flags=0x0C,
           channels_per_frame=2,
           bits_per_channel=24,
           bytes_per_frame=6,
           frames_per_packet=1,
           bytes_per_packet=6
       )

       HIRES_AUDIO = cm.AudioFormat(
           sample_rate=96000.0,
           format_id='lpcm',
           format_flags=0x0C,
           channels_per_frame=2,
           bits_per_channel=24,
           bytes_per_frame=6,
           frames_per_packet=1,
           bytes_per_packet=6
       )

       FLOAT32_STEREO = cm.AudioFormat(
           sample_rate=44100.0,
           format_id='lpcm',
           format_flags=0x09,  # Float, packed
           channels_per_frame=2,
           bits_per_channel=32,
           bytes_per_frame=8,
           frames_per_packet=1,
           bytes_per_packet=8
       )

   # Usage
   format = AudioFormats.CD_QUALITY

See Also
--------

- :doc:`file_operations` - File I/O recipes
- :doc:`audio_processing` - Audio processing recipes
- :doc:`/guides/performance` - Performance optimization
