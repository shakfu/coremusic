File Operations
===============

Common recipes for audio file operations.

Reading Audio Files
-------------------

Read Entire File
^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def read_audio_file(filepath):
       """Read entire audio file into memory."""
       with cm.AudioFile(filepath) as audio:
           data, count = audio.read_packets(0, audio.frame_count)
           return data, audio.format

   # Usage
   audio_data, format_info = read_audio_file("audio.wav")
   print(f"Read {len(audio_data)} bytes")

Read File in Chunks
^^^^^^^^^^^^^^^^^^^

For large files, read in manageable chunks:

.. code-block:: python

   import coremusic as cm

   def read_audio_chunks(filepath, chunk_size=4096):
       """Generator that yields audio chunks."""
       with cm.AudioFile(filepath) as audio:
           total_frames = audio.frame_count
           current = 0

           while current < total_frames:
               to_read = min(chunk_size, total_frames - current)
               data, count = audio.read_packets(current, to_read)
               yield data
               current += count

   # Usage
   for chunk in read_audio_chunks("large_file.wav"):
       process_chunk(chunk)

Read Specific Section
^^^^^^^^^^^^^^^^^^^^^

Read a specific time range from an audio file:

.. code-block:: python

   import coremusic as cm

   def read_time_range(filepath, start_seconds, duration_seconds):
       """Read specific time range from audio file."""
       with cm.AudioFile(filepath) as audio:
           # Calculate frame positions
           sample_rate = audio.format.sample_rate
           start_frame = int(start_seconds * sample_rate)
           frame_count = int(duration_seconds * sample_rate)

           # Ensure we don't read past end
           max_frames = audio.frame_count - start_frame
           frame_count = min(frame_count, max_frames)

           # Read data
           data, count = audio.read_packets(start_frame, frame_count)
           return data

   # Usage: read 2 seconds starting at 5 seconds
   data = read_time_range("audio.wav", start_seconds=5.0, duration_seconds=2.0)

Writing Audio Files
-------------------

Convert to NumPy Array
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   import numpy as np

   def audio_to_numpy(filepath):
       """Convert audio file to NumPy array."""
       with cm.AudioFile(filepath) as audio:
           # Read raw data
           data, count = audio.read_packets(0, audio.frame_count)

           # Get format info
           fmt = audio.format
           dtype = np.int16 if fmt.bits_per_channel == 16 else np.int32

           # Convert to numpy array
           samples = np.frombuffer(data, dtype=dtype)

           # Reshape for channels
           if fmt.channels_per_frame > 1:
               samples = samples.reshape(-1, fmt.channels_per_frame)

           return samples, fmt.sample_rate

   # Usage
   samples, sample_rate = audio_to_numpy("audio.wav")
   print(f"Shape: {samples.shape}, Sample rate: {sample_rate}Hz")

File Information
----------------

Get File Metadata
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   from pathlib import Path

   def get_audio_metadata(filepath):
       """Extract comprehensive audio file metadata."""
       path = Path(filepath)

       with cm.AudioFile(filepath) as audio:
           fmt = audio.format

           metadata = {
               'filename': path.name,
               'file_size': path.stat().st_size,
               'duration': audio.duration,
               'sample_rate': fmt.sample_rate,
               'channels': fmt.channels_per_frame,
               'bit_depth': fmt.bits_per_channel,
               'format_id': fmt.format_id,
               'frame_count': audio.frame_count,
               'bitrate': (fmt.sample_rate * fmt.bytes_per_frame * 8) / 1000
           }

           return metadata

   # Usage
   metadata = get_audio_metadata("audio.wav")
   for key, value in metadata.items():
       print(f"{key}: {value}")

Compare Audio Files
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def compare_audio_files(file1, file2):
       """Compare two audio files for format compatibility."""
       with cm.AudioFile(file1) as audio1, cm.AudioFile(file2) as audio2:
           fmt1 = audio1.format
           fmt2 = audio2.format

           comparison = {
               'same_sample_rate': fmt1.sample_rate == fmt2.sample_rate,
               'same_channels': fmt1.channels_per_frame == fmt2.channels_per_frame,
               'same_bit_depth': fmt1.bits_per_channel == fmt2.bits_per_channel,
               'same_format': fmt1.format_id == fmt2.format_id,
               'same_duration': abs(audio1.duration - audio2.duration) < 0.01,
           }

           comparison['compatible'] = all(comparison.values())

           return comparison

   # Usage
   result = compare_audio_files("audio1.wav", "audio2.wav")
   print(f"Files compatible: {result['compatible']}")

File Validation
---------------

Validate Audio File
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   from pathlib import Path

   def validate_audio_file(filepath, min_duration=0.1, max_duration=3600):
       """Validate audio file meets requirements."""
       errors = []

       # Check file exists
       if not Path(filepath).exists():
           errors.append(f"File not found: {filepath}")
           return False, errors

       try:
           with cm.AudioFile(filepath) as audio:
               fmt = audio.format

               # Check duration
               if audio.duration < min_duration:
                   errors.append(f"Duration too short: {audio.duration}s")

               if audio.duration > max_duration:
                   errors.append(f"Duration too long: {audio.duration}s")

               # Check sample rate
               if fmt.sample_rate < 8000 or fmt.sample_rate > 192000:
                   errors.append(f"Invalid sample rate: {fmt.sample_rate}Hz")

               # Check channels
               if fmt.channels_per_frame < 1 or fmt.channels_per_frame > 32:
                   errors.append(f"Invalid channel count: {fmt.channels_per_frame}")

               # Check bit depth
               if fmt.bits_per_channel not in [8, 16, 24, 32]:
                   errors.append(f"Unsupported bit depth: {fmt.bits_per_channel}")

               # Try to read first frame
               try:
                   data, count = audio.read_packets(0, 1)
                   if count == 0:
                       errors.append("File contains no audio data")
               except Exception as e:
                   errors.append(f"Cannot read audio data: {e}")

       except cm.AudioFileError as e:
           errors.append(f"Audio file error: {e}")
       except Exception as e:
           errors.append(f"Unexpected error: {e}")

       return len(errors) == 0, errors

   # Usage
   is_valid, errors = validate_audio_file("audio.wav")
   if not is_valid:
       print("Validation errors:")
       for error in errors:
           print(f"  - {error}")

Check Format Support
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def is_format_supported(filepath):
       """Check if audio format is supported."""
       try:
           with cm.AudioFile(filepath) as audio:
               # Try to read format
               fmt = audio.format

               # Try to read data
               data, count = audio.read_packets(0, 1)

               return True, f"Supported: {fmt.format_id}"

       except cm.AudioFileError as e:
           return False, f"Not supported: {e}"
       except Exception as e:
           return False, f"Error: {e}"

   # Usage
   supported, message = is_format_supported("audio.mp3")
   print(message)

File Utilities
--------------

Calculate Audio Statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   import numpy as np

   def calculate_audio_stats(filepath):
       """Calculate audio statistics."""
       with cm.AudioFile(filepath) as audio:
           # Read audio data
           data, count = audio.read_packets(0, audio.frame_count)

           # Convert to numpy
           fmt = audio.format
           dtype = np.int16 if fmt.bits_per_channel == 16 else np.int32
           samples = np.frombuffer(data, dtype=dtype)

           # Calculate statistics
           stats = {
               'mean': float(np.mean(samples)),
               'std': float(np.std(samples)),
               'min': int(np.min(samples)),
               'max': int(np.max(samples)),
               'rms': float(np.sqrt(np.mean(samples**2))),
           }

           # Calculate peak amplitude
           max_value = 2**(fmt.bits_per_channel - 1) - 1
           stats['peak_amplitude'] = max(abs(stats['min']), abs(stats['max'])) / max_value

           return stats

   # Usage
   stats = calculate_audio_stats("audio.wav")
   print(f"Peak amplitude: {stats['peak_amplitude']:.2%}")
   print(f"RMS: {stats['rms']:.2f}")

Detect Silence
^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   import numpy as np

   def detect_silence(filepath, threshold=0.01, min_duration=0.5):
       """Detect silent regions in audio file."""
       with cm.AudioFile(filepath) as audio:
           fmt = audio.format
           sample_rate = fmt.sample_rate

           # Read audio
           data, count = audio.read_packets(0, audio.frame_count)

           # Convert to numpy
           dtype = np.int16 if fmt.bits_per_channel == 16 else np.int32
           samples = np.frombuffer(data, dtype=dtype)

           # Normalize to [-1, 1]
           max_value = 2**(fmt.bits_per_channel - 1)
           samples = samples.astype(np.float32) / max_value

           # Calculate RMS in windows
           window_size = int(0.1 * sample_rate)  # 100ms windows
           num_windows = len(samples) // window_size

           silent_regions = []
           in_silence = False
           silence_start = 0

           for i in range(num_windows):
               window = samples[i * window_size:(i + 1) * window_size]
               rms = np.sqrt(np.mean(window**2))

               if rms < threshold:
                   if not in_silence:
                       silence_start = i * window_size / sample_rate
                       in_silence = True
               else:
                   if in_silence:
                       silence_end = i * window_size / sample_rate
                       duration = silence_end - silence_start

                       if duration >= min_duration:
                           silent_regions.append((silence_start, silence_end, duration))

                       in_silence = False

           return silent_regions

   # Usage
   silent_regions = detect_silence("audio.wav")
   print(f"Found {len(silent_regions)} silent regions:")
   for start, end, duration in silent_regions:
       print(f"  {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")

Format Human-Readable Info
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def format_duration(seconds):
       """Format duration as MM:SS."""
       minutes = int(seconds // 60)
       secs = int(seconds % 60)
       return f"{minutes:02d}:{secs:02d}"

   def format_file_size(bytes):
       """Format file size as human-readable."""
       for unit in ['B', 'KB', 'MB', 'GB']:
           if bytes < 1024:
               return f"{bytes:.2f} {unit}"
           bytes /= 1024
       return f"{bytes:.2f} TB"

   def format_audio_info(filepath):
       """Format audio information for display."""
       from pathlib import Path

       path = Path(filepath)
       with cm.AudioFile(filepath) as audio:
           fmt = audio.format

           info = f"""
   File: {path.name}
   Size: {format_file_size(path.stat().st_size)}
   Duration: {format_duration(audio.duration)}
   Format: {fmt.format_id}
   Sample Rate: {fmt.sample_rate:,.0f} Hz
   Channels: {fmt.channels_per_frame}
   Bit Depth: {fmt.bits_per_channel}-bit
   Bitrate: {(fmt.sample_rate * fmt.bytes_per_frame * 8 / 1000):,.0f} kbps
   """
           return info.strip()

   # Usage
   print(format_audio_info("audio.wav"))

See Also
--------

- :doc:`../tutorials/audio_file_basics` - Audio file fundamentals
- :doc:`../api/audio_file` - AudioFile API reference
