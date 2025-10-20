Audio File Inspector
====================

A comprehensive tool to inspect and display detailed audio file information.

Overview
--------

The audio inspector demonstrates:

- Reading audio file metadata
- Accessing format information
- Calculating audio statistics
- Formatting output for readability
- Proper error handling

The complete source code is in ``examples/audio_inspector.py``.

Features
--------

**File Information:**
- File path and size
- File system metadata

**Format Details:**
- Audio format (PCM, AAC, MP3, etc.)
- Sample rate and bit depth
- Channel configuration
- Byte layout

**Duration Information:**
- Total frames and samples
- Duration in multiple formats
- Memory requirements

**Quality Classification:**
- Automatic quality detection (CD, Hi-Res, etc.)
- Channel type description
- Bitrate calculation

Usage
-----

Basic usage:

.. code-block:: bash

   python examples/audio_inspector.py audio.wav

With relative path:

.. code-block:: bash

   python examples/audio_inspector.py tests/amen.wav

From examples directory:

.. code-block:: bash

   cd examples
   python audio_inspector.py ../tests/amen.wav

Example Output
--------------

For a CD-quality WAV file:

.. code-block:: text

   ======================================================================
   Audio File Inspector
   ======================================================================

   FILE INFORMATION
   ----------------------------------------------------------------------
     Filename:     amen.wav
     Path:         /Users/you/coremusic/tests/amen.wav
     File Size:    529.03 KB

   FORMAT INFORMATION
   ----------------------------------------------------------------------
     Format ID:    lpcm
     Sample Rate:  44,100 Hz
     Channels:     2 (Stereo)
     Bit Depth:    16-bit
     Bytes/Frame:  4
     Bytes/Packet: 4
     Frames/Packet: 1
     Format Flags: 0x0000000C

   DURATION INFORMATION
   ----------------------------------------------------------------------
     Total Frames: 120,960
     Duration:     00:02.742 (2.742s)
     Minutes:      0.05

   CLASSIFICATION
   ----------------------------------------------------------------------
     Quality:      CD Quality
     Channel Type: Stereo
     Bitrate:      1,411 kbps

   FORMAT DETAILS
   ----------------------------------------------------------------------
     Format Type:  Linear PCM (Uncompressed)
     Data Type:    Integer
     Byte Order:   Little Endian
     Signed:       Yes
     Packed:       Yes

   SAMPLE INFORMATION
   ----------------------------------------------------------------------
     Total Samples:    241,920
     Samples/Second:   88,200
     Memory Required:  471.09 KB

   DATA RATE
   ----------------------------------------------------------------------
     Bytes/Second:     172.27 KB/s
     Bits/Second:      1,411 kbps

   ======================================================================
   Inspection complete!
   ======================================================================

For an MP3 file:

.. code-block:: text

   FORMAT INFORMATION
   ----------------------------------------------------------------------
     Format ID:    .mp3
     Sample Rate:  44,100 Hz
     Channels:     2 (Stereo)
     Bit Depth:    16-bit

   FORMAT DETAILS
   ----------------------------------------------------------------------
     Format Type:  MP3 (MPEG-1 Audio Layer 3) - Compressed

Implementation Details
----------------------

Key Concepts
^^^^^^^^^^^^

**Format Detection:**

.. code-block:: python

   with cm.AudioFile(filepath) as audio:
       fmt = audio.format

       if fmt.format_id == 'lpcm':
           print("Linear PCM (uncompressed)")
       elif fmt.format_id == 'aac ':
           print("AAC (compressed)")

**Quality Classification:**

.. code-block:: python

   def get_quality_classification(fmt):
       """Classify audio quality based on format."""
       if fmt.sample_rate == 44100 and fmt.bits_per_channel == 16:
           return "CD Quality"
       elif fmt.sample_rate >= 96000:
           return "Hi-Res Audio"
       elif fmt.sample_rate >= 48000:
           return "Professional Audio"
       else:
           return "Standard Audio"

**Format Flags Interpretation:**

.. code-block:: python

   # For Linear PCM, decode format flags
   is_float = fmt.format_flags & 0x01
   is_big_endian = fmt.format_flags & 0x02
   is_signed = fmt.format_flags & 0x04
   is_packed = fmt.format_flags & 0x08

**Human-Readable Formatting:**

.. code-block:: python

   def format_bytes(num_bytes):
       """Format bytes as human-readable string."""
       for unit in ['B', 'KB', 'MB', 'GB']:
           if num_bytes < 1024.0:
               return f"{num_bytes:.2f} {unit}"
           num_bytes /= 1024.0
       return f"{num_bytes:.2f} TB"

   def format_duration(seconds):
       """Format duration as MM:SS.ms."""
       minutes = int(seconds // 60)
       secs = seconds % 60
       return f"{minutes:02d}:{secs:06.3f}"

Error Handling
^^^^^^^^^^^^^^

The inspector handles various error cases:

.. code-block:: python

   # File not found
   if not Path(filepath).exists():
       print(f"Error: File not found: {filepath}")
       return False

   # Audio file errors
   try:
       with cm.AudioFile(filepath) as audio:
           # Process file
           pass
   except cm.AudioFileError as e:
       print(f"Error opening audio file: {e}")
       return False
   except Exception as e:
       print(f"Unexpected error: {e}")
       return False

Extending the Inspector
-----------------------

Add Audio Analysis
^^^^^^^^^^^^^^^^^^

Add statistical analysis using NumPy:

.. code-block:: python

   import numpy as np

   # Read audio data
   data, count = audio.read_packets(0, audio.frame_count)

   # Convert to NumPy array
   samples = np.frombuffer(data, dtype=np.int16)

   # Calculate statistics
   print(f"  Mean:     {np.mean(samples):.2f}")
   print(f"  Std Dev:  {np.std(samples):.2f}")
   print(f"  Min:      {np.min(samples)}")
   print(f"  Max:      {np.max(samples)}")
   print(f"  RMS:      {np.sqrt(np.mean(samples**2)):.2f}")

Add Waveform Display
^^^^^^^^^^^^^^^^^^^^

Display a simple waveform:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Read audio
   data, count = audio.read_packets(0, min(audio.frame_count, 10000))
   samples = np.frombuffer(data, dtype=np.int16)

   # Plot waveform
   plt.figure(figsize=(12, 4))
   plt.plot(samples[:1000])
   plt.title("Waveform Preview")
   plt.xlabel("Sample")
   plt.ylabel("Amplitude")
   plt.show()

Add Format Warnings
^^^^^^^^^^^^^^^^^^^

Warn about potential issues:

.. code-block:: python

   # Check for unusual configurations
   if fmt.sample_rate < 44100:
       print("  WARNING: Low sample rate for music")

   if fmt.bits_per_channel < 16:
       print("  WARNING: Low bit depth may cause quality issues")

   if fmt.channels_per_frame > 2 and fmt.format_id != 'lpcm':
       print("  WARNING: Multi-channel compressed audio")

Complete Source Code
--------------------

The complete, working source code is available in ``examples/audio_inspector.py``.

Key sections:

1. **Imports and helpers** - Formatting utilities
2. **File information** - Path, size, metadata
3. **Format information** - Audio format details
4. **Duration calculation** - Time and frame counts
5. **Classification** - Quality and type detection
6. **Format-specific** - PCM flags, compression info
7. **Error handling** - Robust error management

See Also
--------

- :doc:`audio_converter` - Audio format conversion example
- :doc:`../tutorials/audio_file_basics` - Audio file tutorial
- :doc:`../api/audio_file` - AudioFile API reference
- :doc:`../cookbook/file_operations` - File operation recipes
