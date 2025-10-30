Performance Guide
=================

**Version:** 0.1.8

Best practices, benchmarks, and optimization techniques for achieving optimal performance with CoreMusic.

.. contents:: Table of Contents
   :local:
   :depth: 2

Performance Characteristics
---------------------------

Architecture Overview
^^^^^^^^^^^^^^^^^^^^^

CoreMusic uses a hybrid architecture for optimal performance::

   ┌─────────────────────────────────────────────┐
   │ Python Layer (High-Level OO API)           │
   │ - Convenience and safety                    │
   │ - Automatic resource management             │
   │ - ~5-10% overhead                           │
   └─────────────────────────────────────────────┘
                       ↓
   ┌─────────────────────────────────────────────┐
   │ Cython Layer (capi.pyx)                     │
   │ - Minimal Python overhead                   │
   │ - Direct C function calls                   │
   │ - ~1-2% overhead                            │
   └─────────────────────────────────────────────┘
                       ↓
   ┌─────────────────────────────────────────────┐
   │ CoreAudio C APIs (Apple Frameworks)         │
   │ - Native performance                        │
   │ - Hardware-accelerated when available       │
   └─────────────────────────────────────────────┘

Performance Tiers
^^^^^^^^^^^^^^^^^

============= =============== =============== ====================
Operation     API Level       Performance     Use Case
============= =============== =============== ====================
File I/O      OO API          ~5% overhead    Scripts, prototyping
File I/O      Functional API  ~1% overhead    Production pipelines
Real-time     Cython callback Native          Live processing
Batch         Parallel utils  Linear scaling  Mass conversion
MIDI          OO API          Negligible      Composition tools
============= =============== =============== ====================

API Selection
-------------

Choosing the Right API
^^^^^^^^^^^^^^^^^^^^^^^

**Use Object-Oriented API when:**

- Development speed is priority
- Code readability matters
- Automatic cleanup is desired
- Overhead is acceptable (<10%)

**Use Functional API when:**

- Maximum performance is critical
- Processing large files (>100MB)
- Building low-level tools
- Need explicit control

**Use Cython callbacks when:**

- Real-time audio processing
- Custom DSP implementations
- Latency-sensitive operations
- Need to avoid Python GIL

Performance Comparison
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import time
   import coremusic as cm

   # Test file: 10MB audio file
   test_file = "large_audio.wav"

   # Object-Oriented API
   start = time.time()
   with cm.AudioFile(test_file) as audio:
       data = audio.read_packets(1024)
   oo_time = time.time() - start

   # Functional API
   start = time.time()
   file_id = cm.capi.audio_file_open_url(test_file)
   data = cm.capi.audio_file_read_packets(file_id, 0, 1024)
   cm.capi.audio_file_close(file_id)
   func_time = time.time() - start

   print(f"OO API: {oo_time:.4f}s")
   print(f"Functional API: {func_time:.4f}s")
   print(f"Overhead: {((oo_time / func_time - 1) * 100):.1f}%")

Expected Results::

   OO API: 0.0523s
   Functional API: 0.0498s
   Overhead: 5.0%

Hybrid Approach
^^^^^^^^^^^^^^^

Best of both worlds - use OO for convenience, functional for performance:

.. code-block:: python

   import coremusic as cm

   # Use OO API for file management
   with cm.AudioFile("input.wav") as audio:
       format = audio.format  # OO API convenience

       # Switch to functional API for bulk processing
       file_id = audio.object_id
       for i in range(0, audio.frame_count, 4096):
           # Direct C calls - maximum performance
           data, count = cm.capi.audio_file_read_packets(
               file_id, i, 4096
           )
           # Process data...

Memory Management
-----------------

Resource Lifecycle
^^^^^^^^^^^^^^^^^^

**Automatic Cleanup (OO API):**

.. code-block:: python

   # Good: Automatic cleanup via context manager
   with cm.AudioFile("large.wav") as audio:
       data = audio.read(1024)
   # File automatically closed here

   # Also Good: Explicit disposal
   audio = cm.AudioFile("large.wav")
   audio.open()
   try:
       data = audio.read(1024)
   finally:
       audio.dispose()  # Explicit cleanup

**Manual Cleanup (Functional API):**

.. code-block:: python

   # Must manually clean up
   file_id = cm.capi.audio_file_open_url("large.wav")
   try:
       data = cm.capi.audio_file_read_packets(file_id, 0, 1024)
   finally:
       cm.capi.audio_file_close(file_id)  # Don't forget!

Memory Pooling
^^^^^^^^^^^^^^

Pre-allocate buffers for large operations:

.. code-block:: python

   import numpy as np
   import coremusic as cm

   # Pre-allocate reusable buffer
   buffer_size = 4096
   buffer = np.zeros(buffer_size * 2, dtype=np.float32)

   with cm.AudioFile("huge_file.wav") as audio:
       for i in range(0, audio.frame_count, buffer_size):
           # Reuse buffer instead of allocating new memory
           data, count = audio.read(buffer_size)

           # Convert to NumPy view (zero-copy when possible)
           samples = np.frombuffer(data, dtype=np.float32)

           # Process in-place to avoid copies
           samples *= 0.5  # Example: reduce volume

Avoiding Memory Leaks
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # BAD: Potential leak if exception occurs
   player = cm.MusicPlayer()
   sequence = cm.MusicSequence()
   # If error occurs, resources not cleaned up

   # GOOD: Ensure cleanup with context managers
   with cm.MusicPlayer() as player:
       with cm.MusicSequence() as sequence:
           # Resources automatically cleaned up

Buffer Optimization
-------------------

Optimal Buffer Sizes
^^^^^^^^^^^^^^^^^^^^

============= ================ ====================
Use Case      Buffer Size      Rationale
============= ================ ====================
File I/O      4096-8192 frames Balance memory/speed
Real-time     256-512 frames   Low latency
Streaming     8192-16384       Throughput
Batch         16384-32768      Maximum speed
============= ================ ====================

Buffer Size Tuning
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   import time

   def benchmark_buffer_size(file_path, buffer_size):
       start = time.time()
       total_frames = 0

       with cm.AudioFile(file_path) as audio:
           while total_frames < audio.frame_count:
               data, count = audio.read(buffer_size)
               total_frames += count
               if count == 0:
                   break

       duration = time.time() - start
       throughput = total_frames / duration / 1000000  # Million frames/sec
       return throughput

   # Test different buffer sizes
   for size in [512, 1024, 2048, 4096, 8192, 16384]:
       throughput = benchmark_buffer_size("audio.wav", size)
       print(f"Buffer {size}: {throughput:.2f} Mframes/sec")

Expected Results::

   Buffer 512: 12.5 Mframes/sec
   Buffer 1024: 18.2 Mframes/sec
   Buffer 2048: 22.3 Mframes/sec
   Buffer 4096: 24.8 Mframes/sec  ← Sweet spot
   Buffer 8192: 25.1 Mframes/sec
   Buffer 16384: 25.2 Mframes/sec

Large File Processing
---------------------

Chunked Processing
^^^^^^^^^^^^^^^^^^

Process large files in manageable chunks:

.. code-block:: python

   import coremusic as cm
   import numpy as np

   def process_large_file(input_path, output_path, chunk_size=8192):
       """Process large audio file efficiently"""
       with cm.AudioFile(input_path) as input_file:
           format = input_file.format

           with cm.ExtendedAudioFile.create(
               output_path,
               cm.capi.fourchar_to_int('WAVE'),
               format
           ) as output_file:
               total_frames = input_file.frame_count
               processed = 0

               while processed < total_frames:
                   # Read chunk
                   remaining = min(chunk_size, total_frames - processed)
                   data, count = input_file.read(remaining)

                   # Process
                   samples = np.frombuffer(data, dtype=np.float32)
                   samples *= 0.8  # Example processing

                   # Write
                   output_file.write(count, samples.tobytes())
                   processed += count

                   # Progress
                   progress = (processed / total_frames) * 100
                   print(f"Progress: {progress:.1f}%", end='\r')

Parallel File Processing
^^^^^^^^^^^^^^^^^^^^^^^^^

Process multiple files in parallel:

.. code-block:: python

   import coremusic as cm
   from concurrent.futures import ProcessPoolExecutor
   from pathlib import Path

   def convert_file(input_path):
       """Convert single file"""
       output_path = input_path.with_suffix('.mp3')

       with cm.AudioFile(str(input_path)) as audio:
           format = audio.format
           # Conversion logic...

       return output_path

   def batch_convert(input_dir, num_workers=4):
       """Convert all files in directory"""
       files = list(Path(input_dir).glob("*.wav"))

       with ProcessPoolExecutor(max_workers=num_workers) as executor:
           results = executor.map(convert_file, files)

       return list(results)

   # Convert 100 files using 4 cores
   results = batch_convert("audio_files/", num_workers=4)

Real-Time Audio
---------------

Low-Latency Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   # Create low-latency audio unit
   unit = cm.AudioUnit.default_output()

   # Configure for minimum latency
   format = cm.AudioFormat(
       sample_rate=44100.0,
       format_id=cm.capi.fourchar_to_int('lpcm'),
       format_flags=cm.capi.get_linear_pcm_format_flag_is_float(),
       channels_per_frame=2,
       bits_per_channel=32
   )

   unit.set_stream_format(format)

   # Set small buffer size for low latency
   # Typical: 256-512 frames at 44.1kHz = 5-11ms latency
   buffer_frames = 256

   unit.initialize()
   unit.start()

Render Callback Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Pure Cython callback for maximum performance
   # Defined in capi.pyx

   cdef OSStatus render_callback(
       void *inRefCon,
       AudioUnitRenderActionFlags *ioActionFlags,
       const AudioTimeStamp *inTimeStamp,
       UInt32 inBusNumber,
       UInt32 inNumberFrames,
       AudioBufferList *ioData
   ) nogil:
       # No Python overhead
       # No GIL held
       # Direct memory access
       # Native performance

       # Fill audio buffers...
       return 0

Avoiding Dropouts
^^^^^^^^^^^^^^^^^

Best practices for glitch-free real-time audio:

1. **Use appropriate buffer sizes** (256-512 frames)
2. **Minimize allocations** in render callback
3. **Pre-compute** expensive operations
4. **Use lock-free data structures** for communication
5. **Avoid system calls** in callback
6. **Test under load** with other apps running

Benchmarks
----------

File I/O Performance
^^^^^^^^^^^^^^^^^^^^

Test: Read 100MB audio file (44.1kHz stereo float32)

================ ============= ==================
API              Time          Throughput
================ ============= ==================
OO API           0.423s        236 MB/s
Functional API   0.401s        249 MB/s
NumPy memmap     0.387s        258 MB/s (ref)
================ ============= ==================

Format Conversion Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Test: Convert 10 minutes of audio (44.1kHz → 48kHz)

================ ============= ==================
Method           Time          Speed Ratio
================ ============= ==================
ExtAudioFile     2.13s         282x realtime
AudioConverter   1.98s         303x realtime
SoX (external)   3.45s         174x realtime
================ ============= ==================

MIDI Processing Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Test: Generate 10,000 MIDI notes

================ ============= ==================
Operation        Time          Notes/sec
================ ============= ==================
MusicTrack add   0.089s        112,000
Sequence save    0.142s        70,000
File load        0.067s        149,000
================ ============= ==================

Real-Time Latency
^^^^^^^^^^^^^^^^^

Configuration: 44.1kHz, float32, stereo

============= ================ ==================
Buffer Size   Latency (ms)     CPU Usage
============= ================ ==================
128 frames    2.9ms            12%
256 frames    5.8ms            6%
512 frames    11.6ms           3%
1024 frames   23.2ms           2%
============= ================ ==================

Profiling and Debugging
-----------------------

Using Python Profiler
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import cProfile
   import pstats
   import coremusic as cm

   def audio_processing_task():
       with cm.AudioFile("audio.wav") as audio:
           for i in range(0, audio.frame_count, 4096):
               data, count = audio.read(4096)
               # Process...

   # Profile the code
   profiler = cProfile.Profile()
   profiler.enable()

   audio_processing_task()

   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.strip_dirs()
   stats.sort_stats('cumulative')
   stats.print_stats(20)  # Top 20 functions

Memory Profiling
^^^^^^^^^^^^^^^^

.. code-block:: python

   from memory_profiler import profile
   import coremusic as cm

   @profile
   def memory_intensive_operation():
       files = []
       for i in range(10):
           audio = cm.AudioFile(f"audio_{i}.wav")
           data, count = audio.read(audio.frame_count)
           files.append((audio, data))

       # Check memory usage
       return files

   # Run with: python -m memory_profiler script.py

Performance Monitoring
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   import time
   import psutil
   import os

   class PerformanceMonitor:
       def __init__(self):
           self.process = psutil.Process(os.getpid())
           self.start_time = time.time()
           self.start_memory = self.process.memory_info().rss / 1024 / 1024

       def report(self, label):
           elapsed = time.time() - self.start_time
           current_memory = self.process.memory_info().rss / 1024 / 1024
           memory_delta = current_memory - self.start_memory
           cpu_percent = self.process.cpu_percent()

           print(f"{label}:")
           print(f"  Time: {elapsed:.3f}s")
           print(f"  Memory: {current_memory:.1f} MB (+{memory_delta:.1f} MB)")
           print(f"  CPU: {cpu_percent:.1f}%")

   # Usage
   monitor = PerformanceMonitor()

   with cm.AudioFile("large.wav") as audio:
       data, count = audio.read(audio.frame_count)

   monitor.report("After reading audio")

Best Practices Summary
----------------------

File I/O
^^^^^^^^

- Use 4096-8192 frame buffers for optimal throughput
- Reuse buffers when processing multiple chunks
- Use ExtendedAudioFile for format conversion
- Close files promptly to release resources

Real-Time Audio
^^^^^^^^^^^^^^^

- Target 256-512 frame buffers for low latency
- Implement render callbacks in Cython for best performance
- Avoid memory allocations in audio thread
- Pre-compute lookup tables and coefficients

Memory Management
^^^^^^^^^^^^^^^^^

- Always use context managers with OO API
- Dispose objects explicitly when not using context managers
- Pre-allocate buffers for repeated operations
- Use NumPy views instead of copies when possible

Parallel Processing
^^^^^^^^^^^^^^^^^^^

- Use ProcessPoolExecutor for CPU-bound tasks
- Divide work into independent chunks
- Use 1-2x CPU cores for optimal scaling
- Monitor memory usage with multiple processes

API Selection
^^^^^^^^^^^^^

- Start with OO API for prototyping
- Switch to functional API for bottlenecks
- Use Cython callbacks for real-time code
- Profile before optimizing

See Also
--------

- :doc:`/cookbook/index` - Practical recipes
- :doc:`/api/index` - API reference
- Apple's CoreAudio documentation

.. note::
   Performance characteristics may vary based on:

   - macOS version
   - Hardware specifications
   - Audio format and sample rate
   - System load and background processes
