Async Audio Programming
=======================

This tutorial covers asynchronous audio programming with coremusic, enabling
non-blocking I/O for responsive applications.

Why Async?
----------

Asynchronous programming is valuable for audio applications because:

- **Responsive UIs**: File operations don't freeze your interface
- **Concurrent Processing**: Process multiple files simultaneously
- **Server Applications**: Handle multiple clients efficiently
- **Real-time Integration**: Combine audio I/O with network operations

Prerequisites
-------------

- Python 3.11+ (for best async support)
- Basic understanding of Python's ``async``/``await`` syntax
- Familiarity with coremusic's synchronous API

Async File Operations
---------------------

Basic Async File Reading
^^^^^^^^^^^^^^^^^^^^^^^^

Use ``AsyncAudioFile`` for non-blocking file operations:

.. code-block:: python

   import asyncio
   import coremusic as cm

   async def read_audio_info(filepath):
       """Read audio file info asynchronously."""
       async with cm.AsyncAudioFile(filepath) as audio:
           print(f"File: {filepath}")
           print(f"Duration: {audio.duration:.2f}s")
           print(f"Sample rate: {audio.format.sample_rate}Hz")
           print(f"Channels: {audio.format.channels_per_frame}")

   # Run the async function
   asyncio.run(read_audio_info("audio.wav"))

Streaming Audio Chunks
^^^^^^^^^^^^^^^^^^^^^^

Process large files efficiently using async iteration:

.. code-block:: python

   import asyncio
   import coremusic as cm

   async def process_audio_chunks(filepath, chunk_size=4096):
       """Process audio file in chunks without blocking."""
       total_bytes = 0
       chunk_count = 0

       async with cm.AsyncAudioFile(filepath) as audio:
           async for chunk in audio.read_chunks_async(chunk_size=chunk_size):
               # Process each chunk (non-blocking)
               total_bytes += len(chunk)
               chunk_count += 1

               # Simulate some async processing
               await asyncio.sleep(0)  # Yield to event loop

       print(f"Processed {chunk_count} chunks, {total_bytes:,} bytes total")

   asyncio.run(process_audio_chunks("large_audio.wav"))

Concurrent File Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^

Process multiple files simultaneously:

.. code-block:: python

   import asyncio
   import coremusic as cm
   from pathlib import Path

   async def analyze_file(filepath):
       """Analyze a single audio file."""
       async with cm.AsyncAudioFile(filepath) as audio:
           return {
               'path': str(filepath),
               'duration': audio.duration,
               'sample_rate': audio.format.sample_rate,
               'channels': audio.format.channels_per_frame
           }

   async def analyze_multiple_files(filepaths):
       """Analyze multiple files concurrently."""
       tasks = [analyze_file(fp) for fp in filepaths]
       results = await asyncio.gather(*tasks)
       return results

   async def main():
       # Find all WAV files
       audio_dir = Path("audio_files")
       wav_files = list(audio_dir.glob("*.wav"))

       print(f"Analyzing {len(wav_files)} files...")

       results = await analyze_multiple_files(wav_files)

       # Print results
       total_duration = sum(r['duration'] for r in results)
       print(f"\nTotal duration: {total_duration:.2f} seconds")
       for r in results:
           print(f"  {r['path']}: {r['duration']:.2f}s")

   asyncio.run(main())

Async Audio Queue
-----------------

Basic Async Playback
^^^^^^^^^^^^^^^^^^^^

Use ``AsyncAudioQueue`` for non-blocking playback control:

.. code-block:: python

   import asyncio
   import coremusic as cm

   async def play_audio():
       """Play audio asynchronously."""
       # Create audio format
       format = cm.AudioFormat(
           sample_rate=44100.0,
           format_id='lpcm',
           channels_per_frame=2,
           bits_per_channel=16
       )

       # Create async queue
       queue = await cm.AsyncAudioQueue.new_output_async(format)

       try:
           # Start playback
           await queue.start_async()
           print("Playback started")

           # Play for 2 seconds
           await asyncio.sleep(2.0)

           # Stop playback
           await queue.stop_async()
           print("Playback stopped")

       finally:
           await queue.dispose_async()

   asyncio.run(play_audio())

Combining with Other Async Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Integrate audio with other async tasks:

.. code-block:: python

   import asyncio
   import coremusic as cm

   async def monitor_playback(queue, duration):
       """Monitor playback progress."""
       elapsed = 0.0
       while elapsed < duration:
           print(f"Playing: {elapsed:.1f}s / {duration:.1f}s", end='\r')
           await asyncio.sleep(0.1)
           elapsed += 0.1
       print()

   async def main():
       format = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2)
       queue = await cm.AsyncAudioQueue.new_output_async(format)

       try:
           await queue.start_async()

           # Run monitoring concurrently with playback
           await monitor_playback(queue, duration=3.0)

           await queue.stop_async()
       finally:
           await queue.dispose_async()

   asyncio.run(main())

Error Handling in Async Code
----------------------------

Proper async error handling:

.. code-block:: python

   import asyncio
   import coremusic as cm

   async def safe_read_audio(filepath):
       """Safely read audio with error handling."""
       try:
           async with cm.AsyncAudioFile(filepath) as audio:
               data, count = await audio.read_packets_async(0, 1000)
               return data, count
       except cm.AudioFileError as e:
           print(f"Audio error: {e}")
           return None, 0
       except FileNotFoundError:
           print(f"File not found: {filepath}")
           return None, 0
       except Exception as e:
           print(f"Unexpected error: {e}")
           return None, 0

   async def process_files_safely(filepaths):
       """Process multiple files with error handling."""
       results = []
       for filepath in filepaths:
           result = await safe_read_audio(filepath)
           if result[0] is not None:
               results.append(result)
       return results

   asyncio.run(process_files_safely(["file1.wav", "file2.wav"]))

Patterns and Best Practices
---------------------------

Batch Processing Pattern
^^^^^^^^^^^^^^^^^^^^^^^^

Efficiently process files in batches:

.. code-block:: python

   import asyncio
   import coremusic as cm

   async def process_file(filepath):
       """Process a single file."""
       async with cm.AsyncAudioFile(filepath) as audio:
           # Your processing logic here
           return audio.duration

   async def process_batch(filepaths, batch_size=10):
       """Process files in batches to limit concurrency."""
       results = []
       for i in range(0, len(filepaths), batch_size):
           batch = filepaths[i:i+batch_size]
           batch_results = await asyncio.gather(
               *[process_file(fp) for fp in batch]
           )
           results.extend(batch_results)
           print(f"Processed batch {i//batch_size + 1}")
       return results

Semaphore Pattern
^^^^^^^^^^^^^^^^^

Limit concurrent operations:

.. code-block:: python

   import asyncio
   import coremusic as cm

   async def process_with_limit(filepaths, max_concurrent=5):
       """Process files with limited concurrency."""
       semaphore = asyncio.Semaphore(max_concurrent)

       async def process_one(filepath):
           async with semaphore:
               async with cm.AsyncAudioFile(filepath) as audio:
                   # Process file
                   return audio.duration

       tasks = [process_one(fp) for fp in filepaths]
       return await asyncio.gather(*tasks)

Producer-Consumer Pattern
^^^^^^^^^^^^^^^^^^^^^^^^^

For streaming audio processing:

.. code-block:: python

   import asyncio
   import coremusic as cm

   async def audio_producer(filepath, queue):
       """Produce audio chunks."""
       async with cm.AsyncAudioFile(filepath) as audio:
           async for chunk in audio.read_chunks_async(chunk_size=4096):
               await queue.put(chunk)
       await queue.put(None)  # Signal completion

   async def audio_consumer(queue):
       """Consume and process audio chunks."""
       total_bytes = 0
       while True:
           chunk = await queue.get()
           if chunk is None:
               break
           # Process chunk
           total_bytes += len(chunk)
           queue.task_done()
       return total_bytes

   async def main():
       queue = asyncio.Queue(maxsize=10)

       producer = asyncio.create_task(audio_producer("audio.wav", queue))
       consumer = asyncio.create_task(audio_consumer(queue))

       await producer
       total = await consumer
       print(f"Processed {total:,} bytes")

   asyncio.run(main())

Integration with Web Frameworks
-------------------------------

FastAPI Example
^^^^^^^^^^^^^^^

Using coremusic with FastAPI:

.. code-block:: python

   from fastapi import FastAPI, UploadFile
   import coremusic as cm
   import tempfile
   import os

   app = FastAPI()

   @app.post("/analyze")
   async def analyze_audio(file: UploadFile):
       """Analyze uploaded audio file."""
       # Save uploaded file temporarily
       with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
           content = await file.read()
           tmp.write(content)
           tmp_path = tmp.name

       try:
           async with cm.AsyncAudioFile(tmp_path) as audio:
               return {
                   "filename": file.filename,
                   "duration": audio.duration,
                   "sample_rate": audio.format.sample_rate,
                   "channels": audio.format.channels_per_frame
               }
       finally:
           os.unlink(tmp_path)

See Also
--------

- :doc:`audio_file_basics` - Synchronous file operations
- :doc:`../cookbook/file_operations` - File operation recipes
- :doc:`../api/index` - API reference
