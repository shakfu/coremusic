#!/usr/bin/env python3
"""Error handling, batching, and producer/consumer."""

# --8<-- [start:errors]
import asyncio

from coremusic.audio import AsyncAudioFile
from coremusic.exceptions import AudioFileError


async def safe_read_audio(filepath):
    """Safely read audio with error handling."""
    try:
        async with AsyncAudioFile(filepath) as audio:
            data, count = await audio.read_packets_async(0, 1000)
            return data, count
    except AudioFileError as e:
        print(f"Audio error: {e}")
        return None, 0
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None, 0


async def process_files_safely(filepaths):
    """Process multiple files with error handling."""
    results = []
    for filepath in filepaths:
        result = await safe_read_audio(filepath)
        if result[0] is not None:
            results.append(result)
    return results


asyncio.run(process_files_safely(["audio.wav", "missing.wav"]))
# --8<-- [end:errors]

# --8<-- [start:batches]
import asyncio

from coremusic.audio import AsyncAudioFile


async def process_file(filepath):
    """Process a single file."""
    async with AsyncAudioFile(filepath) as audio:
        # Your processing logic here
        return audio.duration


async def process_batch(filepaths, batch_size=10):
    """Process files in batches to limit concurrency."""
    results = []
    for i in range(0, len(filepaths), batch_size):
        batch = filepaths[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[process_file(fp) for fp in batch]
        )
        results.extend(batch_results)
        print(f"Processed batch {i // batch_size + 1}")
    return results


asyncio.run(process_batch(["audio.wav", "input.wav"], batch_size=1))
# --8<-- [end:batches]

# --8<-- [start:semaphore]
import asyncio

from coremusic.audio import AsyncAudioFile


async def process_with_limit(filepaths, max_concurrent=5):
    """Process files with limited concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_one(filepath):
        async with semaphore:
            async with AsyncAudioFile(filepath) as audio:
                # Process file
                return audio.duration

    tasks = [process_one(fp) for fp in filepaths]
    return await asyncio.gather(*tasks)


print(asyncio.run(process_with_limit(["audio.wav", "input.wav"])))
# --8<-- [end:semaphore]

# --8<-- [start:producer-consumer]
import asyncio

from coremusic.audio import AsyncAudioFile


async def audio_producer(filepath, queue):
    """Produce audio chunks."""
    async with AsyncAudioFile(filepath) as audio:
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
# --8<-- [end:producer-consumer]
