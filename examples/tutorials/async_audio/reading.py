#!/usr/bin/env python3
"""Reading audio files without blocking the event loop."""

# --8<-- [start:info]
import asyncio

from coremusic.audio import AsyncAudioFile


async def read_audio_info(filepath):
    """Read audio file info asynchronously."""
    async with AsyncAudioFile(filepath) as audio:
        print(f"File: {filepath}")
        print(f"Duration: {audio.duration:.2f}s")
        print(f"Sample rate: {audio.format.sample_rate}Hz")
        print(f"Channels: {audio.format.channels_per_frame}")


# Run the async function
asyncio.run(read_audio_info("audio.wav"))
# --8<-- [end:info]

# --8<-- [start:chunks]
import asyncio

from coremusic.audio import AsyncAudioFile


async def process_audio_chunks(filepath, chunk_size=4096):
    """Process audio file in chunks without blocking."""
    total_bytes = 0
    chunk_count = 0

    async with AsyncAudioFile(filepath) as audio:
        async for chunk in audio.read_chunks_async(chunk_size=chunk_size):
            # Process each chunk (non-blocking)
            total_bytes += len(chunk)
            chunk_count += 1

            # Simulate some async processing
            await asyncio.sleep(0)  # Yield to event loop

    print(f"Processed {chunk_count} chunks, {total_bytes:,} bytes total")


asyncio.run(process_audio_chunks("audio.wav"))
# --8<-- [end:chunks]

# --8<-- [start:concurrent]
import asyncio
from pathlib import Path

from coremusic.audio import AsyncAudioFile


async def analyze_file(filepath):
    """Analyze a single audio file."""
    async with AsyncAudioFile(filepath) as audio:
        return {
            "path": str(filepath),
            "duration": audio.duration,
            "sample_rate": audio.format.sample_rate,
            "channels": audio.format.channels_per_frame,
        }


async def analyze_multiple_files(filepaths):
    """Analyze multiple files concurrently."""
    tasks = [analyze_file(fp) for fp in filepaths]
    return await asyncio.gather(*tasks)


async def main():
    wav_files = sorted(Path(".").glob("*.wav"))

    print(f"Analyzing {len(wav_files)} files...")

    results = await analyze_multiple_files(wav_files)

    total_duration = sum(r["duration"] for r in results)
    for r in results:
        print(f"  {r['path']}: {r['duration']:.2f}s")
    print(f"Total: {total_duration:.2f}s")


asyncio.run(main())
# --8<-- [end:concurrent]
