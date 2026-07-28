#!/usr/bin/env python3
"""Read a file asynchronously, without blocking the event loop."""

# --8<-- [start:example]
import asyncio

from coremusic.audio import AsyncAudioFile


async def async_playback(filepath):
    """Non-blocking audio file reading."""
    async with AsyncAudioFile(filepath) as audio:
        print(f"Duration: {audio.duration:.2f}s")

        # Stream chunks asynchronously
        total = 0
        async for chunk in audio.read_chunks_async(chunk_size=4096):
            # Process each chunk without blocking
            total += len(chunk)
            await asyncio.sleep(0)  # Yield to event loop

        print(f"Read {total:,} bytes")


# Run async playback
asyncio.run(async_playback("audio.wav"))
# --8<-- [end:example]
