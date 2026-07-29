#!/usr/bin/env python3
"""Async File Reading."""

# --8<-- [start:example]
import asyncio

from coremusic.audio import AsyncAudioFile


async def read_async():
    async with AsyncAudioFile("audio.wav") as audio:
        print(f"Duration: {audio.duration:.2f}s")

        # Stream chunks
        async for chunk in audio.read_chunks_async(chunk_size=4096):
            # Process chunk
            print(len(chunk))


asyncio.run(read_async())
# --8<-- [end:example]
