#!/usr/bin/env python3
"""Driving an audio queue from asyncio."""

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_output_devices():
    print("No audio output device available.")
    raise SystemExit(0)

# --8<-- [start:queue]
import asyncio

from coremusic.audio import AsyncAudioQueue, AudioFormat


async def play_audio():
    """Start and stop an output queue asynchronously."""
    # Create audio format
    audio_format = AudioFormat.pcm(sample_rate=44100.0, channels=2, bits=16)

    # Create async queue
    queue = await AsyncAudioQueue.new_output_async(audio_format)

    try:
        # Start playback
        await queue.start_async()
        print("Playback started")

        await asyncio.sleep(0.5)

        # Stop playback
        await queue.stop_async()
        print("Playback stopped")

    finally:
        await queue.dispose_async()


asyncio.run(play_audio())
# --8<-- [end:queue]

# --8<-- [start:monitor]
import asyncio

from coremusic.audio import AsyncAudioQueue, AudioFormat


async def monitor_playback(duration):
    """Report progress while something else plays."""
    elapsed = 0.0
    while elapsed < duration:
        print(f"Playing: {elapsed:.1f}s / {duration:.1f}s", end="\r")
        await asyncio.sleep(0.1)
        elapsed += 0.1
    print()


async def main():
    audio_format = AudioFormat.pcm(44100.0, channels=2)
    queue = await AsyncAudioQueue.new_output_async(audio_format)

    try:
        await queue.start_async()

        # Run monitoring concurrently with playback
        await monitor_playback(duration=0.5)

        await queue.stop_async()
    finally:
        await queue.dispose_async()


asyncio.run(main())
# --8<-- [end:monitor]
