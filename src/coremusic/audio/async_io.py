#!/usr/bin/env python3
"""Async I/O support for CoreMusic.

This module provides async/await support for long-running audio operations,
enabling non-blocking file I/O and better integration with modern Python
async frameworks.

Example usage:
    ```python
    import asyncio
    import coremusic as cm

    async def process_audio():
        # Async file reading with chunk streaming
        async with cm.AsyncAudioFile("large_file.wav") as audio:
            print(f"Duration: {audio.duration:.2f}s")
            async for chunk in audio.read_chunks_async(chunk_size=4096):
                await process_audio_chunk(chunk)

        # Async AudioQueue playback
        format = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)
        queue = await cm.AsyncAudioQueue.new_output_async(format)
        await queue.start_async()
        await asyncio.sleep(1.0)
        await queue.stop_async()

    asyncio.run(process_audio())
    ```
"""

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Tuple, Union

from ..objects import (NUMPY_AVAILABLE, AudioBuffer, AudioFile, AudioFormat,
                       AudioQueue)

if NUMPY_AVAILABLE:
    from numpy.typing import NDArray


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "AsyncAudioFile",
    "AsyncAudioQueue",
    "open_audio_file_async",
    "create_output_queue_async",
]

# ============================================================================
# Async Audio File
# ============================================================================


class AsyncAudioFile:
    """Async audio file operations with automatic resource management.

    This class wraps AudioFile to provide async/await support for file
    operations, using asyncio.to_thread() to avoid blocking the event loop.

    Example:
        ```python
        async with AsyncAudioFile("audio.wav") as audio:
            print(f"Duration: {audio.duration:.2f}s")

            # Stream audio data in chunks
            async for chunk in audio.read_chunks_async(chunk_size=4096):
                await process_chunk(chunk)
        ```
    """

    def __init__(self, path: Union[str, Path]):
        """Initialize async audio file.

        Args:
            path: Path to the audio file
        """
        self._audio_file = AudioFile(path)
        self._path = str(path)

    async def __aenter__(self) -> "AsyncAudioFile":
        """Async context manager entry."""
        await asyncio.to_thread(self._audio_file.open)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await asyncio.to_thread(self._audio_file.close)

    async def open_async(self) -> "AsyncAudioFile":
        """Open the audio file asynchronously.

        Returns:
            Self for method chaining
        """
        await asyncio.to_thread(self._audio_file.open)
        return self

    async def close_async(self) -> None:
        """Close the audio file asynchronously."""
        await asyncio.to_thread(self._audio_file.close)

    @property
    def format(self) -> AudioFormat:
        """Get the audio format (cached, synchronous property)."""
        return self._audio_file.format

    @property
    def duration(self) -> float:
        """Get duration in seconds (cached, synchronous property)."""
        return self._audio_file.duration

    @property
    def path(self) -> str:
        """Get file path."""
        return self._path

    async def read_packets_async(
        self, start_packet: int, packet_count: int
    ) -> Tuple[bytes, int]:
        """Read audio packets asynchronously.

        Args:
            start_packet: Starting packet number
            packet_count: Number of packets to read

        Returns:
            Tuple of (audio data as bytes, actual packet count)
        """
        return await asyncio.to_thread(  # type: ignore[call-arg]
            self._audio_file.read_packets,  # type: ignore[arg-type]
            start_packet,
            packet_count,
        )

    async def read_chunks_async(
        self,
        chunk_size: int = 4096,
        start_packet: int = 0,
        total_packets: Optional[int] = None,
    ) -> AsyncIterator[bytes]:
        """Stream audio data in chunks asynchronously.

        This is an async generator that yields chunks of audio data,
        allowing for efficient streaming of large files without blocking
        the event loop.

        Args:
            chunk_size: Number of packets per chunk (default: 4096)
            start_packet: Starting packet number (default: 0)
            total_packets: Total packets to read (default: all remaining)

        Yields:
            Chunks of raw audio data as bytes

        Example:
            ```python
            async with AsyncAudioFile("large.wav") as audio:
                async for chunk in audio.read_chunks_async(chunk_size=4096):
                    await process_chunk(chunk)
            ```
        """
        current_packet = start_packet

        # If total_packets not specified, read until we get no data
        while True:
            remaining = chunk_size
            chunk_data, actual_count = await asyncio.to_thread(  # type: ignore[call-arg]
                self._audio_file.read_packets, current_packet, remaining
            )

            if actual_count == 0 or not chunk_data:  # type: ignore[comparison-overlap]
                break

            yield chunk_data
            current_packet += int(actual_count)

            # Yield control to event loop
            await asyncio.sleep(0)

            # Stop if we've read the requested number of packets
            if (
                total_packets is not None
                and current_packet >= start_packet + total_packets
            ):
                break

    if NUMPY_AVAILABLE:

        async def read_as_numpy_async(
            self, start_packet: int = 0, packet_count: Optional[int] = None
        ) -> "NDArray":
            """Read audio data as NumPy array asynchronously.

            Args:
                start_packet: Starting packet number (default: 0)
                packet_count: Number of packets to read (default: all remaining)

            Returns:
                NumPy array with shape (frames, channels)
            """
            return await asyncio.to_thread(
                self._audio_file.read_as_numpy,
                start_packet,
                packet_count,
            )

        async def read_chunks_numpy_async(
            self,
            chunk_size: int = 4096,
            start_packet: int = 0,
            total_packets: Optional[int] = None,
        ) -> AsyncIterator["NDArray"]:
            """Stream audio data as NumPy arrays asynchronously.

            Args:
                chunk_size: Number of packets per chunk (default: 4096)
                start_packet: Starting packet number (default: 0)
                total_packets: Total packets to read (default: all remaining)

            Yields:
                NumPy arrays with shape (frames, channels)

            Example:
                ```python
                async with AsyncAudioFile("audio.wav") as audio:
                    async for chunk in audio.read_chunks_numpy_async():
                        # chunk is a NumPy array
                        spectrum = np.fft.fft(chunk)
                        await process_spectrum(spectrum)
                ```
            """
            current_packet = start_packet

            while True:
                remaining = chunk_size
                try:
                    chunk = await asyncio.to_thread(
                        self._audio_file.read_as_numpy,
                        current_packet,
                        remaining,
                    )
                except Exception:
                    break

                if chunk.size == 0:
                    break

                yield chunk
                # Calculate how many packets we actually read
                # This is approximate - depends on format
                current_packet += remaining

                # Yield control to event loop
                await asyncio.sleep(0)

                # Stop if we've read the requested number of packets
                if (
                    total_packets is not None
                    and current_packet >= start_packet + total_packets
                ):
                    break

    def __repr__(self) -> str:
        return f"AsyncAudioFile({self._path})"


# ============================================================================
# Async Audio Queue
# ============================================================================


class AsyncAudioQueue:
    """Async audio queue for non-blocking playback and recording.

    This class wraps AudioQueue to provide async/await support for queue
    operations, enabling non-blocking playback and recording.

    Example:
        ```python
        format = AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)

        async with AsyncAudioQueue.new_output_async(format) as queue:
            buffer = await queue.allocate_buffer_async(4096)
            await queue.enqueue_buffer_async(buffer)
            await queue.start_async()
            await asyncio.sleep(1.0)
            await queue.stop_async()
        ```
    """

    def __init__(self, audio_queue: AudioQueue):
        """Initialize async audio queue wrapper.

        Args:
            audio_queue: Underlying AudioQueue instance
        """
        self._queue = audio_queue

    @classmethod
    async def new_output_async(cls, audio_format: AudioFormat) -> "AsyncAudioQueue":
        """Create a new output audio queue asynchronously.

        Args:
            audio_format: Audio format for the queue

        Returns:
            AsyncAudioQueue instance
        """
        queue = await asyncio.to_thread(AudioQueue.new_output, audio_format)  # type: ignore[attr-defined]
        return cls(queue)

    async def __aenter__(self) -> "AsyncAudioQueue":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.dispose_async()

    async def allocate_buffer_async(self, buffer_size: int) -> AudioBuffer:
        """Allocate an audio buffer asynchronously.

        Args:
            buffer_size: Size of the buffer in bytes

        Returns:
            AudioBuffer instance
        """
        return await asyncio.to_thread(self._queue.allocate_buffer, buffer_size)

    async def enqueue_buffer_async(self, buffer: AudioBuffer) -> None:
        """Enqueue an audio buffer asynchronously.

        Args:
            buffer: AudioBuffer to enqueue
        """
        await asyncio.to_thread(self._queue.enqueue_buffer, buffer)  # type: ignore[attr-defined]

    async def start_async(self) -> None:
        """Start the audio queue asynchronously."""
        await asyncio.to_thread(self._queue.start)

    async def stop_async(self, immediate: bool = True) -> None:
        """Stop the audio queue asynchronously.

        Args:
            immediate: If True, stop immediately; if False, wait for buffers to finish
        """
        await asyncio.to_thread(self._queue.stop, immediate)

    async def dispose_async(self, immediate: bool = True) -> None:
        """Dispose the audio queue asynchronously.

        Args:
            immediate: If True, dispose immediately; if False, wait for buffers to finish
        """
        await asyncio.to_thread(self._queue.dispose, immediate)  # type: ignore[call-arg]

    @property
    def format(self) -> AudioFormat:
        """Get the audio format (synchronous property)."""
        return self._queue._format  # type: ignore[no-any-return,attr-defined]

    def __repr__(self) -> str:
        return f"AsyncAudioQueue(format={self.format})"


# ============================================================================
# Convenience Functions
# ============================================================================


async def open_audio_file_async(path: Union[str, Path]) -> AsyncAudioFile:
    """Open an audio file asynchronously.

    Convenience function for opening audio files in async context.

    Args:
        path: Path to the audio file

    Returns:
        Opened AsyncAudioFile instance

    Example:
        ```python
        audio = await open_audio_file_async("audio.wav")
        try:
            print(f"Duration: {audio.duration:.2f}s")
            data = await audio.read_frames_async()
        finally:
            await audio.close_async()
        ```
    """
    audio = AsyncAudioFile(path)
    await audio.open_async()
    return audio


async def create_output_queue_async(audio_format: AudioFormat) -> AsyncAudioQueue:
    """Create an output audio queue asynchronously.

    Convenience function for creating output queues.

    Args:
        audio_format: Audio format for the queue

    Returns:
        AsyncAudioQueue instance

    Example:
        ```python
        format = AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)
        queue = await create_output_queue_async(format)
        await queue.start_async()
        ```
    """
    return await AsyncAudioQueue.new_output_async(audio_format)
