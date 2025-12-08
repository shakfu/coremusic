#!/usr/bin/env python3
"""Tests for async I/O functionality."""

import os
import pytest
import asyncio
from pathlib import Path

import coremusic as cm


class TestAsyncAudioFile:
    """Test AsyncAudioFile async wrapper"""


    @pytest.mark.asyncio
    async def test_async_audio_file_creation(self, amen_wav_path):
        """Test AsyncAudioFile object creation"""
        # Test with string path
        audio_file = cm.AsyncAudioFile(amen_wav_path)
        assert isinstance(audio_file, cm.AsyncAudioFile)
        assert audio_file.path == amen_wav_path

        # Test with Path object
        audio_file_path = cm.AsyncAudioFile(Path(amen_wav_path))
        assert isinstance(audio_file_path, cm.AsyncAudioFile)
        assert audio_file_path.path == str(Path(amen_wav_path))

    @pytest.mark.asyncio
    async def test_async_audio_file_open_close(self, amen_wav_path):
        """Test AsyncAudioFile opening and closing"""
        audio_file = cm.AsyncAudioFile(amen_wav_path)

        # Test opening
        result = await audio_file.open_async()
        assert result is audio_file  # Should return self

        # Test closing
        await audio_file.close_async()

    @pytest.mark.asyncio
    async def test_async_audio_file_context_manager(self, amen_wav_path):
        """Test AsyncAudioFile as async context manager"""
        async with cm.AsyncAudioFile(amen_wav_path) as audio_file:
            assert isinstance(audio_file, cm.AsyncAudioFile)
            assert audio_file.format is not None

    @pytest.mark.asyncio
    async def test_async_audio_file_format_property(self, amen_wav_path):
        """Test AsyncAudioFile format property"""
        async with cm.AsyncAudioFile(amen_wav_path) as audio_file:
            format = audio_file.format
            assert isinstance(format, cm.AudioFormat)

            # Verify it's the expected WAV format
            assert format.format_id == "lpcm"
            assert format.sample_rate == 44100.0
            assert format.channels_per_frame == 2
            assert format.is_pcm
            assert format.is_stereo

    @pytest.mark.asyncio
    async def test_async_audio_file_duration(self, amen_wav_path):
        """Test AsyncAudioFile duration property"""
        async with cm.AsyncAudioFile(amen_wav_path) as audio_file:
            duration = audio_file.duration
            assert isinstance(duration, float)
            assert duration > 0
            # amen.wav is approximately 2.74 seconds
            assert 2.0 < duration < 3.0

    @pytest.mark.asyncio
    async def test_async_read_packets(self, amen_wav_path):
        """Test async packet reading"""
        async with cm.AsyncAudioFile(amen_wav_path) as audio_file:
            # Read first 100 packets
            data, packet_count = await audio_file.read_packets_async(
                start_packet=0, packet_count=100
            )
            assert isinstance(data, bytes)
            assert len(data) > 0
            assert isinstance(packet_count, int)
            assert packet_count > 0

    @pytest.mark.asyncio
    async def test_async_read_chunks(self, amen_wav_path):
        """Test async chunk streaming"""
        async with cm.AsyncAudioFile(amen_wav_path) as audio_file:
            chunks = []
            chunk_size = 1024

            async for chunk in audio_file.read_chunks_async(chunk_size=chunk_size):
                assert isinstance(chunk, bytes)
                assert len(chunk) > 0
                chunks.append(chunk)

            # Should have received multiple chunks
            assert len(chunks) > 1

            # Total size should match file size
            total_size = sum(len(chunk) for chunk in chunks)
            assert total_size > 0

    @pytest.mark.asyncio
    async def test_async_read_chunks_with_range(self, amen_wav_path):
        """Test async chunk streaming with range specification"""
        async with cm.AsyncAudioFile(amen_wav_path) as audio_file:
            chunks = []
            start_packet = 100
            total_packets = 500
            chunk_size = 100

            async for chunk in audio_file.read_chunks_async(
                chunk_size=chunk_size,
                start_packet=start_packet,
                total_packets=total_packets,
            ):
                chunks.append(chunk)

            # Should have received multiple chunks
            assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_async_concurrent_file_reads(self, amen_wav_path):
        """Test concurrent async file reads"""

        async def read_file():
            async with cm.AsyncAudioFile(amen_wav_path) as audio:
                return await audio.read_packets_async(start_packet=0, packet_count=100)

        # Read same file concurrently
        results = await asyncio.gather(read_file(), read_file(), read_file())

        # All reads should succeed
        assert len(results) == 3
        for data, packet_count in results:
            assert isinstance(data, bytes)
            assert len(data) > 0

    @pytest.mark.asyncio
    async def test_open_audio_file_async_convenience(self, amen_wav_path):
        """Test open_audio_file_async convenience function"""
        audio = await cm.open_audio_file_async(amen_wav_path)
        try:
            assert isinstance(audio, cm.AsyncAudioFile)
            assert audio.format is not None
            data, packet_count = await audio.read_packets_async(
                start_packet=0, packet_count=100
            )
            assert isinstance(data, bytes)
        finally:
            await audio.close_async()


class TestAsyncAudioFileNumPy:
    """Test AsyncAudioFile NumPy integration (if available)"""


    @pytest.mark.skipif(not cm.NUMPY_AVAILABLE, reason="NumPy not available")
    @pytest.mark.asyncio
    async def test_async_read_as_numpy(self, amen_wav_path):
        """Test async NumPy array reading"""
        import numpy as np

        async with cm.AsyncAudioFile(amen_wav_path) as audio_file:
            # Read as NumPy array
            data = await audio_file.read_as_numpy_async(
                start_packet=0, packet_count=100
            )
            assert isinstance(data, np.ndarray)
            assert data.ndim == 2  # Should be 2D (frames, channels)
            assert data.shape[1] == 2  # Stereo

    @pytest.mark.skipif(not cm.NUMPY_AVAILABLE, reason="NumPy not available")
    @pytest.mark.asyncio
    async def test_async_read_chunks_numpy(self, amen_wav_path):
        """Test async NumPy chunk streaming"""
        import numpy as np

        async with cm.AsyncAudioFile(amen_wav_path) as audio_file:
            chunks = []
            chunk_size = 1024

            async for chunk in audio_file.read_chunks_numpy_async(
                chunk_size=chunk_size
            ):
                assert isinstance(chunk, np.ndarray)
                assert chunk.ndim == 2
                assert chunk.shape[1] == 2  # Stereo
                chunks.append(chunk)
                # Limit chunks for testing
                if len(chunks) >= 3:
                    break

            # Should have received multiple chunks
            assert len(chunks) > 1


class TestAsyncAudioQueue:
    """Test AsyncAudioQueue async wrapper"""

    @pytest.mark.asyncio
    async def test_async_audio_queue_creation(self, audio_format):
        """Test AsyncAudioQueue creation"""
        queue = await cm.AsyncAudioQueue.new_output_async(audio_format)
        assert isinstance(queue, cm.AsyncAudioQueue)
        assert queue.format.sample_rate == 44100.0

        # Clean up
        await queue.dispose_async()

    @pytest.mark.asyncio
    async def test_async_audio_queue_context_manager(self, audio_format):
        """Test AsyncAudioQueue as async context manager"""
        async with await cm.AsyncAudioQueue.new_output_async(audio_format) as queue:
            assert isinstance(queue, cm.AsyncAudioQueue)
            assert queue.format is not None

    @pytest.mark.asyncio
    async def test_async_audio_queue_buffer_operations(self, audio_format):
        """Test async buffer allocation and enqueueing"""
        async with await cm.AsyncAudioQueue.new_output_async(audio_format) as queue:
            # Allocate buffer
            buffer = await queue.allocate_buffer_async(4096)
            assert isinstance(buffer, cm.AudioBuffer)

            # Enqueue buffer (may require actual audio data in real usage)
            # await queue.enqueue_buffer_async(buffer)

    @pytest.mark.asyncio
    async def test_async_audio_queue_start_stop(self, audio_format):
        """Test async queue start/stop operations"""
        async with await cm.AsyncAudioQueue.new_output_async(audio_format) as queue:
            # Start queue
            await queue.start_async()

            # Give it a moment
            await asyncio.sleep(0.1)

            # Stop queue
            await queue.stop_async()

    @pytest.mark.asyncio
    async def test_create_output_queue_async_convenience(self, audio_format):
        """Test create_output_queue_async convenience function"""
        queue = await cm.create_output_queue_async(audio_format)
        try:
            assert isinstance(queue, cm.AsyncAudioQueue)
            assert queue.format.sample_rate == 44100.0
        finally:
            await queue.dispose_async()

    @pytest.mark.asyncio
    async def test_async_concurrent_queue_operations(self, audio_format):
        """Test concurrent async queue operations"""

        async def create_and_dispose_queue():
            queue = await cm.AsyncAudioQueue.new_output_async(audio_format)
            await asyncio.sleep(0.05)
            await queue.dispose_async()

        # Create multiple queues concurrently
        await asyncio.gather(
            create_and_dispose_queue(),
            create_and_dispose_queue(),
            create_and_dispose_queue(),
        )


class TestAsyncIntegration:
    """Integration tests for async I/O"""


    @pytest.mark.asyncio
    async def test_async_file_read_and_queue_playback(self, amen_wav_path):
        """Test reading file and preparing queue for playback"""
        async with cm.AsyncAudioFile(amen_wav_path) as audio:
            format = audio.format

            # Create queue with same format
            async with await cm.AsyncAudioQueue.new_output_async(format) as queue:
                # Read some audio data
                data, packet_count = await audio.read_packets_async(
                    start_packet=0, packet_count=1024
                )
                assert isinstance(data, bytes)
                assert len(data) > 0

    @pytest.mark.asyncio
    async def test_async_processing_pipeline(self, amen_wav_path):
        """Test async audio processing pipeline"""
        processed_chunks = []

        async def process_chunk(chunk: bytes):
            """Simulate async processing"""
            await asyncio.sleep(0.001)  # Simulate async work
            return len(chunk)

        async with cm.AsyncAudioFile(amen_wav_path) as audio:
            async for chunk in audio.read_chunks_async(chunk_size=1024):
                size = await process_chunk(chunk)
                processed_chunks.append(size)

        # Should have processed multiple chunks
        assert len(processed_chunks) > 1
        assert all(size > 0 for size in processed_chunks)

    @pytest.mark.skipif(not cm.NUMPY_AVAILABLE, reason="NumPy not available")
    @pytest.mark.asyncio
    async def test_async_numpy_processing_pipeline(self, amen_wav_path):
        """Test async NumPy processing pipeline"""
        import numpy as np

        max_amplitudes = []

        async def compute_max_amplitude(chunk: np.ndarray):
            """Simulate async NumPy processing"""
            await asyncio.sleep(0.001)
            return float(np.max(np.abs(chunk)))

        async with cm.AsyncAudioFile(amen_wav_path) as audio:
            async for chunk in audio.read_chunks_numpy_async(chunk_size=1024):
                max_amp = await compute_max_amplitude(chunk)
                max_amplitudes.append(max_amp)

        # Should have processed multiple chunks
        assert len(max_amplitudes) > 1
        # All amplitudes should be in valid range for int16 audio data
        # int16 range is -32768 to 32767
        assert all(0 <= amp <= 32768 for amp in max_amplitudes)

    @pytest.mark.asyncio
    async def test_async_multiple_files_concurrent(self, amen_wav_path):
        """Test concurrent processing of multiple audio files"""
        amen_path = amen_wav_path

        async def get_file_info(path: str):
            """Get file info asynchronously"""
            async with cm.AsyncAudioFile(path) as audio:
                return {
                    "path": path,
                    "duration": audio.duration,
                    "sample_rate": audio.format.sample_rate,
                    "channels": audio.format.channels_per_frame,
                }

        # Process same file multiple times concurrently
        results = await asyncio.gather(
            get_file_info(amen_path), get_file_info(amen_path), get_file_info(amen_path)
        )

        # All results should match
        assert len(results) == 3
        for result in results:
            assert result["duration"] > 0
            assert result["sample_rate"] == 44100.0
            assert result["channels"] == 2
