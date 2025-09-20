#!/usr/bin/env python3
"""Tests for AudioQueue and AudioBuffer object-oriented classes."""

import pytest
import time

import coremusic as cm

# Skip all AudioQueue tests until hardware is available
pytestmark = pytest.mark.skip(reason="AudioQueue tests require audio hardware")


class TestAudioBuffer:
    """Test AudioBuffer object-oriented wrapper"""

    def test_audio_buffer_creation(self):
        """Test AudioBuffer creation"""
        buffer = cm.AudioBuffer(queue_id=12345, buffer_size=1024)

        assert isinstance(buffer, cm.AudioBuffer)
        assert isinstance(buffer, cm.CoreAudioObject)
        assert buffer._queue_id == 12345
        assert buffer.buffer_size == 1024

    def test_audio_buffer_properties(self):
        """Test AudioBuffer properties"""
        buffer = cm.AudioBuffer(queue_id=54321, buffer_size=2048)

        assert buffer.buffer_size == 2048
        assert buffer._queue_id == 54321


class TestAudioQueue:
    """Test AudioQueue object-oriented wrapper"""

    def test_audio_queue_creation_with_format(self):
        """Test AudioQueue creation with AudioFormat"""
        format = cm.AudioFormat(
            sample_rate=44100.0,
            format_id='lpcm',
            channels_per_frame=2,
            bits_per_channel=16
        )

        queue = cm.AudioQueue(format)
        assert isinstance(queue, cm.AudioQueue)
        assert isinstance(queue, cm.CoreAudioObject)
        assert queue._format is format
        assert not queue.is_disposed
        assert len(queue._buffers) == 0

    def test_audio_queue_new_output_factory(self):
        """Test AudioQueue.new_output factory method"""
        format = cm.AudioFormat(
            sample_rate=48000.0,
            format_id='lpcm',
            format_flags=12,
            channels_per_frame=2,
            bits_per_channel=16
        )

        try:
            queue = cm.AudioQueue.new_output(format)
            assert isinstance(queue, cm.AudioQueue)
            assert queue._format is format
            assert queue.object_id != 0  # Should have created actual queue

            # Clean up
            queue.dispose()
        except cm.AudioQueueError as e:
            if "status: -50" in str(e):
                pytest.skip("AudioQueue creation failed - no audio hardware available")
            else:
                raise

    def test_audio_queue_buffer_allocation(self):
        """Test AudioQueue buffer allocation"""
        format = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)
        queue = cm.AudioQueue.new_output(format)

        try:
            # Allocate a buffer
            buffer = queue.allocate_buffer(1024)

            assert isinstance(buffer, cm.AudioBuffer)
            assert buffer.buffer_size == 1024
            assert buffer._queue_id == queue.object_id
            assert buffer.object_id != 0  # Should have actual buffer ID

            # Buffer should be tracked by queue
            assert len(queue._buffers) == 1
            assert queue._buffers[0] is buffer

        finally:
            queue.dispose()

    def test_audio_queue_buffer_enqueue(self):
        """Test AudioQueue buffer enqueuing"""
        format = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)
        queue = cm.AudioQueue.new_output(format)

        try:
            buffer = queue.allocate_buffer(1024)

            # Enqueue the buffer (should not raise)
            queue.enqueue_buffer(buffer)

        finally:
            queue.dispose()

    def test_audio_queue_playback_control(self):
        """Test AudioQueue playback control methods"""
        format = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)
        queue = cm.AudioQueue.new_output(format)

        try:
            # Test start (should not raise)
            queue.start()

            # Brief pause to let it start
            time.sleep(0.01)

            # Test stop
            queue.stop()
            queue.stop(immediate=False)  # Test with immediate=False

        finally:
            queue.dispose()

    def test_audio_queue_disposal(self):
        """Test AudioQueue disposal"""
        format = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)
        queue = cm.AudioQueue.new_output(format)

        # Allocate some buffers
        buffer1 = queue.allocate_buffer(1024)
        buffer2 = queue.allocate_buffer(2048)

        assert len(queue._buffers) == 2
        assert not queue.is_disposed

        # Dispose queue
        queue.dispose()

        assert queue.is_disposed
        assert len(queue._buffers) == 0  # Buffers should be cleared

    def test_audio_queue_disposal_with_immediate_flag(self):
        """Test AudioQueue disposal with immediate flag"""
        format = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)
        queue = cm.AudioQueue.new_output(format)

        try:
            queue.start()
            time.sleep(0.01)

            # Test disposal with immediate=False
            queue.dispose(immediate=False)
            assert queue.is_disposed

        except cm.AudioQueueError:
            # If disposal fails, make sure we still clean up
            if not queue.is_disposed:
                queue.dispose(immediate=True)

    def test_audio_queue_operations_on_disposed_object(self):
        """Test operations on disposed AudioQueue"""
        format = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)
        queue = cm.AudioQueue.new_output(format)
        queue.dispose()

        # Operations on disposed queue should raise
        with pytest.raises(RuntimeError, match="has been disposed"):
            queue.allocate_buffer(1024)

        with pytest.raises(RuntimeError, match="has been disposed"):
            queue.start()

        with pytest.raises(RuntimeError, match="has been disposed"):
            queue.stop()

    def test_audio_queue_error_handling(self):
        """Test AudioQueue error handling"""
        # Test with invalid format that might cause creation to fail
        invalid_format = cm.AudioFormat(0.0, '', channels_per_frame=0, bits_per_channel=0)

        with pytest.raises(cm.AudioQueueError):
            cm.AudioQueue.new_output(invalid_format)

    def test_audio_queue_vs_functional_api_consistency(self):
        """Test AudioQueue OO API vs functional API consistency"""
        # Test that OO API creates equivalent results to functional API
        format_dict = {
            'sample_rate': 44100.0,
            'format_id': 'lpcm',
            'format_flags': 0,
            'bytes_per_packet': 0,
            'frames_per_packet': 0,
            'bytes_per_frame': 0,
            'channels_per_frame': 2,
            'bits_per_channel': 16
        }

        # Functional API
        func_queue_id = cm.audio_queue_new_output(format_dict)
        try:
            func_buffer_id = cm.audio_queue_allocate_buffer(func_queue_id, 1024)
        finally:
            cm.audio_queue_dispose(func_queue_id)

        # OO API
        format = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)
        oo_queue = cm.AudioQueue.new_output(format)
        try:
            oo_buffer = oo_queue.allocate_buffer(1024)

            # Both should succeed and produce valid IDs
            assert func_queue_id != 0
            assert func_buffer_id != 0
            assert oo_queue.object_id != 0
            assert oo_buffer.object_id != 0

        finally:
            oo_queue.dispose()


class TestAudioQueueIntegration:
    """Integration tests for AudioQueue functionality"""

    def test_audio_queue_full_workflow(self):
        """Test complete AudioQueue workflow"""
        format = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)
        queue = cm.AudioQueue.new_output(format)

        try:
            # Allocate multiple buffers
            buffers = []
            for i in range(3):
                buffer = queue.allocate_buffer(1024 * (i + 1))
                buffers.append(buffer)

            assert len(queue._buffers) == 3
            assert all(isinstance(b, cm.AudioBuffer) for b in buffers)

            # Enqueue all buffers
            for buffer in buffers:
                queue.enqueue_buffer(buffer)

            # Start playback
            queue.start()
            time.sleep(0.01)  # Brief playback

            # Stop playback
            queue.stop()

        finally:
            queue.dispose()

    def test_audio_queue_multiple_instances(self):
        """Test creating multiple AudioQueue instances"""
        format1 = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)
        format2 = cm.AudioFormat(48000.0, 'lpcm', channels_per_frame=1, bits_per_channel=24)

        queue1 = cm.AudioQueue.new_output(format1)
        queue2 = cm.AudioQueue.new_output(format2)

        try:
            # Both queues should be independent
            assert queue1.object_id != queue2.object_id
            assert queue1._format is format1
            assert queue2._format is format2

            # Both should work independently
            buffer1 = queue1.allocate_buffer(1024)
            buffer2 = queue2.allocate_buffer(2048)

            assert buffer1.buffer_size == 1024
            assert buffer2.buffer_size == 2048

        finally:
            queue1.dispose()
            queue2.dispose()

    def test_audio_queue_resource_management(self):
        """Test AudioQueue resource management under stress"""
        format = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)

        # Create and dispose multiple queues
        for i in range(5):
            queue = cm.AudioQueue.new_output(format)

            # Allocate some buffers
            for j in range(3):
                queue.allocate_buffer(512)

            # Start and stop
            queue.start()
            time.sleep(0.001)
            queue.stop()

            # Dispose
            queue.dispose()
            assert queue.is_disposed

    def test_audio_queue_error_recovery(self):
        """Test AudioQueue error handling and recovery"""
        format = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)
        queue = cm.AudioQueue.new_output(format)

        try:
            # Try to allocate an unreasonably large buffer (should fail)
            with pytest.raises(cm.AudioQueueError):
                queue.allocate_buffer(1024 * 1024 * 1024)  # 1GB buffer

            # Queue should still be functional after error
            normal_buffer = queue.allocate_buffer(1024)
            assert isinstance(normal_buffer, cm.AudioBuffer)

        finally:
            queue.dispose()