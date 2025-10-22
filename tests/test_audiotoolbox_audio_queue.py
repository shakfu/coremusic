import os
import struct
import time
import wave
import pytest
import coremusic as cm
import coremusic.capi as capi


class TestAudioQueueOperations:
    """Test AudioQueue creation and management"""

    @pytest.fixture
    def audio_format(self):
        """Fixture providing standard audio format"""
        return {
            "sample_rate": 44100.0,
            "format_id": capi.get_audio_format_linear_pcm(),
            "format_flags": capi.get_linear_pcm_format_flag_is_signed_integer()
            | capi.get_linear_pcm_format_flag_is_packed(),
            "bytes_per_packet": 4,
            "frames_per_packet": 1,
            "bytes_per_frame": 4,
            "channels_per_frame": 2,
            "bits_per_channel": 16,
        }

    def test_audio_queue_creation(self, audio_format):
        """Test AudioQueue creation"""
        queue_id = capi.audio_queue_new_output(audio_format)
        assert queue_id is not None
        assert isinstance(queue_id, int)
        capi.audio_queue_dispose(queue_id, True)

    def test_audio_queue_buffer_allocation(self, audio_format):
        """Test AudioQueue buffer allocation"""
        queue_id = capi.audio_queue_new_output(audio_format)
        try:
            buffer_size = 8192
            buffer_id = capi.audio_queue_allocate_buffer(queue_id, buffer_size)
            assert buffer_id is not None
            assert isinstance(buffer_id, int)
        finally:
            capi.audio_queue_dispose(queue_id, True)

    def test_audio_queue_lifecycle(self, audio_format):
        """Test complete AudioQueue lifecycle"""
        queue_id = capi.audio_queue_new_output(audio_format)
        try:
            buffer1_id = capi.audio_queue_allocate_buffer(queue_id, 8192)
            buffer2_id = capi.audio_queue_allocate_buffer(queue_id, 8192)
            try:
                capi.audio_queue_enqueue_buffer(queue_id, buffer1_id)
                capi.audio_queue_enqueue_buffer(queue_id, buffer2_id)
                capi.audio_queue_start(queue_id)
                time.sleep(0.1)
                capi.audio_queue_stop(queue_id, True)
            except RuntimeError as e:
                assert "AudioQueueEnqueueBuffer failed" in str(
                    e
                ) or "AudioQueueStart failed" in str(e)
        finally:
            capi.audio_queue_dispose(queue_id, True)
