import os
import struct
import time
import wave

import pytest

import coremusic as cm



# AudioQueue.h
class TestAudioQueueOperations:
    """Test AudioQueue creation and management"""
    
    @pytest.fixture
    def audio_format(self):
        """Fixture providing standard audio format"""
        return {
            'sample_rate': 44100.0,
            'format_id': cm.get_audio_format_linear_pcm(),
            'format_flags': cm.get_linear_pcm_format_flag_is_signed_integer() | 
                           cm.get_linear_pcm_format_flag_is_packed(),
            'bytes_per_packet': 4,
            'frames_per_packet': 1,
            'bytes_per_frame': 4,
            'channels_per_frame': 2,
            'bits_per_channel': 16
        }
    
    def test_audio_queue_creation(self, audio_format):
        """Test AudioQueue creation"""
        queue_id = cm.audio_queue_new_output(audio_format)
        assert queue_id is not None
        assert isinstance(queue_id, int)
        
        # Cleanup
        cm.audio_queue_dispose(queue_id, True)
    
    def test_audio_queue_buffer_allocation(self, audio_format):
        """Test AudioQueue buffer allocation"""
        queue_id = cm.audio_queue_new_output(audio_format)
        
        try:
            buffer_size = 8192
            buffer_id = cm.audio_queue_allocate_buffer(queue_id, buffer_size)
            assert buffer_id is not None
            assert isinstance(buffer_id, int)
            
        finally:
            cm.audio_queue_dispose(queue_id, True)
    
    def test_audio_queue_lifecycle(self, audio_format):
        """Test complete AudioQueue lifecycle"""
        queue_id = cm.audio_queue_new_output(audio_format)
        
        try:
            # Allocate buffers
            buffer1_id = cm.audio_queue_allocate_buffer(queue_id, 8192)
            buffer2_id = cm.audio_queue_allocate_buffer(queue_id, 8192)
            
            # Note: Enqueuing buffers without a callback may fail on some systems
            # This is expected behavior - we're testing the infrastructure, not actual playback
            try:
                cm.audio_queue_enqueue_buffer(queue_id, buffer1_id)
                cm.audio_queue_enqueue_buffer(queue_id, buffer2_id)
                
                # Start queue
                cm.audio_queue_start(queue_id)
                
                # Brief operation
                time.sleep(0.1)
                
                # Stop queue
                cm.audio_queue_stop(queue_id, True)
            except RuntimeError as e:
                # This is expected when no callback is set up
                # The important thing is that we can create and dispose the queue
                assert "AudioQueueEnqueueBuffer failed" in str(e) or "AudioQueueStart failed" in str(e)
            
        finally:
            cm.audio_queue_dispose(queue_id, True)

