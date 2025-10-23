"""Tests for AudioQueue and AudioBuffer object-oriented classes."""

import pytest
import time
import coremusic as cm
import coremusic.capi as capi


@pytest.fixture(scope="module")
def skip_if_no_audio_hardware():
    """Helper to skip tests that require audio hardware when unavailable"""
    def _skip_check():
        try:
            format = cm.AudioFormat(
                sample_rate=44100.0,
                format_id="lpcm",
                channels_per_frame=2,
                bits_per_channel=16,
            )
            queue = cm.AudioQueue.new_output(format)
            queue.dispose()
        except cm.AudioQueueError as e:
            # Check for paramErr (-50) which indicates no audio hardware
            if e.status_code == -50 or "paramErr" in str(e):
                pytest.skip("Audio hardware not available")
            raise
    return _skip_check


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
            format_id="lpcm",
            channels_per_frame=2,
            bits_per_channel=16,
        )
        queue = cm.AudioQueue(format)
        assert isinstance(queue, cm.AudioQueue)
        assert isinstance(queue, cm.CoreAudioObject)
        assert queue._format is format
        assert not queue.is_disposed
        assert len(queue._buffers) == 0

    def test_audio_queue_new_output_factory(self, skip_if_no_audio_hardware):
        """Test AudioQueue.new_output factory method"""
        skip_if_no_audio_hardware()
        format = cm.AudioFormat(
            sample_rate=48000.0,
            format_id="lpcm",
            format_flags=12,
            channels_per_frame=2,
            bits_per_channel=16,
        )
        queue = cm.AudioQueue.new_output(format)
        assert isinstance(queue, cm.AudioQueue)
        assert queue._format is format
        assert queue.object_id != 0
        queue.dispose()

    def test_audio_queue_buffer_allocation(self, skip_if_no_audio_hardware):
        """Test AudioQueue buffer allocation"""
        skip_if_no_audio_hardware()
        format = cm.AudioFormat(
            44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16
        )
        queue = cm.AudioQueue.new_output(format)
        try:
            buffer = queue.allocate_buffer(1024)
            assert isinstance(buffer, cm.AudioBuffer)
            assert buffer.buffer_size == 1024
            assert buffer._queue_id == queue.object_id
            assert buffer.object_id != 0
            assert len(queue._buffers) == 1
            assert queue._buffers[0] is buffer
        finally:
            queue.dispose()

    def test_audio_queue_buffer_enqueue(self, skip_if_no_audio_hardware):
        """Test AudioQueue buffer enqueuing"""
        skip_if_no_audio_hardware()
        format = cm.AudioFormat(
            44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16
        )
        queue = cm.AudioQueue.new_output(format)
        try:
            buffer = queue.allocate_buffer(1024)
            queue.enqueue_buffer(buffer)
        finally:
            queue.dispose()

    def test_audio_queue_playback_control(self, skip_if_no_audio_hardware):
        """Test AudioQueue playback control methods"""
        skip_if_no_audio_hardware()
        format = cm.AudioFormat(
            44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16
        )
        queue = cm.AudioQueue.new_output(format)
        try:
            queue.start()
            time.sleep(0.01)
            queue.stop()
            queue.stop(immediate=False)
        finally:
            queue.dispose()

    def test_audio_queue_disposal(self, skip_if_no_audio_hardware):
        """Test AudioQueue disposal"""
        skip_if_no_audio_hardware()
        format = cm.AudioFormat(
            44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16
        )
        queue = cm.AudioQueue.new_output(format)
        buffer1 = queue.allocate_buffer(1024)
        buffer2 = queue.allocate_buffer(2048)
        assert len(queue._buffers) == 2
        assert not queue.is_disposed
        queue.dispose()
        assert queue.is_disposed
        assert len(queue._buffers) == 0

    def test_audio_queue_disposal_with_immediate_flag(self, skip_if_no_audio_hardware):
        """Test AudioQueue disposal with immediate flag"""
        skip_if_no_audio_hardware()
        format = cm.AudioFormat(
            44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16
        )
        queue = cm.AudioQueue.new_output(format)
        try:
            queue.start()
            time.sleep(0.01)
            queue.dispose(immediate=False)
            assert queue.is_disposed
        except cm.AudioQueueError:
            if not queue.is_disposed:
                queue.dispose(immediate=True)

    def test_audio_queue_operations_on_disposed_object(self, skip_if_no_audio_hardware):
        """Test operations on disposed AudioQueue"""
        skip_if_no_audio_hardware()
        format = cm.AudioFormat(
            44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16
        )
        queue = cm.AudioQueue.new_output(format)
        queue.dispose()
        with pytest.raises(RuntimeError, match="has been disposed"):
            queue.allocate_buffer(1024)
        with pytest.raises(RuntimeError, match="has been disposed"):
            queue.start()
        with pytest.raises(RuntimeError, match="has been disposed"):
            queue.stop()

    def test_audio_queue_error_handling(self):
        """Test AudioQueue error handling"""
        invalid_format = cm.AudioFormat(
            0.0, "", channels_per_frame=0, bits_per_channel=0
        )
        with pytest.raises(cm.AudioQueueError):
            cm.AudioQueue.new_output(invalid_format)

    def test_audio_queue_vs_functional_api_consistency(self, skip_if_no_audio_hardware):
        """Test AudioQueue OO API vs functional API consistency"""
        skip_if_no_audio_hardware()
        format_dict = {
            "sample_rate": 44100.0,
            "format_id": "lpcm",
            "format_flags": 0,
            "bytes_per_packet": 0,
            "frames_per_packet": 0,
            "bytes_per_frame": 0,
            "channels_per_frame": 2,
            "bits_per_channel": 16,
        }
        func_queue_id = capi.audio_queue_new_output(format_dict)
        try:
            func_buffer_id = capi.audio_queue_allocate_buffer(func_queue_id, 1024)
        finally:
            capi.audio_queue_dispose(func_queue_id)
        format = cm.AudioFormat(
            44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16
        )
        oo_queue = cm.AudioQueue.new_output(format)
        try:
            oo_buffer = oo_queue.allocate_buffer(1024)
            assert func_queue_id != 0
            assert func_buffer_id != 0
            assert oo_queue.object_id != 0
            assert oo_buffer.object_id != 0
        finally:
            oo_queue.dispose()


class TestAudioQueueIntegration:
    """Integration tests for AudioQueue functionality"""

    def test_audio_queue_full_workflow(self, skip_if_no_audio_hardware):
        """Test complete AudioQueue workflow"""
        skip_if_no_audio_hardware()
        format = cm.AudioFormat(
            44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16
        )
        queue = cm.AudioQueue.new_output(format)
        try:
            buffers = []
            for i in range(3):
                buffer = queue.allocate_buffer(1024 * (i + 1))
                buffers.append(buffer)
            assert len(queue._buffers) == 3
            assert all(isinstance(b, cm.AudioBuffer) for b in buffers)
            for buffer in buffers:
                queue.enqueue_buffer(buffer)
            queue.start()
            time.sleep(0.01)
            queue.stop()
        finally:
            queue.dispose()

    def test_audio_queue_multiple_instances(self, skip_if_no_audio_hardware):
        """Test creating multiple AudioQueue instances"""
        skip_if_no_audio_hardware()
        format1 = cm.AudioFormat(
            44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16
        )
        format2 = cm.AudioFormat(
            48000.0, "lpcm", channels_per_frame=1, bits_per_channel=24
        )
        queue1 = cm.AudioQueue.new_output(format1)
        queue2 = cm.AudioQueue.new_output(format2)
        try:
            assert queue1.object_id != queue2.object_id
            assert queue1._format is format1
            assert queue2._format is format2
            buffer1 = queue1.allocate_buffer(1024)
            buffer2 = queue2.allocate_buffer(2048)
            assert buffer1.buffer_size == 1024
            assert buffer2.buffer_size == 2048
        finally:
            queue1.dispose()
            queue2.dispose()

    def test_audio_queue_resource_management(self, skip_if_no_audio_hardware):
        """Test AudioQueue resource management under stress"""
        skip_if_no_audio_hardware()
        format = cm.AudioFormat(
            44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16
        )
        for i in range(5):
            queue = cm.AudioQueue.new_output(format)
            for j in range(3):
                queue.allocate_buffer(512)
            queue.start()
            time.sleep(0.001)
            queue.stop()
            queue.dispose()
            assert queue.is_disposed

    def test_audio_queue_error_recovery(self, skip_if_no_audio_hardware):
        """Test AudioQueue error handling and recovery"""
        skip_if_no_audio_hardware()
        format = cm.AudioFormat(
            44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16
        )
        queue = cm.AudioQueue.new_output(format)
        try:
            with pytest.raises(cm.AudioQueueError):
                queue.allocate_buffer(1024 * 1024 * 1024)
            normal_buffer = queue.allocate_buffer(1024)
            assert isinstance(normal_buffer, cm.AudioBuffer)
        finally:
            queue.dispose()
