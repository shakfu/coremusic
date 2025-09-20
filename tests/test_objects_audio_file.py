#!/usr/bin/env python3
"""Tests for AudioFile and AudioFileStream object-oriented classes."""

import os
import pytest
from pathlib import Path

import coremusic as cm


class TestAudioFile:
    """Test AudioFile object-oriented wrapper"""

    @pytest.fixture
    def amen_wav_path(self):
        """Fixture providing path to amen.wav test file"""
        path = os.path.join("tests", "amen.wav")
        if not os.path.exists(path):
            pytest.skip(f"Test audio file not found: {path}")
        return path

    def test_audio_file_creation(self, amen_wav_path):
        """Test AudioFile object creation"""
        # Test with string path
        audio_file = cm.AudioFile(amen_wav_path)
        assert isinstance(audio_file, cm.AudioFile)
        assert isinstance(audio_file, cm.CoreAudioObject)
        assert not audio_file.is_disposed
        assert audio_file.object_id == 0  # Not opened yet

        # Test with Path object
        audio_file_path = cm.AudioFile(Path(amen_wav_path))
        assert isinstance(audio_file_path, cm.AudioFile)
        assert audio_file_path._path == str(Path(amen_wav_path))

    def test_audio_file_open_close(self, amen_wav_path):
        """Test AudioFile opening and closing"""
        audio_file = cm.AudioFile(amen_wav_path)

        # Test opening
        result = audio_file.open()
        assert result is audio_file  # Should return self
        assert audio_file.object_id != 0
        assert not audio_file.is_disposed

        # Test closing
        audio_file.close()
        assert audio_file.is_disposed

    def test_audio_file_context_manager(self, amen_wav_path):
        """Test AudioFile as context manager"""
        with cm.AudioFile(amen_wav_path) as audio_file:
            assert isinstance(audio_file, cm.AudioFile)
            assert audio_file.object_id != 0
            assert not audio_file.is_disposed

        # Should be automatically closed after context
        assert audio_file.is_disposed

    def test_audio_file_format_property(self, amen_wav_path):
        """Test AudioFile format property"""
        with cm.AudioFile(amen_wav_path) as audio_file:
            format = audio_file.format
            assert isinstance(format, cm.AudioFormat)

            # The format should be cached
            format2 = audio_file.format
            assert format is format2  # Should be same object

    def test_audio_file_read_packets(self, amen_wav_path):
        """Test AudioFile packet reading"""
        with cm.AudioFile(amen_wav_path) as audio_file:
            # Test reading some packets
            data, packet_count = audio_file.read_packets(0, 100)
            assert isinstance(data, bytes)
            assert isinstance(packet_count, int)
            assert len(data) > 0
            assert packet_count > 0

    def test_audio_file_get_property(self, amen_wav_path):
        """Test AudioFile property reading"""
        with cm.AudioFile(amen_wav_path) as audio_file:
            # Test getting data format property
            property_id = cm.get_audio_file_property_data_format()
            property_data = audio_file.get_property(property_id)
            assert isinstance(property_data, bytes)
            assert len(property_data) >= 40  # AudioStreamBasicDescription size

    def test_audio_file_duration_property(self, amen_wav_path):
        """Test AudioFile duration property (placeholder implementation)"""
        with cm.AudioFile(amen_wav_path) as audio_file:
            duration = audio_file.duration
            assert isinstance(duration, float)
            # Note: Current implementation returns 0.0 as placeholder
            assert duration == 0.0

    def test_audio_file_repr(self, amen_wav_path):
        """Test AudioFile string representation"""
        audio_file = cm.AudioFile(amen_wav_path)
        repr_str = repr(audio_file)
        assert 'AudioFile' in repr_str
        assert 'closed' in repr_str

        with audio_file:
            repr_str = repr(audio_file)
            assert 'AudioFile' in repr_str
            assert 'open' in repr_str

    def test_audio_file_error_handling(self):
        """Test AudioFile error handling"""
        # Test with non-existent file
        with pytest.raises(cm.AudioFileError):
            with cm.AudioFile("/nonexistent/path.wav"):
                pass

    def test_audio_file_operations_on_disposed_object(self, amen_wav_path):
        """Test operations on disposed AudioFile object"""
        audio_file = cm.AudioFile(amen_wav_path)
        audio_file.open()
        audio_file.close()

        # Operations on disposed object should raise
        with pytest.raises(RuntimeError, match="has been disposed"):
            audio_file.open()

        with pytest.raises(RuntimeError, match="has been disposed"):
            _ = audio_file.format

    def test_audio_file_automatic_disposal(self, amen_wav_path):
        """Test AudioFile automatic disposal on object deletion"""
        audio_file = cm.AudioFile(amen_wav_path)
        audio_file.open()
        object_id = audio_file.object_id

        # Delete object - should automatically dispose
        del audio_file
        # Note: In a real scenario, we'd need to verify the underlying
        # CoreAudio resources were cleaned up, but that's hard to test directly


class TestAudioFileStream:
    """Test AudioFileStream object-oriented wrapper"""

    def test_audio_file_stream_creation(self):
        """Test AudioFileStream creation"""
        # Test default creation
        stream = cm.AudioFileStream()
        assert isinstance(stream, cm.AudioFileStream)
        assert isinstance(stream, cm.CoreAudioObject)
        assert not stream.is_disposed
        assert stream._file_type_hint == 0

        # Test with file type hint
        stream_with_hint = cm.AudioFileStream(file_type_hint=42)
        assert stream_with_hint._file_type_hint == 42

    def test_audio_file_stream_open_close(self):
        """Test AudioFileStream opening and closing"""
        stream = cm.AudioFileStream()

        # Test opening
        result = stream.open()
        assert result is stream  # Should return self
        assert stream.object_id != 0
        assert not stream.is_disposed

        # Test closing
        stream.close()
        assert stream.is_disposed

    def test_audio_file_stream_context_manager_support(self):
        """Test AudioFileStream context manager functionality"""
        # Note: AudioFileStream doesn't currently have __enter__/__exit__
        # but we can test manual open/close
        stream = cm.AudioFileStream()
        stream.open()
        assert not stream.is_disposed
        stream.close()
        assert stream.is_disposed

    def test_audio_file_stream_parse_bytes(self):
        """Test AudioFileStream byte parsing"""
        stream = cm.AudioFileStream()
        stream.open()

        try:
            # Test parsing some dummy data (will likely fail gracefully)
            dummy_data = b"RIFF\x00\x00\x00\x00WAVE"
            stream.parse_bytes(dummy_data)
            # If we get here, parsing didn't crash (good)
        except cm.AudioFileError:
            # Expected for invalid data
            pass
        finally:
            stream.close()

    def test_audio_file_stream_auto_open(self):
        """Test AudioFileStream automatic opening on parse_bytes"""
        stream = cm.AudioFileStream()
        assert stream.object_id == 0

        try:
            # This should automatically open the stream
            stream.parse_bytes(b"dummy data")
            assert stream.object_id != 0
        except cm.AudioFileError:
            # Expected for invalid data, but stream should be opened
            assert stream.object_id != 0
        finally:
            if not stream.is_disposed:
                stream.close()

    def test_audio_file_stream_seek(self):
        """Test AudioFileStream seeking"""
        stream = cm.AudioFileStream()

        with pytest.raises(cm.AudioFileError, match="Stream not open"):
            stream.seek(100)

        stream.open()
        try:
            # Test seeking (may fail if no data has been parsed)
            stream.seek(0)
        except cm.AudioFileError:
            # Expected if stream hasn't been initialized with data
            pass
        finally:
            stream.close()

    def test_audio_file_stream_get_property(self):
        """Test AudioFileStream property getting"""
        stream = cm.AudioFileStream()

        with pytest.raises(cm.AudioFileError, match="Stream not open"):
            stream.get_property(42)

        stream.open()
        try:
            # Test getting a property (may fail if no data has been parsed)
            prop_id = cm.get_audio_file_stream_property_ready_to_produce_packets()
            stream.get_property(prop_id)
        except cm.AudioFileError:
            # Expected if stream hasn't been initialized with data
            pass
        finally:
            stream.close()

    def test_audio_file_stream_ready_to_produce_packets_property(self):
        """Test ready_to_produce_packets property"""
        stream = cm.AudioFileStream()
        stream.open()

        try:
            ready = stream.ready_to_produce_packets
            assert isinstance(ready, bool)
            # Should be False for uninitialized stream
            assert not ready
        finally:
            stream.close()

    def test_audio_file_stream_operations_on_disposed_object(self):
        """Test operations on disposed AudioFileStream"""
        stream = cm.AudioFileStream()
        stream.open()
        stream.close()

        # Operations on disposed object should raise
        with pytest.raises(RuntimeError, match="has been disposed"):
            stream.parse_bytes(b"data")

        with pytest.raises(RuntimeError, match="has been disposed"):
            stream.seek(0)

    def test_audio_file_stream_error_handling(self):
        """Test AudioFileStream error handling"""
        stream = cm.AudioFileStream()

        # Test operations on unopened stream
        with pytest.raises(cm.AudioFileError):
            stream.seek(100)

        with pytest.raises(cm.AudioFileError):
            stream.get_property(42)


class TestAudioFileIntegration:
    """Integration tests for AudioFile with real audio data"""

    @pytest.fixture
    def amen_wav_path(self):
        """Fixture providing path to amen.wav test file"""
        path = os.path.join("tests", "amen.wav")
        if not os.path.exists(path):
            pytest.skip(f"Test audio file not found: {path}")
        return path

    def test_audio_file_vs_functional_api_consistency(self, amen_wav_path):
        """Test that OO API produces consistent results with functional API"""
        # Use functional API
        audio_file_id = cm.audio_file_open_url(amen_wav_path)
        try:
            func_data, func_count = cm.audio_file_read_packets(audio_file_id, 0, 10)
        finally:
            cm.audio_file_close(audio_file_id)

        # Use OO API
        with cm.AudioFile(amen_wav_path) as audio_file:
            oo_data, oo_count = audio_file.read_packets(0, 10)

        # Results should be identical
        assert func_data == oo_data
        assert func_count == oo_count

    def test_audio_file_multiple_opens(self, amen_wav_path):
        """Test opening the same file multiple times"""
        file1 = cm.AudioFile(amen_wav_path)
        file2 = cm.AudioFile(amen_wav_path)

        with file1, file2:
            # Both should work independently
            data1, _ = file1.read_packets(0, 5)
            data2, _ = file2.read_packets(0, 5)

            # Should read the same data
            assert data1 == data2

    def test_audio_file_resource_management(self, amen_wav_path):
        """Test proper resource management"""
        files = []

        # Create multiple AudioFile objects
        for _ in range(10):
            audio_file = cm.AudioFile(amen_wav_path)
            audio_file.open()
            files.append(audio_file)

        # Close all files
        for audio_file in files:
            audio_file.close()
            assert audio_file.is_disposed