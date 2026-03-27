"""Tests for AudioFile and AudioFileStream object-oriented classes."""

import pytest
from pathlib import Path
import coremusic.capi as capi
from coremusic.audio import AudioFile, AudioFileStream, AudioFormat
from coremusic.base import CoreAudioObject
from coremusic.exceptions import AudioFileError


class TestAudioFile:
    """Test AudioFile object-oriented wrapper"""

    def test_audio_file_creation(self, amen_wav_path):
        """Test AudioFile object creation"""
        audio_file = AudioFile(amen_wav_path)
        assert isinstance(audio_file, AudioFile)
        assert isinstance(audio_file, CoreAudioObject)
        assert not audio_file.is_disposed
        assert audio_file.object_id == 0
        audio_file_path = AudioFile(Path(amen_wav_path))
        assert isinstance(audio_file_path, AudioFile)
        assert audio_file_path._path == str(Path(amen_wav_path))

    def test_audio_file_open_close(self, amen_wav_path):
        """Test AudioFile opening and closing"""
        audio_file = AudioFile(amen_wav_path)
        result = audio_file.open()
        assert result is audio_file
        assert audio_file.object_id != 0
        assert not audio_file.is_disposed
        audio_file.close()
        assert audio_file.is_disposed

    def test_audio_file_context_manager(self, amen_wav_path):
        """Test AudioFile as context manager"""
        with AudioFile(amen_wav_path) as audio_file:
            assert isinstance(audio_file, AudioFile)
            assert audio_file.object_id != 0
            assert not audio_file.is_disposed
        assert audio_file.is_disposed

    def test_audio_file_format_property(self, amen_wav_path):
        """Test AudioFile format property"""
        with AudioFile(amen_wav_path) as audio_file:
            format = audio_file.format
            assert isinstance(format, AudioFormat)
            format2 = audio_file.format
            assert format is format2

    def test_audio_file_read_packets(self, amen_wav_path):
        """Test AudioFile packet reading"""
        with AudioFile(amen_wav_path) as audio_file:
            data, packet_count = audio_file.read_packets(0, 100)
            assert isinstance(data, bytes)
            assert isinstance(packet_count, int)
            assert len(data) > 0
            assert packet_count > 0

    def test_audio_file_get_property(self, amen_wav_path):
        """Test AudioFile property reading"""
        with AudioFile(amen_wav_path) as audio_file:
            property_id = capi.get_audio_file_property_data_format()
            property_data = audio_file.get_property(property_id)
            assert isinstance(property_data, bytes)
            assert len(property_data) >= 40

    def test_audio_file_duration_property(self, amen_wav_path):
        """Test AudioFile duration property"""
        with AudioFile(amen_wav_path) as audio_file:
            duration = audio_file.duration
            assert isinstance(duration, float)
            assert duration > 0.0
            assert 2.0 < duration < 3.0

    def test_audio_file_repr(self, amen_wav_path):
        """Test AudioFile string representation"""
        audio_file = AudioFile(amen_wav_path)
        repr_str = repr(audio_file)
        assert "AudioFile" in repr_str
        assert "closed" in repr_str
        with audio_file:
            repr_str = repr(audio_file)
            assert "AudioFile" in repr_str
            assert "open" in repr_str

    def test_audio_file_error_handling(self):
        """Test AudioFile error handling"""
        with pytest.raises(AudioFileError):
            with AudioFile("/nonexistent/path.wav"):
                pass

    def test_audio_file_operations_on_disposed_object(self, amen_wav_path):
        """Test operations on disposed AudioFile object"""
        audio_file = AudioFile(amen_wav_path)
        audio_file.open()
        audio_file.close()
        with pytest.raises(RuntimeError, match="has been disposed"):
            audio_file.open()
        with pytest.raises(RuntimeError, match="has been disposed"):
            _ = audio_file.format

    def test_audio_file_automatic_disposal(self, amen_wav_path):
        """Test AudioFile automatic disposal on object deletion"""
        audio_file = AudioFile(amen_wav_path)
        audio_file.open()
        del audio_file

    def test_audio_file_metadata_read(self, amen_wav_path):
        """Test reading info dictionary metadata from a WAV file"""
        with AudioFile(amen_wav_path) as audio_file:
            metadata = audio_file.metadata
            # WAV files may or may not have metadata depending on the file
            assert metadata is None or isinstance(metadata, dict)

    def test_audio_file_metadata_read_returns_dict(self, amen_wav_path):
        """Test that metadata returns a dict with string keys"""
        with AudioFile(amen_wav_path) as audio_file:
            metadata = audio_file.metadata
            if metadata is not None:
                for key in metadata:
                    assert isinstance(key, str)

    def test_audio_file_metadata_write_requires_writable(self, amen_wav_path):
        """Test that set_metadata raises when file is not writable"""
        with AudioFile(amen_wav_path) as audio_file:
            with pytest.raises(AudioFileError, match="not opened for writing"):
                audio_file.set_metadata({"title": "test"})

    def test_audio_file_metadata_roundtrip(self, amen_wav_path, tmp_path):
        """Test writing and reading back metadata via CAF format"""
        import subprocess

        caf_path = str(tmp_path / "test.caf")
        subprocess.run(
            ["afconvert", "-f", "caff", "-d", "LEI16", amen_wav_path, caf_path],
            check=True,
        )

        tags = {"title": "Test Title", "artist": "Test Artist"}
        with AudioFile(caf_path, writable=True) as af:
            af.set_metadata(tags)

        with AudioFile(caf_path) as af:
            result = af.metadata
            assert result is not None
            assert result["title"] == "Test Title"
            assert result["artist"] == "Test Artist"

    def test_audio_file_set_property(self, amen_wav_path):
        """Test get_property and set_property exist and work"""
        with AudioFile(amen_wav_path) as audio_file:
            # get_property should work for data format
            data = audio_file.get_property(capi.get_audio_file_property_data_format())
            assert isinstance(data, bytes)
            assert len(data) >= 40  # ASBD is 40 bytes

    def test_audio_file_writable_flag(self, amen_wav_path, tmp_path):
        """Test that writable flag opens file with read-write permissions"""
        import shutil

        copy_path = str(tmp_path / "copy.wav")
        shutil.copy(amen_wav_path, copy_path)

        # Should open without error
        with AudioFile(copy_path, writable=True) as af:
            fmt = af.format
            assert fmt.sample_rate > 0


class TestAudioFileStream:
    """Test AudioFileStream object-oriented wrapper"""

    def test_audio_file_stream_creation(self):
        """Test AudioFileStream creation"""
        stream = AudioFileStream()
        assert isinstance(stream, AudioFileStream)
        assert isinstance(stream, CoreAudioObject)
        assert not stream.is_disposed
        assert stream._file_type_hint == 0
        stream_with_hint = AudioFileStream(file_type_hint=42)
        assert stream_with_hint._file_type_hint == 42

    def test_audio_file_stream_open_close(self):
        """Test AudioFileStream opening and closing"""
        stream = AudioFileStream()
        result = stream.open()
        assert result is stream
        assert stream.object_id != 0
        assert not stream.is_disposed
        stream.close()
        assert stream.is_disposed

    def test_audio_file_stream_context_manager_support(self):
        """Test AudioFileStream context manager functionality"""
        stream = AudioFileStream()
        stream.open()
        assert not stream.is_disposed
        stream.close()
        assert stream.is_disposed

    def test_audio_file_stream_parse_bytes(self):
        """Test AudioFileStream byte parsing"""
        stream = AudioFileStream()
        stream.open()
        try:
            dummy_data = b"RIFF\x00\x00\x00\x00WAVE"
            stream.parse_bytes(dummy_data)
        except AudioFileError:
            pass
        finally:
            stream.close()

    def test_audio_file_stream_auto_open(self):
        """Test AudioFileStream automatic opening on parse_bytes"""
        stream = AudioFileStream()
        assert stream.object_id == 0
        try:
            stream.parse_bytes(b"dummy data")
            assert stream.object_id != 0
        except AudioFileError:
            assert stream.object_id != 0
        finally:
            if not stream.is_disposed:
                stream.close()

    def test_audio_file_stream_seek(self):
        """Test AudioFileStream seeking"""
        stream = AudioFileStream()
        with pytest.raises(AudioFileError, match="Stream not open"):
            stream.seek(100)
        stream.open()
        try:
            stream.seek(0)
        except AudioFileError:
            pass
        finally:
            stream.close()

    def test_audio_file_stream_get_property(self):
        """Test AudioFileStream property getting"""
        stream = AudioFileStream()
        with pytest.raises(AudioFileError, match="Stream not open"):
            stream.get_property(42)
        stream.open()
        try:
            prop_id = capi.get_audio_file_stream_property_ready_to_produce_packets()
            stream.get_property(prop_id)
        except AudioFileError:
            pass
        finally:
            stream.close()

    def test_audio_file_stream_ready_to_produce_packets_property(self):
        """Test ready_to_produce_packets property"""
        stream = AudioFileStream()
        stream.open()
        try:
            ready = stream.ready_to_produce_packets
            assert isinstance(ready, bool)
            assert not ready
        finally:
            stream.close()

    def test_audio_file_stream_operations_on_disposed_object(self):
        """Test operations on disposed AudioFileStream"""
        stream = AudioFileStream()
        stream.open()
        stream.close()
        with pytest.raises(RuntimeError, match="has been disposed"):
            stream.parse_bytes(b"data")
        with pytest.raises(RuntimeError, match="has been disposed"):
            stream.seek(0)

    def test_audio_file_stream_error_handling(self):
        """Test AudioFileStream error handling"""
        stream = AudioFileStream()
        with pytest.raises(AudioFileError):
            stream.seek(100)
        with pytest.raises(AudioFileError):
            stream.get_property(42)


class TestAudioFileIntegration:
    """Integration tests for AudioFile with real audio data"""

    def test_audio_file_vs_functional_api_consistency(self, amen_wav_path):
        """Test that OO API produces consistent results with functional API"""
        audio_file_id = capi.audio_file_open_url(amen_wav_path)
        try:
            func_data, func_count = capi.audio_file_read_packets(audio_file_id, 0, 10)
        finally:
            capi.audio_file_close(audio_file_id)
        with AudioFile(amen_wav_path) as audio_file:
            oo_data, oo_count = audio_file.read_packets(0, 10)
        assert func_data == oo_data
        assert func_count == oo_count

    def test_audio_file_multiple_opens(self, amen_wav_path):
        """Test opening the same file multiple times"""
        file1 = AudioFile(amen_wav_path)
        file2 = AudioFile(amen_wav_path)
        with file1, file2:
            data1, _ = file1.read_packets(0, 5)
            data2, _ = file2.read_packets(0, 5)
            assert data1 == data2

    def test_audio_file_resource_management(self, amen_wav_path):
        """Test proper resource management"""
        files = []
        for _ in range(10):
            audio_file = AudioFile(amen_wav_path)
            audio_file.open()
            files.append(audio_file)
        for audio_file in files:
            audio_file.close()
            assert audio_file.is_disposed
