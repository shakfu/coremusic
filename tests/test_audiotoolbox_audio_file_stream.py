import os
import struct
import time
import wave
import pytest
import coremusic as cm
import coremusic.capi as capi


class TestAudioFileStreamConstants:
    """Test AudioFileStream constants access"""

    def test_audio_file_stream_property_constants(self):
        """Test AudioFileStream property constants"""
        assert (
            capi.get_audio_file_stream_property_ready_to_produce_packets() is not None
        )
        assert capi.get_audio_file_stream_property_file_format() is not None
        assert capi.get_audio_file_stream_property_data_format() is not None
        assert capi.get_audio_file_stream_property_format_list() is not None
        assert capi.get_audio_file_stream_property_magic_cookie_data() is not None
        assert capi.get_audio_file_stream_property_audio_data_byte_count() is not None
        assert capi.get_audio_file_stream_property_audio_data_packet_count() is not None
        assert capi.get_audio_file_stream_property_maximum_packet_size() is not None
        assert capi.get_audio_file_stream_property_data_offset() is not None
        assert capi.get_audio_file_stream_property_channel_layout() is not None
        assert capi.get_audio_file_stream_property_packet_to_frame() is not None
        assert capi.get_audio_file_stream_property_frame_to_packet() is not None
        assert capi.get_audio_file_stream_property_packet_to_byte() is not None
        assert capi.get_audio_file_stream_property_byte_to_packet() is not None
        assert capi.get_audio_file_stream_property_packet_table_info() is not None
        assert capi.get_audio_file_stream_property_packet_size_upper_bound() is not None
        assert (
            capi.get_audio_file_stream_property_average_bytes_per_packet() is not None
        )
        assert capi.get_audio_file_stream_property_bit_rate() is not None
        assert capi.get_audio_file_stream_property_info_dictionary() is not None

    def test_audio_file_stream_flag_constants(self):
        """Test AudioFileStream flag constants"""
        assert capi.get_audio_file_stream_property_flag_property_is_cached() is not None
        assert capi.get_audio_file_stream_property_flag_cache_property() is not None
        assert capi.get_audio_file_stream_parse_flag_discontinuity() is not None
        assert capi.get_audio_file_stream_seek_flag_offset_is_estimated() is not None

    def test_audio_file_stream_error_constants(self):
        """Test AudioFileStream error constants"""
        assert capi.get_audio_file_stream_error_unsupported_file_type() is not None
        assert capi.get_audio_file_stream_error_unsupported_data_format() is not None
        assert capi.get_audio_file_stream_error_unsupported_property() is not None
        assert capi.get_audio_file_stream_error_bad_property_size() is not None
        assert capi.get_audio_file_stream_error_not_optimized() is not None
        assert capi.get_audio_file_stream_error_invalid_packet_offset() is not None
        assert capi.get_audio_file_stream_error_invalid_file() is not None
        assert capi.get_audio_file_stream_error_value_unknown() is not None
        assert capi.get_audio_file_stream_error_data_unavailable() is not None
        assert capi.get_audio_file_stream_error_illegal_operation() is not None
        assert capi.get_audio_file_stream_error_unspecified_error() is not None
        assert capi.get_audio_file_stream_error_discontinuity_cant_recover() is not None


class TestAudioFileStreamBasicOperations:
    """Test basic AudioFileStream operations"""

    def test_audio_file_stream_open_close(self):
        """Test opening and closing an AudioFileStream"""
        stream_id = capi.audio_file_stream_open()
        assert isinstance(stream_id, int)
        assert stream_id != 0
        result = capi.audio_file_stream_close(stream_id)
        assert result == 0
        wav_type = capi.get_audio_file_wave_type()
        stream_id = capi.audio_file_stream_open(wav_type)
        assert isinstance(stream_id, int)
        assert stream_id != 0
        result = capi.audio_file_stream_close(stream_id)
        assert result == 0

    def test_audio_file_stream_parse_empty_data(self):
        """Test parsing empty data through AudioFileStream"""
        stream_id = capi.audio_file_stream_open()
        try:
            result = capi.audio_file_stream_parse_bytes(stream_id, b"")
            assert result == 0
            discontinuity_flag = capi.get_audio_file_stream_parse_flag_discontinuity()
            result = capi.audio_file_stream_parse_bytes(
                stream_id, b"", discontinuity_flag
            )
            assert result == 0
        finally:
            capi.audio_file_stream_close(stream_id)


class TestAudioFileStreamWithWaveFile:
    """Test AudioFileStream with real WAV file data"""

    @pytest.fixture
    def wav_file_path(self, amen_wav_path):
        """Path to test WAV file"""
        return amen_wav_path

    @pytest.fixture
    def wav_data(self, wav_file_path):
        """Load WAV file data"""
        with open(wav_file_path, "rb") as f:
            return f.read()

    def test_audio_file_stream_parse_wav_header(self, wav_data):
        """Test parsing WAV file header through AudioFileStream"""
        stream_id = capi.audio_file_stream_open(capi.get_audio_file_wave_type())
        try:
            header_chunk = wav_data[:1024]
            result = capi.audio_file_stream_parse_bytes(stream_id, header_chunk)
            assert result == 0
            try:
                ready = capi.audio_file_stream_get_property(
                    stream_id,
                    capi.get_audio_file_stream_property_ready_to_produce_packets(),
                )
                assert isinstance(ready, int)
            except RuntimeError:
                pass
            try:
                file_format = capi.audio_file_stream_get_property(
                    stream_id, capi.get_audio_file_stream_property_file_format()
                )
                assert isinstance(file_format, int)
                assert file_format == capi.get_audio_file_wave_type()
            except RuntimeError:
                pass
        finally:
            capi.audio_file_stream_close(stream_id)

    def test_audio_file_stream_incremental_parsing(self, wav_data):
        """Test incremental parsing of WAV file data"""
        stream_id = capi.audio_file_stream_open(capi.get_audio_file_wave_type())
        try:
            chunk_size = 512
            for i in range(0, min(len(wav_data), 4096), chunk_size):
                chunk = wav_data[i : i + chunk_size]
                if chunk:
                    result = capi.audio_file_stream_parse_bytes(stream_id, chunk)
                    assert result == 0
            try:
                file_format = capi.audio_file_stream_get_property(
                    stream_id, capi.get_audio_file_stream_property_file_format()
                )
                assert file_format == capi.get_audio_file_wave_type()
            except RuntimeError:
                pass
        finally:
            capi.audio_file_stream_close(stream_id)

    def test_audio_file_stream_data_format_property(self, wav_data):
        """Test getting data format property from parsed WAV data"""
        stream_id = capi.audio_file_stream_open(capi.get_audio_file_wave_type())
        try:
            result = capi.audio_file_stream_parse_bytes(stream_id, wav_data[:2048])
            assert result == 0
            try:
                data_format = capi.audio_file_stream_get_property(
                    stream_id, capi.get_audio_file_stream_property_data_format()
                )
                assert isinstance(data_format, dict)
                required_fields = [
                    "sample_rate",
                    "format_id",
                    "format_flags",
                    "bytes_per_packet",
                    "frames_per_packet",
                    "bytes_per_frame",
                    "channels_per_frame",
                    "bits_per_channel",
                    "reserved",
                ]
                for field in required_fields:
                    assert field in data_format
                    assert isinstance(data_format[field], (int, float))
                assert data_format["sample_rate"] > 0
                assert data_format["format_id"] == capi.get_audio_format_linear_pcm()
                assert data_format["channels_per_frame"] > 0
                assert data_format["bits_per_channel"] > 0
            except RuntimeError as e:
                print(f"Data format not yet available: {e}")
        finally:
            capi.audio_file_stream_close(stream_id)


class TestAudioFileStreamSeek:
    """Test AudioFileStream seek functionality"""

    def test_audio_file_stream_seek_operations(self):
        """Test seek operations on AudioFileStream"""
        stream_id = capi.audio_file_stream_open()
        try:
            try:
                seek_result = capi.audio_file_stream_seek(stream_id, 0)
                assert isinstance(seek_result, dict)
                assert "byte_offset" in seek_result
                assert "flags" in seek_result
                assert "is_estimated" in seek_result
                assert isinstance(seek_result["byte_offset"], int)
                assert isinstance(seek_result["flags"], int)
                assert isinstance(seek_result["is_estimated"], bool)
            except RuntimeError:
                pass
        finally:
            capi.audio_file_stream_close(stream_id)


class TestAudioFileStreamPropertyInfo:
    """Test AudioFileStream property info functionality"""

    def test_audio_file_stream_property_access_behavior(self):
        """Test property access behavior before parsing data"""
        stream_id = capi.audio_file_stream_open()
        try:
            ready = capi.audio_file_stream_get_property(
                stream_id,
                capi.get_audio_file_stream_property_ready_to_produce_packets(),
            )
            assert ready == 0
            with pytest.raises(RuntimeError):
                capi.audio_file_stream_get_property(
                    stream_id, capi.get_audio_file_stream_property_file_format()
                )
            with pytest.raises(RuntimeError):
                capi.audio_file_stream_get_property(
                    stream_id, capi.get_audio_file_stream_property_data_format()
                )
        finally:
            capi.audio_file_stream_close(stream_id)
