"""pytest test suite for coremusic wrapper functionality."""

import os
import pytest
import wave
import struct
import time
import coremusic as cm
import coremusic.capi as capi


class TestCoreAudioConstants:
    """Test CoreAudio constants access"""

    def test_audio_format_constants(self):
        """Test audio format constants"""
        assert capi.get_audio_format_linear_pcm() is not None
        assert capi.get_linear_pcm_format_flag_is_signed_integer() is not None
        assert capi.get_linear_pcm_format_flag_is_packed() is not None
        assert capi.get_linear_pcm_format_flag_is_non_interleaved() is not None

    def test_audio_file_constants(self):
        """Test audio file constants"""
        assert capi.get_audio_file_wave_type() is not None
        assert capi.get_audio_file_read_permission() is not None
        assert capi.get_audio_file_property_data_format() is not None
        assert capi.get_audio_file_property_maximum_packet_size() is not None

    def test_audio_unit_constants(self):
        """Test AudioUnit constants"""
        assert capi.get_audio_unit_type_output() is not None
        assert capi.get_audio_unit_subtype_default_output() is not None
        assert capi.get_audio_unit_manufacturer_apple() is not None
        assert capi.get_audio_unit_property_stream_format() is not None
        assert capi.get_audio_unit_scope_input() is not None
        assert capi.get_audio_unit_scope_output() is not None


class TestFourCCConversion:
    """Test FourCC conversion utilities"""

    def test_fourchar_to_int_conversion(self):
        """Test string to int FourCC conversion"""
        test_codes = ["WAVE", "TEXT", "AIFF", "mp4f", "RIFF"]
        for code in test_codes:
            int_val = capi.fourchar_to_int(code)
            assert isinstance(int_val, int)
            assert int_val > 0

    def test_int_to_fourchar_conversion(self):
        """Test int to string FourCC conversion"""
        test_codes = ["WAVE", "TEXT", "AIFF", "mp4f", "RIFF"]
        for code in test_codes:
            int_val = capi.fourchar_to_int(code)
            back_to_str = capi.int_to_fourchar(int_val)
            assert back_to_str == code

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion maintains data integrity"""
        test_codes = ["WAVE", "TEXT", "AIFF", "mp4f", "RIFF", "CAFF", "caff"]
        for code in test_codes:
            int_val = capi.fourchar_to_int(code)
            back_to_str = capi.int_to_fourchar(int_val)
            assert back_to_str == code, (
                f"Roundtrip failed for '{code}': {code} -> {int_val} -> {back_to_str}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
