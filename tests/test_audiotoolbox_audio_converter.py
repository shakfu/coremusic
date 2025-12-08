"""Tests for AudioConverter functional API."""

import os
import struct
import pytest
import coremusic as cm
import coremusic.capi as capi


class TestAudioConverterAPI:
    """Test AudioConverter functional API"""

    def test_audio_converter_creation(self, source_format, dest_format_mono):
        """Test AudioConverter creation and disposal"""
        converter_id = capi.audio_converter_new(source_format, dest_format_mono)
        assert converter_id is not None
        assert converter_id > 0
        capi.audio_converter_dispose(converter_id)

    def test_audio_converter_convert_buffer(self, source_format, dest_format_mono):
        """Test AudioConverter buffer conversion (stereo to mono)"""
        converter_id = capi.audio_converter_new(source_format, dest_format_mono)
        try:
            num_frames = 100
            input_data = b"\x00\x01" * (num_frames * 2)
            output_data = capi.audio_converter_convert_buffer(converter_id, input_data)
            assert isinstance(output_data, bytes)
            assert len(output_data) > 0
            assert len(output_data) < len(input_data)
        finally:
            capi.audio_converter_dispose(converter_id)

    def test_audio_converter_sample_rate_conversion(
        self, source_format, dest_format_48k
    ):
        """Test AudioConverter creation for sample rate conversion"""
        converter_id = capi.audio_converter_new(source_format, dest_format_48k)
        assert converter_id is not None
        assert converter_id > 0
        capi.audio_converter_dispose(converter_id)

    def test_audio_converter_reset(self, source_format, dest_format_mono):
        """Test AudioConverter reset functionality"""
        converter_id = capi.audio_converter_new(source_format, dest_format_mono)
        try:
            input_data = b"\x00\x01" * 200
            output1 = capi.audio_converter_convert_buffer(converter_id, input_data)
            capi.audio_converter_reset(converter_id)
            output2 = capi.audio_converter_convert_buffer(converter_id, input_data)
            assert isinstance(output2, bytes)
            assert len(output2) > 0
        finally:
            capi.audio_converter_dispose(converter_id)

    def test_audio_converter_property_getters(self):
        """Test AudioConverter property ID getter functions"""
        assert isinstance(
            capi.get_audio_converter_property_min_input_buffer_size(), int
        )
        assert isinstance(
            capi.get_audio_converter_property_min_output_buffer_size(), int
        )
        assert isinstance(
            capi.get_audio_converter_property_max_output_packet_size(), int
        )
        assert isinstance(
            capi.get_audio_converter_property_max_input_packet_size(), int
        )
        assert isinstance(
            capi.get_audio_converter_property_sample_rate_converter_quality(), int
        )
        assert isinstance(capi.get_audio_converter_property_codec_quality(), int)

    def test_audio_converter_with_real_file(
        self, amen_wav_path, source_format, dest_format_mono
    ):
        """Test AudioConverter with real audio file data"""
        audio_file_id = capi.audio_file_open_url(
            amen_wav_path,
            capi.get_audio_file_read_permission(),
            capi.get_audio_file_wave_type(),
        )
        try:
            packet_data, packets_read = capi.audio_file_read_packets(
                audio_file_id, 0, 100
            )
            assert len(packet_data) > 0
            converter_id = capi.audio_converter_new(source_format, dest_format_mono)
            try:
                output_data = capi.audio_converter_convert_buffer(
                    converter_id, packet_data
                )
                assert isinstance(output_data, bytes)
                assert len(output_data) > 0
                assert len(output_data) < len(packet_data)
            finally:
                capi.audio_converter_dispose(converter_id)
        finally:
            capi.audio_file_close(audio_file_id)

    def test_audio_converter_error_handling(self):
        """Test AudioConverter error handling"""
        with pytest.raises((RuntimeError, cm.CoreAudioError)):
            capi.audio_converter_convert_buffer(999999, b"\x00\x01")

    def test_audio_converter_empty_buffer(self, source_format, dest_format_mono):
        """Test AudioConverter with empty buffer"""
        converter_id = capi.audio_converter_new(source_format, dest_format_mono)
        try:
            try:
                output_data = capi.audio_converter_convert_buffer(converter_id, b"")
                assert isinstance(output_data, bytes)
            except (RuntimeError, cm.CoreAudioError):
                pass
        finally:
            capi.audio_converter_dispose(converter_id)
