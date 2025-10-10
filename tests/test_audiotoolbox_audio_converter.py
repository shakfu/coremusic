#!/usr/bin/env python3
"""Tests for AudioConverter functional API."""

import os
import struct
import pytest

import coremusic as cm


class TestAudioConverterAPI:
    """Test AudioConverter functional API"""

    @pytest.fixture
    def amen_wav_path(self):
        """Fixture providing path to amen.wav test file"""
        path = os.path.join("tests", "amen.wav")
        if not os.path.exists(path):
            pytest.skip(f"Test audio file not found: {path}")
        return path

    @pytest.fixture
    def source_format(self):
        """Fixture providing source audio format (44.1kHz, stereo, 16-bit)"""
        return {
            'sample_rate': 44100.0,
            'format_id': cm.get_audio_format_linear_pcm(),
            'format_flags': 12,  # kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked
            'bytes_per_packet': 4,
            'frames_per_packet': 1,
            'bytes_per_frame': 4,
            'channels_per_frame': 2,
            'bits_per_channel': 16,
            'reserved': 0
        }

    @pytest.fixture
    def dest_format_mono(self):
        """Fixture providing destination format (44.1kHz, mono, 16-bit)"""
        return {
            'sample_rate': 44100.0,
            'format_id': cm.get_audio_format_linear_pcm(),
            'format_flags': 12,
            'bytes_per_packet': 2,
            'frames_per_packet': 1,
            'bytes_per_frame': 2,
            'channels_per_frame': 1,
            'bits_per_channel': 16,
            'reserved': 0
        }

    @pytest.fixture
    def dest_format_48k(self):
        """Fixture providing destination format (48kHz, stereo, 16-bit)"""
        return {
            'sample_rate': 48000.0,
            'format_id': cm.get_audio_format_linear_pcm(),
            'format_flags': 12,
            'bytes_per_packet': 4,
            'frames_per_packet': 1,
            'bytes_per_frame': 4,
            'channels_per_frame': 2,
            'bits_per_channel': 16,
            'reserved': 0
        }

    def test_audio_converter_creation(self, source_format, dest_format_mono):
        """Test AudioConverter creation and disposal"""
        converter_id = cm.audio_converter_new(source_format, dest_format_mono)
        assert converter_id is not None
        assert converter_id > 0

        cm.audio_converter_dispose(converter_id)

    def test_audio_converter_convert_buffer(self, source_format, dest_format_mono):
        """Test AudioConverter buffer conversion (stereo to mono)"""
        converter_id = cm.audio_converter_new(source_format, dest_format_mono)

        try:
            # Create test audio data: 100 frames of stereo 16-bit PCM
            # Each frame = 4 bytes (2 channels * 2 bytes)
            num_frames = 100
            input_data = b'\x00\x01' * (num_frames * 2)  # Simple test pattern

            # Convert stereo to mono
            output_data = cm.audio_converter_convert_buffer(converter_id, input_data)

            assert isinstance(output_data, bytes)
            # Output should be mono (half the size)
            # Input: 100 frames * 4 bytes = 400 bytes
            # Output: 100 frames * 2 bytes = 200 bytes (approximately)
            assert len(output_data) > 0
            assert len(output_data) < len(input_data)

        finally:
            cm.audio_converter_dispose(converter_id)

    def test_audio_converter_sample_rate_conversion(self, source_format, dest_format_48k):
        """Test AudioConverter creation for sample rate conversion"""
        # Note: Simple buffer conversion doesn't support sample rate changes well
        # Sample rate conversion typically requires callback-based approach
        # This test just verifies converter creation succeeds
        converter_id = cm.audio_converter_new(source_format, dest_format_48k)
        assert converter_id is not None
        assert converter_id > 0
        cm.audio_converter_dispose(converter_id)

    def test_audio_converter_reset(self, source_format, dest_format_mono):
        """Test AudioConverter reset functionality"""
        converter_id = cm.audio_converter_new(source_format, dest_format_mono)

        try:
            # Convert some data
            input_data = b'\x00\x01' * 200
            output1 = cm.audio_converter_convert_buffer(converter_id, input_data)

            # Reset converter
            cm.audio_converter_reset(converter_id)

            # Convert again - should work after reset
            output2 = cm.audio_converter_convert_buffer(converter_id, input_data)

            assert isinstance(output2, bytes)
            assert len(output2) > 0

        finally:
            cm.audio_converter_dispose(converter_id)

    def test_audio_converter_property_getters(self):
        """Test AudioConverter property ID getter functions"""
        # Test that all property getter functions return valid integers
        assert isinstance(cm.get_audio_converter_property_min_input_buffer_size(), int)
        assert isinstance(cm.get_audio_converter_property_min_output_buffer_size(), int)
        assert isinstance(cm.get_audio_converter_property_max_output_packet_size(), int)
        assert isinstance(cm.get_audio_converter_property_max_input_packet_size(), int)
        assert isinstance(cm.get_audio_converter_property_sample_rate_converter_quality(), int)
        assert isinstance(cm.get_audio_converter_property_codec_quality(), int)

    def test_audio_converter_with_real_file(self, amen_wav_path, source_format, dest_format_mono):
        """Test AudioConverter with real audio file data"""
        # Open the audio file
        audio_file_id = cm.audio_file_open_url(
            amen_wav_path,
            cm.get_audio_file_read_permission(),
            cm.get_audio_file_wave_type()
        )

        try:
            # Read some packets from the file
            packet_data, packets_read = cm.audio_file_read_packets(audio_file_id, 0, 100)
            assert len(packet_data) > 0

            # Create converter
            converter_id = cm.audio_converter_new(source_format, dest_format_mono)

            try:
                # Convert the real audio data from stereo to mono
                output_data = cm.audio_converter_convert_buffer(converter_id, packet_data)

                assert isinstance(output_data, bytes)
                assert len(output_data) > 0
                # Mono should be roughly half the size
                assert len(output_data) < len(packet_data)

            finally:
                cm.audio_converter_dispose(converter_id)

        finally:
            cm.audio_file_close(audio_file_id)

    def test_audio_converter_error_handling(self):
        """Test AudioConverter error handling"""
        # Test with invalid converter ID - should raise exception
        with pytest.raises((RuntimeError, cm.CoreAudioError)):
            cm.audio_converter_convert_buffer(999999, b'\x00\x01')

    def test_audio_converter_empty_buffer(self, source_format, dest_format_mono):
        """Test AudioConverter with empty buffer"""
        converter_id = cm.audio_converter_new(source_format, dest_format_mono)

        try:
            # Try to convert empty buffer - CoreAudio may handle this differently
            try:
                output_data = cm.audio_converter_convert_buffer(converter_id, b'')
                # Should handle gracefully (empty output or error)
                assert isinstance(output_data, bytes)
            except (RuntimeError, cm.CoreAudioError):
                # Empty buffer might cause error - that's acceptable
                pass

        finally:
            cm.audio_converter_dispose(converter_id)
