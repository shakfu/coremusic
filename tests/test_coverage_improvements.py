"""
Additional tests to improve code coverage to 90%+

This file contains targeted tests for previously untested code paths,
focusing on error handling, edge cases, and integration scenarios.
"""

import pytest

import coremusic as cm


class TestUtilitiesEdgeCases:
    """Test edge cases and error handling in utilities module"""

    def test_detect_silence_error_handling(self):
        """Test silence detection with invalid inputs"""
        with pytest.raises(Exception):
            cm.detect_silence("/nonexistent/file.wav")

    def test_get_peak_amplitude_error_handling(self):
        """Test peak amplitude with invalid inputs"""
        with pytest.raises(Exception):
            cm.get_peak_amplitude("/nonexistent/file.wav")

    def test_calculate_rms_error_handling(self):
        """Test RMS calculation with invalid inputs"""
        with pytest.raises(Exception):
            cm.calculate_rms("/nonexistent/file.wav")

    def test_get_file_info_error_handling(self):
        """Test file info with invalid file"""
        with pytest.raises(Exception):
            cm.get_file_info("/nonexistent/file.wav")


class TestErrorHandlingPaths:
    """Test error handling and invalid inputs"""

    def test_invalid_audio_file_id(self):
        """Test operations with invalid audio file ID"""
        invalid_id = 999999

        with pytest.raises(Exception):
            cm.audio_file_read_packets(invalid_id, 0, 1024)

    def test_invalid_audio_queue_id(self):
        """Test operations with invalid audio queue ID"""
        invalid_id = 999999

        with pytest.raises(Exception):
            cm.audio_queue_start(invalid_id)

    def test_invalid_audio_unit_id(self):
        """Test operations with invalid audio unit ID"""
        invalid_id = 999999

        with pytest.raises(Exception):
            cm.audio_unit_initialize(invalid_id)

    def test_invalid_audio_converter_id(self):
        """Test operations with invalid audio converter ID"""
        invalid_id = 999999

        with pytest.raises(Exception):
            cm.audio_converter_dispose(invalid_id)
