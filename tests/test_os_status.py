"""Tests for OSStatus error code translation utilities."""

import pytest
import coremusic as cm
from coremusic import os_status


class TestOSStatusTranslation:
    """Test OSStatus error code translation"""

    def test_os_status_to_string_success(self):
        """Test translation of success code"""
        result = os_status.os_status_to_string(0)
        assert result == "No error"

    def test_os_status_to_string_file_not_found(self):
        """Test translation of file not found error"""
        result = os_status.os_status_to_string(-43)
        assert "kAudioFileFileNotFoundError" in result
        assert "File not found" in result

    def test_os_status_to_string_param_error(self):
        """Test translation of parameter error"""
        result = os_status.os_status_to_string(-50)
        assert "paramErr" in result
        assert "Invalid parameter" in result

    def test_os_status_to_string_permissions_error(self):
        """Test translation of permissions error (FourCC)"""
        result = os_status.os_status_to_string(0x70726D3F)  # 'prm?'
        assert "kAudioFilePermissionsError" in result
        assert "Permissions error" in result

    def test_os_status_to_string_unsupported_file_type(self):
        """Test translation of unsupported file type error"""
        result = os_status.os_status_to_string(0x7479703F)  # 'typ?'
        # FourCC 'typ?' appears in multiple error dictionaries, accept any match
        assert "UnsupportedFileType" in result or "Unsupported file type" in result

    def test_os_status_to_string_audio_unit_error(self):
        """Test translation of AudioUnit error"""
        result = os_status.os_status_to_string(-10875)
        assert "kAudioUnitErr_InvalidProperty" in result
        assert "Invalid property" in result

    def test_os_status_to_string_audio_queue_error(self):
        """Test translation of AudioQueue error"""
        result = os_status.os_status_to_string(-66687)
        assert "kAudioQueueErr_InvalidDevice" in result
        assert "Invalid device" in result

    def test_os_status_to_string_unknown_error(self):
        """Test translation of unknown error code"""
        result = os_status.os_status_to_string(-99999)
        assert "Unknown error code" in result
        assert "-99999" in result

    def test_os_status_to_string_fourcc_unknown(self):
        """Test translation of unknown FourCC code"""
        result = os_status.os_status_to_string(0x61626364)  # 'abcd'
        assert "Unknown error" in result or "abcd" in result


class TestErrorSuggestions:
    """Test error recovery suggestions"""

    def test_get_error_suggestion_file_not_found(self):
        """Test suggestion for file not found error"""
        suggestion = os_status.get_error_suggestion(-43)
        assert suggestion is not None
        assert "file path" in suggestion.lower()
        assert "exist" in suggestion.lower()

    def test_get_error_suggestion_param_error(self):
        """Test suggestion for parameter error"""
        suggestion = os_status.get_error_suggestion(-50)
        assert suggestion is not None
        assert "parameter" in suggestion.lower() or "format" in suggestion.lower()

    def test_get_error_suggestion_permissions(self):
        """Test suggestion for permissions error"""
        suggestion = os_status.get_error_suggestion(0x70726D3F)  # 'prm?'
        assert suggestion is not None
        assert "permission" in suggestion.lower()

    def test_get_error_suggestion_no_suggestion(self):
        """Test that unknown errors return None"""
        suggestion = os_status.get_error_suggestion(-99999)
        assert suggestion is None


class TestFormatOSStatusError:
    """Test formatted error message generation"""

    def test_format_os_status_error_with_operation(self):
        """Test formatting error with operation description"""
        result = os_status.format_os_status_error(-43, "open audio file")
        assert "Failed to open audio file" in result
        assert "kAudioFileFileNotFoundError" in result
        assert "File not found" in result
        assert "Suggestion:" in result

    def test_format_os_status_error_without_operation(self):
        """Test formatting error without operation description"""
        result = os_status.format_os_status_error(-43)
        assert "kAudioFileFileNotFoundError" in result
        assert "File not found" in result
        assert "Suggestion:" in result

    def test_format_os_status_error_no_suggestion(self):
        """Test formatting error with no available suggestion"""
        result = os_status.format_os_status_error(-99999, "test operation")
        assert "Failed to test operation" in result
        assert "Suggestion:" not in result

    def test_format_os_status_error_permissions(self):
        """Test formatting permissions error"""
        result = os_status.format_os_status_error(0x70726D3F, "write to file")
        assert "Failed to write to file" in result
        assert "kAudioFilePermissionsError" in result
        assert "Permissions error" in result
        assert "Suggestion:" in result


class TestGetErrorInfo:
    """Test error info tuple retrieval"""

    def test_get_error_info_success(self):
        """Test error info for success code"""
        name, desc, suggestion = os_status.get_error_info(0)
        assert name == "kAudioHardwareNoError"
        assert desc == "No error"
        assert suggestion is None

    def test_get_error_info_file_not_found(self):
        """Test error info for file not found"""
        name, desc, suggestion = os_status.get_error_info(-43)
        assert name == "kAudioFileFileNotFoundError"
        assert desc == "File not found"
        assert suggestion is not None
        assert "file path" in suggestion.lower()

    def test_get_error_info_param_error(self):
        """Test error info for parameter error"""
        name, desc, suggestion = os_status.get_error_info(-50)
        assert name == "paramErr"
        assert desc == "Invalid parameter"
        assert suggestion is not None

    def test_get_error_info_unknown(self):
        """Test error info for unknown error"""
        name, desc, suggestion = os_status.get_error_info(-99999)
        assert name == "UnknownError"
        assert "Unknown error" in desc
        assert suggestion is None


class TestCoreAudioErrorFromOSStatus:
    """Test CoreAudioError.from_os_status() class method"""

    def test_from_os_status_with_operation(self):
        """Test creating exception from OSStatus with operation"""
        exc = cm.CoreAudioError.from_os_status(-43, "open audio file")
        assert isinstance(exc, cm.CoreAudioError)
        assert exc.status_code == -43
        assert "Failed to open audio file" in str(exc)
        assert "kAudioFileFileNotFoundError" in str(exc)
        assert "File not found" in str(exc)

    def test_from_os_status_without_operation(self):
        """Test creating exception from OSStatus without operation"""
        exc = cm.CoreAudioError.from_os_status(-50)
        assert isinstance(exc, cm.CoreAudioError)
        assert exc.status_code == -50
        assert "paramErr" in str(exc)
        assert "Invalid parameter" in str(exc)

    def test_from_os_status_with_suggestion(self):
        """Test that suggestion is included in message"""
        exc = cm.CoreAudioError.from_os_status(-43, "load file")
        message = str(exc)
        assert "Failed to load file" in message
        assert "kAudioFileFileNotFoundError" in message
        # Suggestion should be appended
        assert len(message) > len("Failed to load file: kAudioFileFileNotFoundError")

    def test_from_os_status_subclass(self):
        """Test that subclasses can use from_os_status"""
        exc = cm.AudioFileError.from_os_status(-43, "open file")
        assert isinstance(exc, cm.AudioFileError)
        assert isinstance(exc, cm.CoreAudioError)
        assert exc.status_code == -43

    def test_from_os_status_audio_queue_error(self):
        """Test AudioQueueError with OSStatus"""
        exc = cm.AudioQueueError.from_os_status(-50, "create audio queue")
        assert isinstance(exc, cm.AudioQueueError)
        assert exc.status_code == -50
        assert "create audio queue" in str(exc)


class TestErrorCodeCoverage:
    """Test coverage of various error code categories"""

    def test_audio_hardware_errors(self):
        """Test audio hardware error codes"""
        codes = [
            (0x73746F70, "NotRunning"),  # 'stop' - match generic part
            (0x77686F3F, "UnknownProperty"),  # 'who?' - match generic part
            (0x21646576, "BadDevice"),  # '!dev' - match generic part
        ]
        for code, expected_substr in codes:
            result = os_status.os_status_to_string(code)
            assert expected_substr in result

    def test_audio_file_errors(self):
        """Test audio file error codes"""
        codes = [
            (0x7479703F, "UnsupportedFileType"),  # 'typ?' - match generic part
            (0x666D743F, "UnsupportedDataFormat"),  # 'fmt?' - match generic part
            (-38, "NotOpenError"),
            (-39, "EndOfFileError"),
        ]
        for code, expected_substr in codes:
            result = os_status.os_status_to_string(code)
            assert expected_substr in result

    def test_audio_unit_errors(self):
        """Test audio unit error codes"""
        codes = [
            (-10875, "kAudioUnitErr_InvalidProperty"),
            (-10876, "kAudioUnitErr_InvalidParameter"),
            (-10879, "kAudioUnitErr_FailedInitialization"),
            (-10885, "kAudioUnitErr_Uninitialized"),
        ]
        for code, expected_name in codes:
            result = os_status.os_status_to_string(code)
            assert expected_name in result

    def test_audio_queue_errors(self):
        """Test audio queue error codes"""
        codes = [
            (-66680, "kAudioQueueErr_InvalidBuffer"),
            (-66685, "kAudioQueueErr_InvalidParameter"),
            (-66687, "kAudioQueueErr_InvalidDevice"),
        ]
        for code, expected_name in codes:
            result = os_status.os_status_to_string(code)
            assert expected_name in result

    def test_system_errors(self):
        """Test system error codes"""
        codes = [
            (-50, "paramErr"),
            (-108, "memFullErr"),
            (-128, "userCanceledErr"),
        ]
        for code, expected_name in codes:
            result = os_status.os_status_to_string(code)
            assert expected_name in result
