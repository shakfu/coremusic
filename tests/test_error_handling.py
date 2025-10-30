"""Tests for error handling decorators and OSStatus utilities."""

import pytest
import coremusic as cm
from coremusic.os_status import (
    check_os_status,
    check_return_status,
    raises_on_error,
    handle_exceptions,
    format_os_status_error,
    os_status_to_string,
    get_error_suggestion,
)


class TestOSStatusTranslation:
    """Test OSStatus error code translation."""

    def test_os_status_to_string_success(self):
        """Test translation of success code."""
        result = os_status_to_string(0)
        assert result == "No error"

    def test_os_status_to_string_file_not_found(self):
        """Test translation of file not found error."""
        result = os_status_to_string(-43)
        assert "kAudioFileFileNotFoundError" in result
        assert "File not found" in result

    def test_os_status_to_string_audiounit_error(self):
        """Test translation of AudioUnit error."""
        result = os_status_to_string(-10875)
        assert "kAudioUnitErr_InvalidProperty" in result

    def test_os_status_to_string_fourcc_error(self):
        """Test translation of FourCC error code."""
        result = os_status_to_string(0x7479703F)  # 'typ?'
        assert "Unsupported file type" in result

    def test_os_status_to_string_unknown(self):
        """Test translation of unknown error code."""
        result = os_status_to_string(-99999)
        assert "Unknown error code" in result

    def test_get_error_suggestion_file_not_found(self):
        """Test recovery suggestion for file not found."""
        suggestion = get_error_suggestion(-43)
        assert suggestion is not None
        assert "file path" in suggestion.lower()

    def test_get_error_suggestion_no_suggestion(self):
        """Test no suggestion for unknown error."""
        suggestion = get_error_suggestion(-99999)
        assert suggestion is None

    def test_format_os_status_error_with_operation(self):
        """Test formatted error with operation description."""
        result = format_os_status_error(-43, "open audio file")
        assert "Failed to open audio file" in result
        assert "kAudioFileFileNotFoundError" in result
        assert "Suggestion:" in result

    def test_format_os_status_error_without_operation(self):
        """Test formatted error without operation description."""
        result = format_os_status_error(-43)
        assert "kAudioFileFileNotFoundError" in result


class TestCheckOSStatusDecorator:
    """Test check_os_status decorator."""

    def test_check_os_status_success(self):
        """Test decorator with successful return (0)."""
        @check_os_status("test operation", ValueError)
        def test_func():
            return 0

        # Should not raise
        result = test_func()
        assert result == 0

    def test_check_os_status_error(self):
        """Test decorator with error return."""
        @check_os_status("test operation", ValueError)
        def test_func():
            return -43  # File not found

        with pytest.raises(ValueError) as exc_info:
            test_func()

        assert "Failed to test operation" in str(exc_info.value)
        assert "kAudioFileFileNotFoundError" in str(exc_info.value)

    def test_check_os_status_non_integer(self):
        """Test decorator with non-integer return."""
        @check_os_status("test operation", ValueError)
        def test_func():
            return "not an integer"

        # Should pass through non-integer returns
        result = test_func()
        assert result == "not an integer"

    def test_check_os_status_with_args(self):
        """Test decorator with function arguments."""
        @check_os_status("test operation", ValueError)
        def test_func(a, b):
            return a + b  # Returns integer

        # If sum is 0, should succeed
        result = test_func(0, 0)
        assert result == 0

        # If sum is non-zero error code
        with pytest.raises(ValueError):
            test_func(-43, 0)


class TestCheckReturnStatusDecorator:
    """Test check_return_status decorator."""

    def test_check_return_status_success_status_first(self):
        """Test decorator with (status, result) tuple - success."""
        @check_return_status("test operation", ValueError, status_index=0)
        def test_func():
            return (0, "success data")

        result = test_func()
        assert result == "success data"

    def test_check_return_status_error_status_first(self):
        """Test decorator with (status, result) tuple - error."""
        @check_return_status("test operation", ValueError, status_index=0)
        def test_func():
            return (-43, None)

        with pytest.raises(ValueError) as exc_info:
            test_func()

        assert "Failed to test operation" in str(exc_info.value)

    def test_check_return_status_success_status_last(self):
        """Test decorator with (result, status) tuple - success."""
        @check_return_status("test operation", ValueError, status_index=1)
        def test_func():
            return ("success data", 0)

        result = test_func()
        assert result == "success data"

    def test_check_return_status_error_status_last(self):
        """Test decorator with (result, status) tuple - error."""
        @check_return_status("test operation", ValueError, status_index=1)
        def test_func():
            return (None, -43)

        with pytest.raises(ValueError) as exc_info:
            test_func()

        assert "Failed to test operation" in str(exc_info.value)

    def test_check_return_status_multiple_values(self):
        """Test decorator with tuple of multiple values."""
        @check_return_status("test operation", ValueError, status_index=0)
        def test_func():
            return (0, "data1", "data2", 123)

        result = test_func()
        assert result == ("data1", "data2", 123)

    def test_check_return_status_non_tuple(self):
        """Test decorator raises TypeError for non-tuple."""
        @check_return_status("test operation", ValueError, status_index=0)
        def test_func():
            return 0  # Not a tuple

        with pytest.raises(TypeError) as exc_info:
            test_func()

        assert "must return a tuple" in str(exc_info.value)


class TestRaisesOnErrorDecorator:
    """Test raises_on_error decorator."""

    def test_raises_on_error_valid_return(self):
        """Test decorator with valid return value."""
        @raises_on_error("test operation", ValueError)
        def test_func():
            return "valid data"

        result = test_func()
        assert result == "valid data"

    def test_raises_on_error_none(self):
        """Test decorator raises on None."""
        @raises_on_error("test operation", ValueError)
        def test_func():
            return None

        with pytest.raises(ValueError) as exc_info:
            test_func()

        assert "Failed to test operation" in str(exc_info.value)

    def test_raises_on_error_zero(self):
        """Test decorator raises on zero."""
        @raises_on_error("test operation", ValueError)
        def test_func():
            return 0

        with pytest.raises(ValueError) as exc_info:
            test_func()

        assert "Failed to test operation" in str(exc_info.value)

    def test_raises_on_error_empty_string(self):
        """Test decorator raises on empty string."""
        @raises_on_error("test operation", ValueError)
        def test_func():
            return ""

        with pytest.raises(ValueError):
            test_func()

    def test_raises_on_error_empty_list(self):
        """Test decorator raises on empty list."""
        @raises_on_error("test operation", ValueError)
        def test_func():
            return []

        with pytest.raises(ValueError):
            test_func()

    def test_raises_on_error_valid_number(self):
        """Test decorator allows non-zero numbers."""
        @raises_on_error("test operation", ValueError)
        def test_func():
            return 42

        result = test_func()
        assert result == 42


class TestHandleExceptionsDecorator:
    """Test handle_exceptions decorator."""

    def test_handle_exceptions_no_error(self):
        """Test decorator with successful execution."""
        @handle_exceptions("test operation", reraise_as=ValueError)
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_handle_exceptions_with_reraise(self):
        """Test decorator converts exception type."""
        @handle_exceptions("test operation", reraise_as=ValueError)
        def test_func():
            raise RuntimeError("original error")

        with pytest.raises(ValueError) as exc_info:
            test_func()

        assert "Failed to test operation" in str(exc_info.value)
        assert "original error" in str(exc_info.value)

    def test_handle_exceptions_without_reraise(self):
        """Test decorator enhances original exception."""
        @handle_exceptions("test operation")
        def test_func():
            raise RuntimeError("original error")

        with pytest.raises(RuntimeError) as exc_info:
            test_func()

        assert "Failed to test operation" in str(exc_info.value)

    def test_handle_exceptions_preserves_traceback(self):
        """Test decorator preserves exception chain."""
        @handle_exceptions("test operation", reraise_as=ValueError)
        def test_func():
            raise RuntimeError("original error")

        with pytest.raises(ValueError) as exc_info:
            test_func()

        # Check exception chain
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RuntimeError)


class TestDecoratorIntegration:
    """Test decorators working together."""

    def test_combined_decorators(self):
        """Test multiple decorators on same function."""
        @handle_exceptions("outer operation", reraise_as=cm.AudioFileError)
        @check_return_status("inner operation", ValueError, status_index=1)
        def test_func(status_code):
            return ("data", status_code)

        # Success case
        result = test_func(0)
        assert result == "data"

        # Error case - should raise AudioFileError
        with pytest.raises(cm.AudioFileError):
            test_func(-43)

    def test_decorator_with_capi_style_function(self):
        """Test decorator with capi-style return."""
        @check_return_status("read data", cm.AudioFileError, status_index=0)
        def mock_capi_read():
            # Simulates capi.audio_file_read_packets return
            return (0, b'data', 1024)

        result = mock_capi_read()
        assert result == (b'data', 1024)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
