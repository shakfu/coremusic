"""OSStatus error code translation and formatting utilities.

This module provides human-readable translations of CoreAudio OSStatus error codes
and helpful recovery suggestions for common error scenarios.
"""

from functools import wraps
from typing import Any, Callable, Optional, Tuple, TypeVar

# CoreAudio error code mappings
# FourCC codes are converted to integers using: ord(a)<<24 | ord(b)<<16 | ord(c)<<8 | ord(d)

# General CoreAudio errors
AUDIO_HARDWARE_ERRORS = {
    0: ("kAudioHardwareNoError", "No error"),
    0x73746F70: ("kAudioHardwareNotRunningError", "Hardware not running"),  # 'stop'
    0x77686174: ("kAudioHardwareUnspecifiedError", "Unspecified error"),  # 'what'
    0x77686F3F: ("kAudioHardwareUnknownPropertyError", "Unknown property"),  # 'who?'
    0x2173697A: ("kAudioHardwareBadPropertySizeError", "Bad property size"),  # '!siz'
    0x6E6F7065: ("kAudioHardwareIllegalOperationError", "Illegal operation"),  # 'nope'
    0x216F626A: ("kAudioHardwareBadObjectError", "Bad object"),  # '!obj'
    0x21646576: ("kAudioHardwareBadDeviceError", "Bad device"),  # '!dev'
    0x21737472: ("kAudioHardwareBadStreamError", "Bad stream"),  # '!str'
    0x756E6F70: ("kAudioHardwareUnsupportedOperationError", "Unsupported operation"),  # 'unop'
    0x6E726479: ("kAudioHardwareNotReadyError", "Hardware not ready"),  # 'nrdy'
    0x21646174: ("kAudioDeviceUnsupportedFormatError", "Unsupported format"),  # '!dat'
    0x21686F67: ("kAudioDevicePermissionsError", "Permissions error"),  # '!hog'
}

# AudioFile errors
AUDIO_FILE_ERRORS = {
    0x7768743F: ("kAudioFileUnspecifiedError", "Unspecified error"),  # 'wht?'
    0x7479703F: ("kAudioFileUnsupportedFileTypeError", "Unsupported file type"),  # 'typ?'
    0x666D743F: ("kAudioFileUnsupportedDataFormatError", "Unsupported data format"),  # 'fmt?'
    0x7074793F: ("kAudioFileUnsupportedPropertyError", "Unsupported property"),  # 'pty?'
    0x2173697A: ("kAudioFileBadPropertySizeError", "Bad property size"),  # '!siz'
    0x70726D3F: ("kAudioFilePermissionsError", "Permissions error"),  # 'prm?'
    0x6F70746D: ("kAudioFileNotOptimizedError", "File not optimized"),  # 'optm'
    0x63686B3F: ("kAudioFileInvalidChunkError", "Invalid chunk"),  # 'chk?'
    0x6F66663F: ("kAudioFileDoesNotAllow64BitDataSizeError", "Does not allow 64-bit data size"),  # 'off?'
    0x70636B3F: ("kAudioFileInvalidPacketOffsetError", "Invalid packet offset"),  # 'pck?'
    0x6465703F: ("kAudioFileInvalidPacketDependencyError", "Invalid packet dependency"),  # 'dep?'
    0x6474613F: ("kAudioFileInvalidFileError", "Invalid file"),  # 'dta?'
    0x6F703F3F: ("kAudioFileOperationNotSupportedError", "Operation not supported"),  # 'op??'
    -38: ("kAudioFileNotOpenError", "File not open"),
    -39: ("kAudioFileEndOfFileError", "End of file"),
    -40: ("kAudioFilePositionError", "Invalid position"),
    -43: ("kAudioFileFileNotFoundError", "File not found"),
}

# AudioFormat errors
AUDIO_FORMAT_ERRORS = {
    0x77686174: ("kAudioFormatUnspecifiedError", "Unspecified error"),  # 'what'
    0x70726F70: ("kAudioFormatUnsupportedPropertyError", "Unsupported property"),  # 'prop'
    0x2173697A: ("kAudioFormatBadPropertySizeError", "Bad property size"),  # '!siz'
    0x21737063: ("kAudioFormatBadSpecifierSizeError", "Bad specifier size"),  # '!spc'
    0x666D743F: ("kAudioFormatUnsupportedDataFormatError", "Unsupported data format"),  # 'fmt?'
    0x21666D74: ("kAudioFormatUnknownFormatError", "Unknown format"),  # '!fmt'
}

# AudioFileStream errors
AUDIO_FILE_STREAM_ERRORS = {
    0x7479703F: ("kAudioFileStreamError_UnsupportedFileType", "Unsupported file type"),  # 'typ?'
    0x666D743F: ("kAudioFileStreamError_UnsupportedDataFormat", "Unsupported data format"),  # 'fmt?'
    0x7074793F: ("kAudioFileStreamError_UnsupportedProperty", "Unsupported property"),  # 'pty?'
    0x2173697A: ("kAudioFileStreamError_BadPropertySize", "Bad property size"),  # '!siz'
    0x6F70746D: ("kAudioFileStreamError_NotOptimized", "Not optimized"),  # 'optm'
    0x70636B3F: ("kAudioFileStreamError_InvalidPacketOffset", "Invalid packet offset"),  # 'pck?'
    0x6474613F: ("kAudioFileStreamError_InvalidFile", "Invalid file"),  # 'dta?'
    0x756E6B3F: ("kAudioFileStreamError_ValueUnknown", "Value unknown"),  # 'unk?'
    0x6D6F7265: ("kAudioFileStreamError_DataUnavailable", "Data unavailable"),  # 'more'
    0x6E6F7065: ("kAudioFileStreamError_IllegalOperation", "Illegal operation"),  # 'nope'
    0x7768743F: ("kAudioFileStreamError_UnspecifiedError", "Unspecified error"),  # 'wht?'
    0x64736321: ("kAudioFileStreamError_DiscontinuityCantRecover", "Discontinuity can't recover"),  # 'dsc!'
}

# AudioCodec errors
AUDIO_CODEC_ERRORS = {
    0: ("kAudioCodecNoError", "No error"),
    0x77686174: ("kAudioCodecUnspecifiedError", "Unspecified error"),  # 'what'
    0x77686F3F: ("kAudioCodecUnknownPropertyError", "Unknown property"),  # 'who?'
    0x2173697A: ("kAudioCodecBadPropertySizeError", "Bad property size"),  # '!siz'
    0x6E6F7065: ("kAudioCodecIllegalOperationError", "Illegal operation"),  # 'nope'
    0x21646174: ("kAudioCodecUnsupportedFormatError", "Unsupported format"),  # '!dat'
    0x21737474: ("kAudioCodecStateError", "State error"),  # '!stt'
    0x21627566: ("kAudioCodecNotEnoughBufferSpaceError", "Not enough buffer space"),  # '!buf'
    0x62616461: ("kAudioCodecBadDataError", "Bad data"),  # 'bada'
}

# AudioUnit errors (negative integers)
AUDIO_UNIT_ERRORS = {
    -10875: ("kAudioUnitErr_InvalidProperty", "Invalid property"),
    -10876: ("kAudioUnitErr_InvalidParameter", "Invalid parameter"),
    -10877: ("kAudioUnitErr_InvalidElement", "Invalid element"),
    -10878: ("kAudioUnitErr_NoConnection", "No connection"),
    -10879: ("kAudioUnitErr_FailedInitialization", "Failed initialization"),
    -10880: ("kAudioUnitErr_TooManyFramesToProcess", "Too many frames to process"),
    -10881: ("kAudioUnitErr_InvalidFile", "Invalid file"),
    -10882: ("kAudioUnitErr_UnknownFileType", "Unknown file type"),
    -10883: ("kAudioUnitErr_FileNotSpecified", "File not specified"),
    -10884: ("kAudioUnitErr_FormatNotSupported", "Format not supported"),
    -10885: ("kAudioUnitErr_Uninitialized", "Uninitialized"),
    -10886: ("kAudioUnitErr_InvalidScope", "Invalid scope"),
    -10887: ("kAudioUnitErr_PropertyNotWritable", "Property not writable"),
    -10888: ("kAudioUnitErr_CannotDoInCurrentContext", "Cannot do in current context"),
    -10889: ("kAudioUnitErr_InvalidPropertyValue", "Invalid property value"),
    -10890: ("kAudioUnitErr_PropertyNotInUse", "Property not in use"),
    -10891: ("kAudioUnitErr_Initialized", "Already initialized"),
    -10892: ("kAudioUnitErr_InvalidOfflineRender", "Invalid offline render"),
    -10893: ("kAudioUnitErr_Unauthorized", "Unauthorized"),
    -10863: ("kAudioUnitErr_InvalidFile", "Invalid file (broken plugin)"),
}

# Common system errors
SYSTEM_ERRORS = {
    -50: ("paramErr", "Invalid parameter"),
    -108: ("memFullErr", "Out of memory"),
    -128: ("userCanceledErr", "User canceled or security restriction"),
    -1: ("unimpErr", "Unimplemented"),
}

# AudioQueue errors
AUDIO_QUEUE_ERRORS = {
    -66680: ("kAudioQueueErr_InvalidBuffer", "Invalid buffer"),
    -66681: ("kAudioQueueErr_BufferEmpty", "Buffer empty"),
    -66682: ("kAudioQueueErr_DisposalPending", "Disposal pending"),
    -66683: ("kAudioQueueErr_InvalidProperty", "Invalid property"),
    -66684: ("kAudioQueueErr_InvalidPropertySize", "Invalid property size"),
    -66685: ("kAudioQueueErr_InvalidParameter", "Invalid parameter"),
    -66686: ("kAudioQueueErr_CannotStart", "Cannot start"),
    -66687: ("kAudioQueueErr_InvalidDevice", "Invalid device"),
    -66688: ("kAudioQueueErr_BufferInQueue", "Buffer in queue"),
    -66689: ("kAudioQueueErr_InvalidRunState", "Invalid run state"),
    -66690: ("kAudioQueueErr_InvalidQueueType", "Invalid queue type"),
    -66691: ("kAudioQueueErr_Permissions", "Permissions error"),
    -66692: ("kAudioQueueErr_InvalidPropertyValue", "Invalid property value"),
    -66693: ("kAudioQueueErr_PrimeTimedOut", "Prime timed out"),
    -66694: ("kAudioQueueErr_CodecNotFound", "Codec not found"),
    -66695: ("kAudioQueueErr_InvalidCodecAccess", "Invalid codec access"),
    -66696: ("kAudioQueueErr_QueueInvalidated", "Queue invalidated"),
    -66697: ("kAudioQueueErr_TooManyTaps", "Too many taps"),
    -66698: ("kAudioQueueErr_InvalidTapContext", "Invalid tap context"),
    -66699: ("kAudioQueueErr_RecordUnderrun", "Record underrun"),
    -66700: ("kAudioQueueErr_InvalidTapType", "Invalid tap type"),
    -66701: ("kAudioQueueErr_BufferEnqueuedTwice", "Buffer enqueued twice"),
    -66702: ("kAudioQueueErr_EnqueueDuringReset", "Enqueue during reset"),
    -66703: ("kAudioQueueErr_InvalidOfflineMode", "Invalid offline mode"),
}

# Combine all error dictionaries
ALL_ERRORS = {
    **AUDIO_HARDWARE_ERRORS,
    **AUDIO_FILE_ERRORS,
    **AUDIO_FORMAT_ERRORS,
    **AUDIO_FILE_STREAM_ERRORS,
    **AUDIO_CODEC_ERRORS,
    **AUDIO_UNIT_ERRORS,
    **SYSTEM_ERRORS,
    **AUDIO_QUEUE_ERRORS,
}


# Recovery suggestions based on error codes
# Note: Some error codes like -50 (paramErr) appear in multiple contexts
# The suggestion here is generic enough to apply to all cases
RECOVERY_SUGGESTIONS = {
    -50: "Check that all function parameters are valid and within acceptable ranges",
    -43: "Verify the file path exists and is accessible",
    -38: "Ensure the file is opened before performing this operation",
    -108: "Free up system memory or reduce buffer sizes",
    -128: "Check system security settings or user permissions",

    # Audio file errors
    0x70726D3F: "Check file permissions - the file may be read-only or locked by another process",
    0x7479703F: "Verify the audio file format is supported (WAV, AIFF, MP3, AAC, etc.)",
    0x666D743F: "Check that the audio format is supported by CoreAudio",
    0x6474613F: "The file may be corrupted or incomplete",
    -43: "Verify the file path exists and is spelled correctly",

    # Audio hardware errors
    0x21646576: "Check that the audio device is connected and recognized by the system",
    0x21686F67: "Another application may have exclusive access to the audio device",
    0x73746F70: "Start the audio hardware before performing this operation",
    0x6E726479: "Wait for audio hardware to become ready or restart the audio system",

    # Audio unit errors
    -10875: "The AudioUnit property you're trying to access doesn't exist",
    -10876: "Check AudioUnit parameter values are within valid ranges",
    -10877: "Verify the element (bus) number is valid for this AudioUnit",
    -10878: "Connect the AudioUnit nodes before attempting to process audio",
    -10879: "AudioUnit initialization failed - check format compatibility",
    -10881: "The AudioUnit preset or configuration file is invalid",
    -10863: "The AudioUnit plugin file is broken or incompatible",
    -10885: "Initialize the AudioUnit before attempting to use it",
    -10891: "The AudioUnit is already initialized - dispose and recreate if needed",

    # Audio queue errors
    -66687: "No audio output device available - check system audio settings",
    -66685: "Verify AudioQueue parameters (buffer size, format, etc.) are valid",
    -66691: "Ensure AudioQueue format matches hardware capabilities",
}


def os_status_to_string(status: int) -> str:
    """Convert OSStatus error code to human-readable string.

    Args:
        status: OSStatus error code (integer)

    Returns:
        Human-readable error name and description

    Example::

        >>> os_status_to_string(-43)
        'kAudioFileFileNotFoundError: File not found'
        >>> os_status_to_string(0x7479703F)
        'kAudioFileUnsupportedFileTypeError: Unsupported file type'
    """
    if status == 0:
        return "No error"

    if status in ALL_ERRORS:
        name, description = ALL_ERRORS[status]
        return f"{name}: {description}"

    # Try to interpret as FourCC if it looks like a character code
    if status > 0x20202020 and status < 0x7F7F7F7F:
        try:
            fourcc = bytes([
                (status >> 24) & 0xFF,
                (status >> 16) & 0xFF,
                (status >> 8) & 0xFF,
                status & 0xFF
            ]).decode('ascii', errors='ignore')
            return f"Unknown error '{fourcc}' (0x{status:08X})"
        except Exception:
            pass

    return f"Unknown error code {status}"


def get_error_suggestion(status: int) -> Optional[str]:
    """Get recovery suggestion for an OSStatus error code.

    Args:
        status: OSStatus error code

    Returns:
        Recovery suggestion string, or None if no suggestion available

    Example::

        >>> get_error_suggestion(-43)
        'Verify the file path exists and is spelled correctly'
    """
    return RECOVERY_SUGGESTIONS.get(status)


def format_os_status_error(status: int, operation: str = "") -> str:
    """Format a complete error message with status translation and recovery suggestion.

    Args:
        status: OSStatus error code
        operation: Description of the operation that failed (e.g., "open audio file")

    Returns:
        Formatted error message with status code, name, and suggestion

    Example::

        >>> format_os_status_error(-43, "open audio file")
        'Failed to open audio file: kAudioFileFileNotFoundError (File not found).
         Suggestion: Verify the file path exists and is spelled correctly'
    """
    error_str = os_status_to_string(status)
    suggestion = get_error_suggestion(status)

    if operation:
        message = f"Failed to {operation}: {error_str}"
    else:
        message = error_str

    if suggestion:
        message += f"\nSuggestion: {suggestion}"

    return message


def get_error_info(status: int) -> Tuple[str, str, Optional[str]]:
    """Get complete error information as a tuple.

    Args:
        status: OSStatus error code

    Returns:
        Tuple of (error_name, description, suggestion)

    Example::

        >>> name, desc, suggestion = get_error_info(-43)
        >>> print(name)
        'kAudioFileFileNotFoundError'
        >>> print(desc)
        'File not found'
    """
    if status == 0:
        return ("kAudioHardwareNoError", "No error", None)

    if status in ALL_ERRORS:
        name, description = ALL_ERRORS[status]
        suggestion = RECOVERY_SUGGESTIONS.get(status)
        return (name, description, suggestion)

    # Unknown error
    error_str = os_status_to_string(status)
    return ("UnknownError", error_str, None)


# Type variable for decorator return types
F = TypeVar('F', bound=Callable[..., Any])


def check_os_status(
    operation: str = "",
    exception_class: type = Exception
) -> Callable[[F], F]:
    """Decorator to check OSStatus return values and raise appropriate exceptions.

    This decorator automatically checks if a function returns an OSStatus error code
    (non-zero integer) and raises an exception with detailed error information.

    Args:
        operation: Description of the operation being performed (e.g., "open audio file")
        exception_class: Exception class to raise (default: Exception)

    Returns:
        Decorated function that raises exception on error

    Example::

        from coremusic.os_status import check_os_status
        from coremusic import AudioFileError

        @check_os_status("open audio file", AudioFileError)
        def open_file(path):
            # Returns OSStatus code
            return capi.audio_file_open_url(path)

        # Usage:
        file_id = open_file("test.wav")  # Raises AudioFileError if status != 0

    Notes:
        - The decorated function must return an integer OSStatus code
        - If the status is 0, the original return value is passed through
        - If the status is non-zero, an exception is raised with detailed error info
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # Check if result is an OSStatus error code
            if isinstance(result, int) and result != 0:
                error_msg = format_os_status_error(result, operation)
                raise exception_class(error_msg)

            return result

        return wrapper  # type: ignore

    return decorator


def check_return_status(
    operation: str = "",
    exception_class: type = Exception,
    status_index: int = 0
) -> Callable[[F], F]:
    """Decorator for functions that return (result, status) tuples.

    This decorator checks the OSStatus code from functions that return both
    a result value and a status code, raising an exception on error.

    Args:
        operation: Description of the operation being performed
        exception_class: Exception class to raise (default: Exception)
        status_index: Index of the status code in the return tuple (default: 0)
                      Use 0 for (status, result), 1 for (result, status)

    Returns:
        Decorated function that returns only the result on success, raises on error

    Example::

        from coremusic.os_status import check_return_status
        from coremusic import AudioFileError

        @check_return_status("read audio data", AudioFileError, status_index=1)
        def read_packets(file_id, num_packets):
            # Returns (data, status)
            data, status = capi.audio_file_read_packets(file_id, num_packets)
            return (data, status)

        # Usage:
        data = read_packets(file_id, 1024)  # Returns data, raises on error

    Notes:
        - The decorated function must return a tuple with status code
        - On success (status == 0), returns the non-status values
        - On error (status != 0), raises exception with detailed error info
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # Extract status from tuple
            if not isinstance(result, tuple):
                raise TypeError(
                    f"Function {func.__name__} must return a tuple, got {type(result)}"
                )

            status = result[status_index]

            # Check for error
            if isinstance(status, int) and status != 0:
                error_msg = format_os_status_error(status, operation)
                raise exception_class(error_msg)

            # Return non-status values
            if len(result) == 2:
                # Return the other value
                return result[1 - status_index]
            else:
                # Return tuple without status
                return tuple(v for i, v in enumerate(result) if i != status_index)

        return wrapper  # type: ignore

    return decorator


def raises_on_error(
    operation: str = "",
    exception_class: type = Exception
) -> Callable[[F], F]:
    """Decorator that converts None/empty returns into exceptions.

    This decorator is useful for functions that signal errors by returning None
    or empty values, converting them into proper exceptions with context.

    Args:
        operation: Description of the operation being performed
        exception_class: Exception class to raise (default: Exception)

    Returns:
        Decorated function that raises exception on None/empty return

    Example::

        from coremusic.os_status import raises_on_error
        from coremusic import AudioFileError

        @raises_on_error("find audio component", AudioFileError)
        def find_component(description):
            # Returns component ID or None
            return capi.audio_component_find_next(None, description)

        # Usage:
        component = find_component(desc)  # Raises AudioFileError if not found

    Notes:
        - Raises exception if return value is None, 0, empty string, or empty container
        - Otherwise passes through the return value unchanged
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # Check for error conditions
            if result is None or result == 0 or result == "" or result == []:
                error_msg = f"Failed to {operation}" if operation else f"{func.__name__} failed"
                raise exception_class(error_msg)

            return result

        return wrapper  # type: ignore

    return decorator


def handle_exceptions(
    operation: str = "",
    reraise_as: Optional[type] = None
) -> Callable[[F], F]:
    """Decorator that provides better error messages for caught exceptions.

    This decorator catches exceptions and re-raises them with additional context
    about what operation was being performed.

    Args:
        operation: Description of the operation being performed
        reraise_as: Optional exception class to convert to (default: keep original)

    Returns:
        Decorated function with enhanced error messages

    Example::

        from coremusic.os_status import handle_exceptions
        from coremusic import AudioFileError

        @handle_exceptions("process audio buffer", reraise_as=AudioFileError)
        def process_buffer(data):
            # May raise various exceptions
            return some_risky_operation(data)

        # Usage:
        result = process_buffer(data)  # Any exception becomes AudioFileError

    Notes:
        - Preserves the original exception message and adds operation context
        - Can optionally convert exception types
        - Useful for converting low-level exceptions to domain-specific ones
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if operation:
                    error_msg = f"Failed to {operation}: {e}"
                else:
                    error_msg = f"{func.__name__} failed: {e}"

                if reraise_as:
                    raise reraise_as(error_msg) from e
                else:
                    # Re-raise with enhanced message
                    e.args = (error_msg,) + e.args[1:]
                    raise

        return wrapper  # type: ignore

    return decorator
