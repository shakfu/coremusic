"""Exception classes for coremusic.

This module defines the exception hierarchy used throughout coremusic.
All exceptions inherit from CoreAudioError, which provides automatic
OSStatus error code translation.
"""

from __future__ import annotations

from coremusic import os_status

__all__ = [
    "AUGraphError",
    "FRAMEWORK_ERRORS",
    "AudioConverterError",
    "AudioDeviceError",
    "AudioFileError",
    "AudioQueueError",
    "AudioUnitError",
    "CoreAudioError",
    "MIDIError",
    "MusicPlayerError",
]


class CoreAudioError(Exception):
    """Base exception for CoreAudio errors"""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code

    @classmethod
    def from_os_status(cls, status: int, operation: str = "") -> CoreAudioError:
        """Create exception from OSStatus code with human-readable error message.

        Args:
            status: OSStatus error code
            operation: Description of failed operation (e.g., "open audio file")

        Returns:
            CoreAudioError with formatted message including error name and suggestion
        """
        error_str = os_status.os_status_to_string(status)
        suggestion = os_status.get_error_suggestion(status)

        if operation:
            message = f"Failed to {operation}: {error_str}"
        else:
            message = error_str

        if suggestion:
            message += f". {suggestion}"

        return cls(message, status_code=status)


class AudioFileError(CoreAudioError):
    """Exception for AudioFile operations"""


class AudioQueueError(CoreAudioError):
    """Exception for AudioQueue operations"""


class AudioUnitError(CoreAudioError):
    """Exception for AudioUnit operations"""


class AudioConverterError(CoreAudioError):
    """Exception for AudioConverter operations"""


class MIDIError(CoreAudioError):
    """Exception for MIDI operations"""


class MusicPlayerError(CoreAudioError):
    """Exception for MusicPlayer operations"""


class AudioDeviceError(CoreAudioError):
    """Exception for AudioDevice operations"""


class AUGraphError(CoreAudioError):
    """Exception for AUGraph operations"""


#: Errors that mean the OS frameworks refused an operation, as opposed to a
#: defect in this library or its caller.
#:
#: Catch this when falling back to a default value or skipping an item, so that
#: a genuine CoreAudio failure is absorbed but a programming error still
#: propagates. ``except Exception`` at such sites turns ``AttributeError``,
#: ``TypeError`` and ``NameError`` into a plausible-looking return value the
#: caller cannot distinguish from a real answer -- which is how
#: ``AudioFileStream.ready_to_produce_packets`` reported "not ready" for every
#: stream in every state while calling a function that did not exist.
#:
#: The members are what the layers below actually raise:
#:
#: - ``CoreAudioError`` and its subclasses, from the object wrappers.
#: - ``RuntimeError``, from ``capi`` for a non-zero ``OSStatus`` (223 sites).
#: - ``OSError``, from file and path operations.
#:
#: Deliberately excluded: ``capi`` also raises ``ValueError`` for an invalid
#: argument, ``MemoryError`` on allocation failure, and ``TypeError`` /
#: ``IndexError``. Those signal a bad call or an exhausted machine, not a
#: refused operation, and must not be swallowed.
#:
#: Broad ``except Exception`` remains correct in three places, each commented
#: where it appears: invoking a caller-supplied callback (which may raise
#: anything), the top of a worker-thread loop (where dying is worse than
#: continuing), and ``cli doctor`` (whose contract is to report any failure).
FRAMEWORK_ERRORS: tuple[type[BaseException], ...] = (
    CoreAudioError,
    RuntimeError,
    OSError,
)
