"""Exception classes for coremusic.

This module defines the exception hierarchy used throughout coremusic.
All exceptions inherit from CoreAudioError, which provides automatic
OSStatus error code translation.
"""

from __future__ import annotations

from .. import os_status

__all__ = [
    "CoreAudioError",
    "AudioFileError",
    "AudioQueueError",
    "AudioUnitError",
    "AudioConverterError",
    "MIDIError",
    "MusicPlayerError",
    "AudioDeviceError",
    "AUGraphError",
]


class CoreAudioError(Exception):
    """Base exception for CoreAudio errors"""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code

    @classmethod
    def from_os_status(cls, status: int, operation: str = ""):
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
