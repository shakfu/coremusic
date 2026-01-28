#!/usr/bin/env python3
"""Tutorial: Quick Start

This module provides quick start examples for coremusic.
All examples are executable doctests.

Run with: pytest tests/tutorials/test_quickstart.py --doctest-modules -v
"""
from __future__ import annotations

import os
from pathlib import Path

# Test data path
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "wav"
TEST_AUDIO_FILE = TEST_DATA_DIR / "amen.wav"


def get_test_audio_path() -> str:
    """Get path to test audio file.

    >>> path = get_test_audio_path()
    >>> Path(path).exists()
    True
    """
    return str(TEST_AUDIO_FILE)


# =============================================================================
# Import Patterns
# =============================================================================


def import_main_package():
    """Import the main coremusic package.

    >>> import coremusic as cm
    >>> assert cm is not None
    """
    pass


def import_functional_api():
    """Import the low-level functional API.

    >>> import coremusic.capi as capi
    >>> assert capi is not None
    >>> # Check common functions are available
    >>> assert hasattr(capi, 'audio_file_open_url')
    >>> assert hasattr(capi, 'midi_get_number_of_devices')
    """
    pass


def import_constants():
    """Import constants for property IDs and format IDs.

    >>> from coremusic import AudioFileProperty, AudioFormatID
    >>> # These are enum-like classes with integer values
    >>> assert AudioFileProperty.DATA_FORMAT is not None
    >>> assert AudioFormatID.LINEAR_PCM is not None
    """
    pass


def import_music_theory():
    """Import music theory module.

    >>> from coremusic.music.theory import Note, Scale, Chord, Interval
    >>> assert Note is not None
    >>> assert Scale is not None
    >>> assert Chord is not None
    >>> assert Interval is not None
    """
    pass


# =============================================================================
# Audio File Operations
# =============================================================================


def open_audio_file():
    """Open and read an audio file.

    >>> import coremusic as cm
    >>> path = get_test_audio_path()
    >>> with cm.AudioFile(path) as audio:
    ...     assert audio.duration > 0
    ...     fmt = audio.format
    ...     assert fmt.sample_rate > 0
    ...     assert fmt.channels_per_frame >= 1
    """
    pass


def get_audio_info():
    """Get comprehensive audio file information.

    >>> import coremusic as cm
    >>> path = get_test_audio_path()
    >>> with cm.AudioFile(path) as audio:
    ...     info = {
    ...         'duration': audio.duration,
    ...         'sample_rate': audio.format.sample_rate,
    ...         'channels': audio.format.channels_per_frame,
    ...         'bits': audio.format.bits_per_channel,
    ...     }
    ...     assert info['duration'] > 0
    ...     assert info['sample_rate'] > 0
    ...     assert info['channels'] >= 1
    ...     assert info['bits'] > 0
    """
    pass


def read_audio_data():
    """Read audio data from a file.

    >>> import coremusic as cm
    >>> path = get_test_audio_path()
    >>> with cm.AudioFile(path) as audio:
    ...     # Read first 1000 frames
    ...     data, count = audio.read_packets(0, 1000)
    ...     assert isinstance(data, bytes)
    ...     assert len(data) > 0
    ...     assert count > 0
    """
    pass


# =============================================================================
# MIDI Operations
# =============================================================================


def check_midi_devices():
    """Check for available MIDI devices.

    >>> import coremusic.capi as capi
    >>> # Get counts (may be 0 if no devices connected)
    >>> num_devices = capi.midi_get_number_of_devices()
    >>> num_sources = capi.midi_get_number_of_sources()
    >>> num_destinations = capi.midi_get_number_of_destinations()
    >>> # All should be non-negative integers
    >>> assert num_devices >= 0
    >>> assert num_sources >= 0
    >>> assert num_destinations >= 0
    """
    pass


def create_midi_client():
    """Create a MIDI client.

    >>> import coremusic as cm
    >>> client = cm.MIDIClient("Quick Start")
    >>> assert client is not None
    >>> client.dispose()
    """
    pass


def build_midi_note():
    """Build a MIDI note message.

    >>> # Note On: channel 0, note 60 (C4), velocity 100
    >>> note_on = bytes([0x90, 60, 100])
    >>> assert len(note_on) == 3
    >>> assert note_on[0] == 0x90  # Note On, channel 0
    >>> assert note_on[1] == 60    # Middle C
    >>> assert note_on[2] == 100   # Velocity
    """
    pass


# =============================================================================
# AudioUnit Operations
# =============================================================================


def list_audio_units():
    """List available AudioUnits.

    >>> import coremusic as cm
    >>> units = cm.list_available_audio_units()
    >>> assert isinstance(units, list)
    >>> assert len(units) > 0  # macOS has built-in AudioUnits
    """
    pass


def find_effect_by_name():
    """Find an AudioUnit effect by name.

    >>> import coremusic as cm
    >>> # AUDelay is built into macOS
    >>> component = cm.find_audio_unit_by_name("AUDelay")
    >>> assert component is not None
    """
    pass


def create_effects_chain():
    """Create an audio effects chain.

    >>> import coremusic as cm
    >>> chain = cm.AudioEffectsChain()
    >>> delay = chain.add_effect_by_name("AUDelay")
    >>> output = chain.add_output()
    >>> chain.connect(delay, output)
    >>> chain.dispose()
    """
    pass


# =============================================================================
# Music Theory
# =============================================================================


def create_note():
    """Create a musical note.

    >>> from coremusic.music.theory import Note
    >>> c4 = Note.from_midi(60)
    >>> assert c4.midi == 60
    """
    pass


def create_scale():
    """Create a musical scale.

    >>> from coremusic.music.theory import Note, Scale, ScaleType
    >>> c4 = Note.from_midi(60)
    >>> c_major = Scale(c4, ScaleType.MAJOR)
    >>> notes = list(c_major)
    >>> assert len(notes) == 7
    """
    pass


def create_chord():
    """Create a musical chord.

    >>> from coremusic.music.theory import Note, Chord, ChordType
    >>> c4 = Note.from_midi(60)
    >>> cmaj = Chord(c4, ChordType.MAJOR)
    >>> notes = list(cmaj)
    >>> assert len(notes) >= 3
    """
    pass


# =============================================================================
# Error Handling
# =============================================================================


def handle_audio_file_error():
    """Handle errors when opening audio files.

    >>> import coremusic as cm
    >>> try:
    ...     with cm.AudioFile("nonexistent.wav") as audio:
    ...         pass
    ... except (cm.AudioFileError, FileNotFoundError, OSError):
    ...     pass  # Expected
    """
    pass


def exception_hierarchy():
    """CoreMusic exception hierarchy.

    >>> import coremusic as cm
    >>> # Base exception
    >>> assert hasattr(cm, 'CoreAudioError')
    >>> # Specific exceptions
    >>> assert hasattr(cm, 'AudioFileError')
    >>> assert hasattr(cm, 'AudioQueueError')
    >>> assert hasattr(cm, 'AudioUnitError')
    >>> assert hasattr(cm, 'MIDIError')
    """
    pass


# =============================================================================
# NumPy Integration
# =============================================================================


def check_numpy_available():
    """Check if NumPy integration is available.

    >>> import coremusic as cm
    >>> # NUMPY_AVAILABLE is True if numpy is installed
    >>> assert hasattr(cm, 'NUMPY_AVAILABLE')
    >>> assert isinstance(cm.NUMPY_AVAILABLE, bool)
    """
    pass


def numpy_dtype_from_format():
    """Get NumPy dtype from audio format.

    >>> import coremusic as cm
    >>> path = get_test_audio_path()
    >>> if cm.NUMPY_AVAILABLE:
    ...     with cm.AudioFile(path) as audio:
    ...         dtype = audio.format.to_numpy_dtype()
    ...         assert dtype is not None
    """
    pass


# =============================================================================
# Constants
# =============================================================================


def audio_file_properties():
    """Audio file property constants.

    >>> from coremusic import AudioFileProperty
    >>> # Common properties
    >>> _ = AudioFileProperty.DATA_FORMAT
    >>> _ = AudioFileProperty.ESTIMATED_DURATION
    >>> _ = AudioFileProperty.FILE_FORMAT
    >>> _ = AudioFileProperty.AUDIO_DATA_BYTE_COUNT
    """
    pass


def audio_format_ids():
    """Audio format ID constants.

    >>> from coremusic import AudioFormatID
    >>> # Common formats
    >>> _ = AudioFormatID.LINEAR_PCM
    >>> _ = AudioFormatID.APPLE_LOSSLESS
    >>> _ = AudioFormatID.MPEG4_AAC
    >>> _ = AudioFormatID.MPEG_LAYER_3
    """
    pass


def audiounit_properties():
    """AudioUnit property constants.

    >>> from coremusic import AudioUnitProperty, AudioUnitScope
    >>> # Common properties
    >>> _ = AudioUnitProperty.STREAM_FORMAT
    >>> _ = AudioUnitProperty.SAMPLE_RATE
    >>> _ = AudioUnitProperty.ELEMENT_COUNT
    >>> # Scopes
    >>> _ = AudioUnitScope.INPUT
    >>> _ = AudioUnitScope.OUTPUT
    >>> _ = AudioUnitScope.GLOBAL
    """
    pass


# Test runner
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
