#!/usr/bin/env python3
"""Tests for the base infrastructure of the object-oriented coremusic API."""

import pytest
import weakref
import gc

import coremusic as cm


class TestCoreAudioObject:
    """Test CoreAudioObject base class functionality"""

    def test_core_audio_object_creation(self):
        """Test basic CoreAudioObject creation and properties"""
        obj = cm.CoreAudioObject()

        # Test initial state
        assert not obj.is_disposed
        # Note: object_id is a cdef attribute and not accessible from Python

    def test_core_audio_object_disposal(self):
        """Test explicit disposal"""
        obj = cm.CoreAudioObject()

        # Test explicit disposal
        obj.dispose()
        assert obj.is_disposed

        # Test double disposal doesn't cause issues
        obj.dispose()
        assert obj.is_disposed

    def test_core_audio_object_automatic_disposal(self):
        """Test automatic disposal via __dealloc__"""
        obj = cm.CoreAudioObject()

        # Test that object can be deleted without errors
        # Note: Weak references don't work with Cython extension types by default
        del obj
        gc.collect()

        # If we get here, disposal worked without crashing

    def test_ensure_not_disposed_check(self):
        """Test _ensure_not_disposed method"""
        obj = cm.CoreAudioObject()

        # Should not raise when not disposed
        obj._ensure_not_disposed()

        # Should raise after disposal
        obj.dispose()
        with pytest.raises(RuntimeError, match="has been disposed"):
            obj._ensure_not_disposed()


class TestAudioFormat:
    """Test AudioFormat class functionality"""

    def test_audio_format_creation(self):
        """Test AudioFormat creation with various parameters"""
        # Test with minimal parameters
        format1 = cm.AudioFormat(44100.0, 'lpcm')
        assert format1.sample_rate == 44100.0
        assert format1.format_id == 'lpcm'
        assert format1.channels_per_frame == 2  # default
        assert format1.bits_per_channel == 16   # default

        # Test with full parameters
        format2 = cm.AudioFormat(
            sample_rate=48000.0,
            format_id='aac ',
            format_flags=12,
            bytes_per_packet=1024,
            frames_per_packet=512,
            bytes_per_frame=4,
            channels_per_frame=6,
            bits_per_channel=24
        )
        assert format2.sample_rate == 48000.0
        assert format2.format_id == 'aac '
        assert format2.format_flags == 12
        assert format2.bytes_per_packet == 1024
        assert format2.frames_per_packet == 512
        assert format2.bytes_per_frame == 4
        assert format2.channels_per_frame == 6
        assert format2.bits_per_channel == 24

    def test_audio_format_properties(self):
        """Test AudioFormat computed properties"""
        # Test PCM format
        pcm_format = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2)
        assert pcm_format.is_pcm
        assert pcm_format.is_stereo
        assert not pcm_format.is_mono

        # Test non-PCM format
        aac_format = cm.AudioFormat(44100.0, 'aac ', channels_per_frame=1)
        assert not aac_format.is_pcm
        assert not aac_format.is_stereo
        assert aac_format.is_mono

        # Test multi-channel format
        surround_format = cm.AudioFormat(48000.0, 'lpcm', channels_per_frame=6)
        assert surround_format.is_pcm
        assert not surround_format.is_stereo
        assert not surround_format.is_mono

    def test_audio_format_repr(self):
        """Test AudioFormat string representation"""
        format = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)
        repr_str = repr(format)

        assert 'AudioFormat' in repr_str
        assert '44100.0' in repr_str
        assert 'lpcm' in repr_str
        assert 'channels=2' in repr_str
        assert 'bits=16' in repr_str


class TestExceptionHierarchy:
    """Test exception hierarchy and functionality"""

    def test_core_audio_error(self):
        """Test CoreAudioError base exception"""
        # Test basic creation
        error = cm.CoreAudioError("Test message")
        assert str(error) == "Test message"
        assert error.status_code == 0

        # Test with status code
        error_with_code = cm.CoreAudioError("Test with code", 42)
        assert str(error_with_code) == "Test with code"
        assert error_with_code.status_code == 42

        # Test inheritance
        assert isinstance(error, Exception)

    def test_audio_file_error(self):
        """Test AudioFileError"""
        error = cm.AudioFileError("File error", 123)
        assert str(error) == "File error"
        assert error.status_code == 123
        assert isinstance(error, cm.CoreAudioError)
        assert isinstance(error, Exception)

    def test_audio_queue_error(self):
        """Test AudioQueueError"""
        error = cm.AudioQueueError("Queue error", -50)
        assert str(error) == "Queue error"
        assert error.status_code == -50
        assert isinstance(error, cm.CoreAudioError)

    def test_audio_unit_error(self):
        """Test AudioUnitError"""
        error = cm.AudioUnitError("Unit error")
        assert str(error) == "Unit error"
        assert error.status_code == 0
        assert isinstance(error, cm.CoreAudioError)

    def test_midi_error(self):
        """Test MIDIError"""
        error = cm.MIDIError("MIDI error", -10830)
        assert str(error) == "MIDI error"
        assert error.status_code == -10830
        assert isinstance(error, cm.CoreAudioError)

    def test_music_player_error(self):
        """Test MusicPlayerError"""
        error = cm.MusicPlayerError("Player error")
        assert str(error) == "Player error"
        assert isinstance(error, cm.CoreAudioError)

    def test_exception_raising_and_catching(self):
        """Test raising and catching various exceptions"""
        # Test raising and catching specific exception
        with pytest.raises(cm.AudioFileError):
            raise cm.AudioFileError("Test file error")

        # Test catching as base class
        with pytest.raises(cm.CoreAudioError):
            raise cm.AudioUnitError("Test unit error")

        # Test catching as Exception
        with pytest.raises(Exception):
            raise cm.MIDIError("Test MIDI error")


class TestObjectOrientedAPIAvailability:
    """Test that all OO API classes are available through main import"""

    def test_base_classes_available(self):
        """Test base infrastructure classes are available"""
        assert hasattr(cm, 'CoreAudioObject')
        assert hasattr(cm, 'AudioFormat')

    def test_exception_classes_available(self):
        """Test exception classes are available"""
        assert hasattr(cm, 'CoreAudioError')
        assert hasattr(cm, 'AudioFileError')
        assert hasattr(cm, 'AudioQueueError')
        assert hasattr(cm, 'AudioUnitError')
        assert hasattr(cm, 'MIDIError')
        assert hasattr(cm, 'MusicPlayerError')

    def test_audio_file_classes_available(self):
        """Test audio file classes are available"""
        assert hasattr(cm, 'AudioFile')
        assert hasattr(cm, 'AudioFileStream')

    def test_audio_queue_classes_available(self):
        """Test audio queue classes are available"""
        assert hasattr(cm, 'AudioBuffer')
        assert hasattr(cm, 'AudioQueue')

    def test_audio_unit_classes_available(self):
        """Test audio unit classes are available"""
        assert hasattr(cm, 'AudioComponentDescription')
        assert hasattr(cm, 'AudioComponent')
        assert hasattr(cm, 'AudioUnit')

    def test_midi_classes_available(self):
        """Test MIDI classes are available"""
        assert hasattr(cm, 'MIDIClient')
        assert hasattr(cm, 'MIDIPort')
        assert hasattr(cm, 'MIDIInputPort')
        assert hasattr(cm, 'MIDIOutputPort')

    def test_functional_api_still_available(self):
        """Test that functional API is still available alongside OO API"""
        # Test some key functional API functions
        assert hasattr(cm, 'fourchar_to_int')
        assert hasattr(cm, 'int_to_fourchar')
        assert hasattr(cm, 'audio_file_open_url')
        assert hasattr(cm, 'midi_client_create')

        # Test they work
        assert cm.fourchar_to_int('TEST') == 1413829460
        assert cm.int_to_fourchar(1413829460) == 'TEST'


class TestDualAPIInteraction:
    """Test interaction between functional and object-oriented APIs"""

    def test_both_apis_accessible(self):
        """Test that both APIs can be used simultaneously"""
        # Use functional API
        fourcc = cm.fourchar_to_int('WAVE')

        # Use OO API
        format = cm.AudioFormat(44100.0, 'lpcm')

        # Both should work
        assert fourcc == cm.fourchar_to_int('WAVE')  # Just verify it returns a consistent value
        assert format.is_pcm

    def test_oo_api_uses_functional_api_internally(self):
        """Test that OO API correctly uses functional API internally"""
        # This tests the import structure - OO classes should be able to call
        # functional API functions
        format = cm.AudioFormat(44100.0, 'lpcm', channels_per_frame=2, bits_per_channel=16)

        # The OO API should work correctly, indicating it can access the functional API
        assert format.is_pcm
        assert format.is_stereo
        assert repr(format)  # Should not raise any import errors