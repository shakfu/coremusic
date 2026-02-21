"""Tests for the base infrastructure of the object-oriented coremusic API."""

import pytest
import gc
import coremusic.capi as capi
from coremusic.audio import AudioFormat
from coremusic.base import CoreAudioObject
from coremusic.exceptions import (
    AUGraphError,
    AudioDeviceError,
    AudioFileError,
    AudioQueueError,
    AudioUnitError,
    CoreAudioError,
    MIDIError,
    MusicPlayerError,
)


class TestCoreAudioObject:
    """Test CoreAudioObject base class functionality"""

    def test_core_audio_object_creation(self):
        """Test basic CoreAudioObject creation and properties"""
        obj = CoreAudioObject()
        assert not obj.is_disposed

    def test_core_audio_object_disposal(self):
        """Test explicit disposal"""
        obj = CoreAudioObject()
        obj.dispose()
        assert obj.is_disposed
        obj.dispose()
        assert obj.is_disposed

    def test_core_audio_object_automatic_disposal(self):
        """Test automatic disposal via __dealloc__"""
        obj = CoreAudioObject()
        del obj
        gc.collect()

    def test_ensure_not_disposed_check(self):
        """Test _ensure_not_disposed method"""
        obj = CoreAudioObject()
        obj._ensure_not_disposed()
        obj.dispose()
        with pytest.raises(RuntimeError, match="has been disposed"):
            obj._ensure_not_disposed()


class TestAudioFormat:
    """Test AudioFormat class functionality"""

    def test_audio_format_creation(self):
        """Test AudioFormat creation with various parameters"""
        format1 = AudioFormat(44100.0, "lpcm")
        assert format1.sample_rate == 44100.0
        assert format1.format_id == "lpcm"
        assert format1.channels_per_frame == 2
        assert format1.bits_per_channel == 16
        format2 = AudioFormat(
            sample_rate=48000.0,
            format_id="aac ",
            format_flags=12,
            bytes_per_packet=1024,
            frames_per_packet=512,
            bytes_per_frame=4,
            channels_per_frame=6,
            bits_per_channel=24,
        )
        assert format2.sample_rate == 48000.0
        assert format2.format_id == "aac "
        assert format2.format_flags == 12
        assert format2.bytes_per_packet == 1024
        assert format2.frames_per_packet == 512
        assert format2.bytes_per_frame == 4
        assert format2.channels_per_frame == 6
        assert format2.bits_per_channel == 24

    def test_audio_format_properties(self):
        """Test AudioFormat computed properties"""
        pcm_format = AudioFormat(44100.0, "lpcm", channels_per_frame=2)
        assert pcm_format.is_pcm
        assert pcm_format.is_stereo
        assert not pcm_format.is_mono
        aac_format = AudioFormat(44100.0, "aac ", channels_per_frame=1)
        assert not aac_format.is_pcm
        assert not aac_format.is_stereo
        assert aac_format.is_mono
        surround_format = AudioFormat(48000.0, "lpcm", channels_per_frame=6)
        assert surround_format.is_pcm
        assert not surround_format.is_stereo
        assert not surround_format.is_mono

    def test_audio_format_repr(self):
        """Test AudioFormat string representation"""
        format = AudioFormat(44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16)
        repr_str = repr(format)
        assert "AudioFormat" in repr_str
        assert "44100.0" in repr_str
        assert "lpcm" in repr_str
        assert "channels=2" in repr_str
        assert "bits=16" in repr_str


class TestExceptionHierarchy:
    """Test exception hierarchy and functionality"""

    def test_core_audio_error(self):
        """Test CoreAudioError base exception"""
        error = CoreAudioError("Test message")
        assert str(error) == "Test message"
        assert error.status_code == 0
        error_with_code = CoreAudioError("Test with code", 42)
        assert str(error_with_code) == "Test with code"
        assert error_with_code.status_code == 42
        assert isinstance(error, Exception)

    def test_audio_file_error(self):
        """Test AudioFileError"""
        error = AudioFileError("File error", 123)
        assert str(error) == "File error"
        assert error.status_code == 123
        assert isinstance(error, CoreAudioError)
        assert isinstance(error, Exception)

    def test_audio_queue_error(self):
        """Test AudioQueueError"""
        error = AudioQueueError("Queue error", -50)
        assert str(error) == "Queue error"
        assert error.status_code == -50
        assert isinstance(error, CoreAudioError)

    def test_audio_unit_error(self):
        """Test AudioUnitError"""
        error = AudioUnitError("Unit error")
        assert str(error) == "Unit error"
        assert error.status_code == 0
        assert isinstance(error, CoreAudioError)

    def test_midi_error(self):
        """Test MIDIError"""
        error = MIDIError("MIDI error", -10830)
        assert str(error) == "MIDI error"
        assert error.status_code == -10830
        assert isinstance(error, CoreAudioError)

    def test_music_player_error(self):
        """Test MusicPlayerError"""
        error = MusicPlayerError("Player error")
        assert str(error) == "Player error"
        assert isinstance(error, CoreAudioError)

    def test_audio_device_error(self):
        """Test AudioDeviceError"""
        error = AudioDeviceError("Device error", -66687)
        assert str(error) == "Device error"
        assert error.status_code == -66687
        assert isinstance(error, CoreAudioError)

    def test_au_graph_error(self):
        """Test AUGraphError"""
        error = AUGraphError("Graph error", -10860)
        assert str(error) == "Graph error"
        assert error.status_code == -10860
        assert isinstance(error, CoreAudioError)

    def test_exception_raising_and_catching(self):
        """Test raising and catching various exceptions"""
        with pytest.raises(AudioFileError):
            raise AudioFileError("Test file error")
        with pytest.raises(CoreAudioError):
            raise AudioUnitError("Test unit error")
        with pytest.raises(Exception):
            raise MIDIError("Test MIDI error")


class TestObjectOrientedAPIAvailability:
    """Test that all OO API classes are available through domain subpackages"""

    AUDIO_CLASSES = [
        "AudioFormat",
        "AudioFile",
        "AudioFileStream",
        "AudioBuffer",
        "AudioQueue",
        "AudioComponentDescription",
        "AudioComponent",
        "AudioUnit",
        "AudioConverter",
        "ExtendedAudioFile",
        "AudioDevice",
        "AudioDeviceManager",
        "AUGraph",
        "AudioClock",
        "ClockTimeFormat",
    ]

    MIDI_CLASSES = [
        "MIDIClient",
        "MIDIPort",
        "MIDIInputPort",
        "MIDIOutputPort",
        "MusicPlayer",
        "MusicSequence",
        "MusicTrack",
    ]

    EXCEPTION_CLASSES = [
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

    BASE_CLASSES = [
        "CoreAudioObject",
        "AudioPlayer",
        "NUMPY_AVAILABLE",
    ]

    @pytest.mark.parametrize("class_name", AUDIO_CLASSES)
    def test_audio_class_available(self, class_name):
        """Test that required class is available via coremusic.audio"""
        import coremusic.audio as audio

        assert hasattr(audio, class_name), f"Missing class: {class_name}"

    @pytest.mark.parametrize("class_name", MIDI_CLASSES)
    def test_midi_class_available(self, class_name):
        """Test that required class is available via coremusic.midi"""
        import coremusic.midi as midi

        assert hasattr(midi, class_name), f"Missing class: {class_name}"

    @pytest.mark.parametrize("class_name", EXCEPTION_CLASSES)
    def test_exception_class_available(self, class_name):
        """Test that required class is available via coremusic.exceptions"""
        import coremusic.exceptions as exceptions

        assert hasattr(exceptions, class_name), f"Missing class: {class_name}"

    @pytest.mark.parametrize("class_name", BASE_CLASSES)
    def test_base_class_available(self, class_name):
        """Test that required class is available via coremusic.base"""
        import coremusic.base as base

        assert hasattr(base, class_name), f"Missing class: {class_name}"

    def test_functional_api_available_via_capi(self):
        """Test that functional API is available via capi submodule"""
        assert hasattr(capi, "fourchar_to_int")
        assert hasattr(capi, "int_to_fourchar")
        assert hasattr(capi, "audio_file_open_url")
        assert hasattr(capi, "midi_client_create")
        assert capi.fourchar_to_int("TEST") == 1413829460
        assert capi.int_to_fourchar(1413829460) == "TEST"


class TestDualAPIInteraction:
    """Test interaction between functional and object-oriented APIs"""

    def test_both_apis_accessible(self):
        """Test that both APIs can be used simultaneously"""
        fourcc = capi.fourchar_to_int("WAVE")
        format = AudioFormat(44100.0, "lpcm")
        assert fourcc == capi.fourchar_to_int("WAVE")
        assert format.is_pcm

    def test_oo_api_uses_functional_api_internally(self):
        """Test that OO API correctly uses functional API internally"""
        format = AudioFormat(44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16)
        assert format.is_pcm
        assert format.is_stereo
        assert repr(format)
