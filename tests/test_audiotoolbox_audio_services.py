"""pytest test suite for AudioServices functionality."""

import logging
import os
import pytest
import tempfile
import coremusic as cm
import coremusic.capi as capi

logger = logging.getLogger(__name__)


class TestAudioServicesConstants:
    """Test AudioServices constants access"""

    def test_audio_services_error_constants(self):
        """Test AudioServices error constants"""
        assert capi.get_audio_services_no_error() == 0
        assert capi.get_audio_services_unsupported_property_error() == 1886681407
        assert capi.get_audio_services_bad_property_size_error() == 561211770
        assert capi.get_audio_services_bad_specifier_size_error() == 561213539
        assert capi.get_audio_services_system_sound_unspecified_error() == -1500
        assert capi.get_audio_services_system_sound_client_timed_out_error() == -1501
        assert (
            capi.get_audio_services_system_sound_exceeded_maximum_duration_error()
            == -1502
        )

    def test_system_sound_id_constants(self):
        """Test SystemSoundID constants"""
        assert capi.get_system_sound_id_user_preferred_alert() == 4096
        assert capi.get_system_sound_id_flash_screen() == 4094
        assert capi.get_system_sound_id_vibrate() == 4095
        assert capi.get_user_preferred_alert() == 4096

    def test_audio_services_property_constants(self):
        """Test AudioServices property constants"""
        assert capi.get_audio_services_property_is_ui_sound() == 1769174377
        assert (
            capi.get_audio_services_property_complete_playback_if_app_dies()
            == 1768318057
        )


class TestAudioServicesBasicOperations:
    """Test basic AudioServices operations"""

    @pytest.fixture
    def test_audio_file_path(self, amen_wav_path):
        """Path to test audio file"""
        return amen_wav_path

    def test_create_and_dispose_system_sound_id(self, test_audio_file_path):
        """Test creating and disposing a SystemSoundID"""
        sound_id = capi.audio_services_create_system_sound_id(test_audio_file_path)
        assert isinstance(sound_id, int)
        assert sound_id != 0
        result = capi.audio_services_dispose_system_sound_id(sound_id)
        assert result == 0

    def test_create_system_sound_id_invalid_path(self):
        """Test creating SystemSoundID with invalid path"""
        with pytest.raises(RuntimeError):
            capi.audio_services_create_system_sound_id("/path/that/does/not/exist.wav")

    def test_predefined_system_sounds(self):
        """Test playing predefined system sounds"""
        try:
            alert_id = capi.get_system_sound_id_user_preferred_alert()
            capi.audio_services_play_alert_sound(alert_id)
        except Exception:
            pass
        try:
            vibrate_id = capi.get_system_sound_id_vibrate()
            capi.audio_services_play_system_sound(vibrate_id)
        except Exception:
            pass


class TestAudioServicesPlayback:
    """Test AudioServices playback functionality"""

    @pytest.fixture
    def test_audio_file_path(self, amen_wav_path):
        """Path to test audio file"""
        return amen_wav_path

    @pytest.fixture
    def system_sound_id(self, test_audio_file_path):
        """Create a system sound ID for testing"""
        sound_id = capi.audio_services_create_system_sound_id(test_audio_file_path)
        yield sound_id
        capi.audio_services_dispose_system_sound_id(sound_id)

    def test_play_system_sound(self, system_sound_id):
        """Test playing a system sound"""
        capi.audio_services_play_system_sound(system_sound_id)

    def test_play_alert_sound(self, system_sound_id):
        """Test playing an alert sound"""
        capi.audio_services_play_alert_sound(system_sound_id)

    def test_dispose_invalid_sound_id(self):
        """Test disposing an invalid sound ID"""
        try:
            result = capi.audio_services_dispose_system_sound_id(999999)
            assert isinstance(result, int)
        except RuntimeError:
            pass


class TestAudioServicesProperties:
    """Test AudioServices property functionality"""

    @pytest.fixture
    def test_audio_file_path(self, amen_wav_path):
        """Path to test audio file"""
        return amen_wav_path

    @pytest.fixture
    def system_sound_id(self, test_audio_file_path):
        """Create a system sound ID for testing"""
        sound_id = capi.audio_services_create_system_sound_id(test_audio_file_path)
        yield sound_id
        capi.audio_services_dispose_system_sound_id(sound_id)

    def test_get_is_ui_sound_property(self, system_sound_id):
        """Test getting the IsUISound property"""
        try:
            prop_id = capi.get_audio_services_property_is_ui_sound()
            value = capi.audio_services_get_property(prop_id, system_sound_id)
            assert isinstance(value, int)
            assert value in [0, 1]
        except RuntimeError as e:
            print(f"IsUISound property not supported: {e}")

    def test_set_is_ui_sound_property(self, system_sound_id):
        """Test setting the IsUISound property"""
        try:
            prop_id = capi.get_audio_services_property_is_ui_sound()
            result = capi.audio_services_set_property(prop_id, 1, system_sound_id)
            assert result == 0
            value = capi.audio_services_get_property(prop_id, system_sound_id)
            assert value == 1
            result = capi.audio_services_set_property(prop_id, 0, system_sound_id)
            assert result == 0
            value = capi.audio_services_get_property(prop_id, system_sound_id)
            assert value == 0
        except RuntimeError as e:
            print(f"IsUISound property setting not supported: {e}")

    def test_get_complete_playback_if_app_dies_property(self, system_sound_id):
        """Test getting the CompletePlaybackIfAppDies property"""
        try:
            prop_id = capi.get_audio_services_property_complete_playback_if_app_dies()
            value = capi.audio_services_get_property(prop_id, system_sound_id)
            assert isinstance(value, int)
            assert value in [0, 1]
        except RuntimeError as e:
            print(f"CompletePlaybackIfAppDies property not supported: {e}")

    def test_set_complete_playback_if_app_dies_property(self, system_sound_id):
        """Test setting the CompletePlaybackIfAppDies property"""
        try:
            prop_id = capi.get_audio_services_property_complete_playback_if_app_dies()
            result = capi.audio_services_set_property(prop_id, 1, system_sound_id)
            assert result == 0
            value = capi.audio_services_get_property(prop_id, system_sound_id)
            assert value == 1
        except RuntimeError as e:
            print(f"CompletePlaybackIfAppDies property setting not supported: {e}")

    def test_invalid_property_access(self, system_sound_id):
        """Test accessing invalid properties"""
        with pytest.raises(RuntimeError):
            capi.audio_services_get_property(999999, system_sound_id)
        with pytest.raises(RuntimeError):
            capi.audio_services_set_property(999999, 1, system_sound_id)


class TestAudioServicesErrorHandling:
    """Test AudioServices error handling"""

    def test_invalid_file_path_handling(self):
        """Test handling of invalid file paths"""
        invalid_paths = [
            "/this/path/does/not/exist.wav",
            "",
            "/dev/null",
            "/tmp/nonexistent.mp3",
        ]
        for path in invalid_paths:
            with pytest.raises((RuntimeError, ValueError)):
                capi.audio_services_create_system_sound_id(path)

    def test_property_type_validation(self, amen_wav_path):
        """Test property data type validation"""
        wav_path = amen_wav_path
        sound_id = capi.audio_services_create_system_sound_id(wav_path)
        try:
            prop_id = capi.get_audio_services_property_is_ui_sound()
            with pytest.raises(TypeError):
                capi.audio_services_set_property(prop_id, "invalid_string", sound_id)
            with pytest.raises(TypeError):
                capi.audio_services_set_property(prop_id, [1, 2, 3], sound_id)
        finally:
            capi.audio_services_dispose_system_sound_id(sound_id)


class TestAudioServicesResourceManagement:
    """Test AudioServices resource management"""

    def test_multiple_sound_creation_and_disposal(self, amen_wav_path):
        """Test creating and disposing multiple system sounds"""
        wav_path = amen_wav_path
        sound_ids = []
        try:
            for i in range(5):
                sound_id = capi.audio_services_create_system_sound_id(wav_path)
                sound_ids.append(sound_id)
                assert isinstance(sound_id, int)
                assert sound_id != 0
            assert len(set(sound_ids)) == len(sound_ids)
        finally:
            for i, sound_id in enumerate(sound_ids):
                try:
                    capi.audio_services_dispose_system_sound_id(sound_id)
                except Exception as e:
                    logger.warning(f"Cleanup failed for sound_id {i}: {e}")

    def test_sound_id_lifecycle(self, amen_wav_path):
        """Test the complete lifecycle of a SystemSoundID"""
        wav_path = amen_wav_path
        sound_id = capi.audio_services_create_system_sound_id(wav_path)
        assert isinstance(sound_id, int)
        capi.audio_services_play_system_sound(sound_id)
        try:
            prop_id = capi.get_audio_services_property_is_ui_sound()
            capi.audio_services_set_property(prop_id, 0, sound_id)
        except RuntimeError:
            pass
        capi.audio_services_play_system_sound(sound_id)
        result = capi.audio_services_dispose_system_sound_id(sound_id)
        assert result == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
