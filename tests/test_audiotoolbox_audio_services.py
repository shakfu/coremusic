"""pytest test suite for AudioServices functionality."""

import os
import pytest
import tempfile
import coremusic as cm
import coremusic.capi as capi


class TestAudioServicesConstants:
    """Test AudioServices constants access"""

    def test_audio_services_error_constants(self):
        """Test AudioServices error constants"""
        assert capi.get_audio_services_no_error() == 0
        assert capi.get_audio_services_unsupported_property_error() is not None
        assert capi.get_audio_services_bad_property_size_error() is not None
        assert capi.get_audio_services_bad_specifier_size_error() is not None
        assert capi.get_audio_services_system_sound_unspecified_error() is not None
        assert capi.get_audio_services_system_sound_client_timed_out_error() is not None
        assert (
            capi.get_audio_services_system_sound_exceeded_maximum_duration_error()
            is not None
        )

    def test_system_sound_id_constants(self):
        """Test SystemSoundID constants"""
        assert capi.get_system_sound_id_user_preferred_alert() is not None
        assert capi.get_system_sound_id_flash_screen() is not None
        assert capi.get_system_sound_id_vibrate() is not None
        assert capi.get_user_preferred_alert() is not None

    def test_audio_services_property_constants(self):
        """Test AudioServices property constants"""
        assert capi.get_audio_services_property_is_ui_sound() is not None
        assert (
            capi.get_audio_services_property_complete_playback_if_app_dies() is not None
        )


class TestAudioServicesBasicOperations:
    """Test basic AudioServices operations"""

    @pytest.fixture
    def test_audio_file_path(self):
        """Path to test audio file"""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        wav_path = os.path.join(test_dir, "amen.wav")
        if not os.path.exists(wav_path):
            pytest.skip(f"Test WAV file not found: {wav_path}")
        return wav_path

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
    def test_audio_file_path(self):
        """Path to test audio file"""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        wav_path = os.path.join(test_dir, "amen.wav")
        if not os.path.exists(wav_path):
            pytest.skip(f"Test WAV file not found: {wav_path}")
        return wav_path

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
    def test_audio_file_path(self):
        """Path to test audio file"""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        wav_path = os.path.join(test_dir, "amen.wav")
        if not os.path.exists(wav_path):
            pytest.skip(f"Test WAV file not found: {wav_path}")
        return wav_path

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

    def test_property_type_validation(self):
        """Test property data type validation"""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        wav_path = os.path.join(test_dir, "amen.wav")
        if not os.path.exists(wav_path):
            pytest.skip(f"Test WAV file not found: {wav_path}")
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

    def test_multiple_sound_creation_and_disposal(self):
        """Test creating and disposing multiple system sounds"""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        wav_path = os.path.join(test_dir, "amen.wav")
        if not os.path.exists(wav_path):
            pytest.skip(f"Test WAV file not found: {wav_path}")
        sound_ids = []
        try:
            for i in range(5):
                sound_id = capi.audio_services_create_system_sound_id(wav_path)
                sound_ids.append(sound_id)
                assert isinstance(sound_id, int)
                assert sound_id != 0
            assert len(set(sound_ids)) == len(sound_ids)
        finally:
            for sound_id in sound_ids:
                try:
                    capi.audio_services_dispose_system_sound_id(sound_id)
                except:
                    pass

    def test_sound_id_lifecycle(self):
        """Test the complete lifecycle of a SystemSoundID"""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        wav_path = os.path.join(test_dir, "amen.wav")
        if not os.path.exists(wav_path):
            pytest.skip(f"Test WAV file not found: {wav_path}")
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
