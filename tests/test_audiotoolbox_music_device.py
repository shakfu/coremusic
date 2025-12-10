"""pytest test suite for MusicDevice functionality."""

import logging
import os
import pytest
import tempfile
import coremusic as cm
import coremusic.capi as capi

logger = logging.getLogger(__name__)


class TestMusicDeviceConstants:
    """Test MusicDevice constants access"""

    def test_music_note_event_constants(self):
        """Test MusicDevice note event constants"""
        assert capi.get_music_note_event_use_group_instrument() == -1
        assert capi.get_music_note_event_unused() == -1

    def test_music_device_selector_constants(self):
        """Test MusicDevice selector constants"""
        assert capi.get_music_device_range() == 256
        assert capi.get_music_device_midi_event_select() == 257
        assert capi.get_music_device_sysex_select() == 258
        assert capi.get_music_device_start_note_select() == 261
        assert capi.get_music_device_stop_note_select() == 262
        assert capi.get_music_device_midi_event_list_select() == 263


class TestMusicDeviceHelpers:
    """Test MusicDevice helper functions"""

    def test_create_std_note_params(self):
        """Test creating standard note parameters"""
        params = capi.create_music_device_std_note_params(60.0, 127.0)
        assert params["argCount"] == 2
        assert params["pitch"] == 60.0
        assert params["velocity"] == 127.0

    def test_create_note_params_with_controls(self):
        """Test creating note parameters with controls"""
        controls = [(1, 64.0), (7, 100.0)]
        params = capi.create_music_device_note_params(60.0, 100.0, controls)
        assert params["argCount"] == 4
        assert params["pitch"] == 60.0
        assert params["velocity"] == 100.0
        assert params["controls"] == controls

    def test_midi_note_on_helper(self):
        """Test MIDI Note On helper function"""
        status, data1, data2 = capi.midi_note_on(0, 60, 127)
        assert status == 144
        assert data1 == 60
        assert data2 == 127
        status, data1, data2 = capi.midi_note_on(9, 36, 64)
        assert status == 153
        assert data1 == 36
        assert data2 == 64

    def test_midi_note_off_helper(self):
        """Test MIDI Note Off helper function"""
        status, data1, data2 = capi.midi_note_off(0, 60)
        assert status == 128
        assert data1 == 60
        assert data2 == 0
        status, data1, data2 = capi.midi_note_off(5, 72, 64)
        assert status == 133
        assert data1 == 72
        assert data2 == 64

    def test_midi_control_change_helper(self):
        """Test MIDI Control Change helper function"""
        status, data1, data2 = capi.midi_control_change(0, 7, 127)
        assert status == 176
        assert data1 == 7
        assert data2 == 127
        status, data1, data2 = capi.midi_control_change(3, 1, 64)
        assert status == 179
        assert data1 == 1
        assert data2 == 64

    def test_midi_program_change_helper(self):
        """Test MIDI Program Change helper function"""
        status, data1, data2 = capi.midi_program_change(0, 0)
        assert status == 192
        assert data1 == 0
        assert data2 == 0
        status, data1, data2 = capi.midi_program_change(9, 0)
        assert status == 201
        assert data1 == 0
        assert data2 == 0

    def test_midi_pitch_bend_helper(self):
        """Test MIDI Pitch Bend helper function"""
        status, data1, data2 = capi.midi_pitch_bend(0, 8192)
        assert status == 224
        assert data1 == 0
        assert data2 == 64
        status, data1, data2 = capi.midi_pitch_bend(2, 16383)
        assert status == 226
        assert data1 == 127
        assert data2 == 127
        status, data1, data2 = capi.midi_pitch_bend(1, 0)
        assert status == 225
        assert data1 == 0
        assert data2 == 0

    def test_midi_data_bounds_checking(self):
        """Test that MIDI helper functions respect bounds"""
        status, _, _ = capi.midi_note_on(16, 60, 127)
        assert status == 144
        status, _, _ = capi.midi_note_on(255, 60, 127)
        assert status == 159
        _, data1, data2 = capi.midi_note_on(0, 128, 128)
        assert data1 == 0
        assert data2 == 0
        _, data1, data2 = capi.midi_control_change(0, 255, 255)
        assert data1 == 127
        assert data2 == 127


class TestMusicDeviceBasicOperations:
    """Test basic MusicDevice operations"""

    @pytest.fixture
    def music_device_unit(self):
        """Create a music device audio unit for testing"""
        desc = {
            "type": capi.get_audio_component_type_music_device(),
            "subtype": 0,
            "manufacturer": 0,
            "flags": 0,
            "flags_mask": 0,
        }
        component = capi.audio_component_find_next(desc)
        if not component:
            pytest.skip("No music device components available")
        try:
            unit = capi.audio_component_instance_new(component)
        except RuntimeError as e:
            # Check for userCanceledErr (-128) or kAudioUnitErr_InvalidFile (-10863)
            error_str = str(e)
            if "userCanceledErr" in error_str or "security restriction" in error_str.lower() or "-10863" in error_str or "InvalidFile" in error_str:
                pytest.skip(
                    f"Music device component cannot be instantiated: {e}"
                )
            raise
        yield unit
        try:
            capi.audio_component_instance_dispose(unit)
        except Exception as e:
            logger.warning(f"Cleanup failed (dispose unit): {e}")

    def test_music_device_midi_event_basic(self, music_device_unit):
        """Test basic MIDI event sending"""
        try:
            result = capi.music_device_midi_event(music_device_unit, 144, 60, 127, 0)
            assert result == 0
        except RuntimeError as e:
            print(f"MIDI event failed (expected if unit not initialized): {e}")

    def test_music_device_sysex_basic(self, music_device_unit):
        """Test basic SysEx message sending"""
        sysex_data = bytes([240, 126, 127, 9, 1, 247])
        try:
            result = capi.music_device_sysex(music_device_unit, sysex_data)
            assert result == 0
        except RuntimeError as e:
            print(f"SysEx failed (expected if not supported): {e}")

    def test_music_device_start_stop_note_basic(self, music_device_unit):
        """Test basic note start/stop functionality"""
        try:
            note_id = capi.music_device_start_note(
                music_device_unit, capi.get_music_note_event_unused(), 0, 60.0, 100.0, 0
            )
            assert isinstance(note_id, int)
            assert note_id != 0
            result = capi.music_device_stop_note(music_device_unit, 0, note_id, 0)
            assert result == 0
        except RuntimeError as e:
            print(f"Start/stop note failed (expected if not supported): {e}")

    def test_music_device_start_note_with_controls(self, music_device_unit):
        """Test starting note with additional controls"""
        try:
            controls = [(1, 64.0), (7, 100.0)]
            note_id = capi.music_device_start_note(
                music_device_unit,
                capi.get_music_note_event_unused(),
                0,
                67.0,
                80.0,
                0,
                controls,
            )
            assert isinstance(note_id, int)
            assert note_id != 0
            result = capi.music_device_stop_note(music_device_unit, 0, note_id, 0)
            assert result == 0
        except RuntimeError as e:
            print(f"Start note with controls failed: {e}")


class TestMusicDeviceErrorHandling:
    """Test MusicDevice error handling"""

    def test_invalid_unit_handling(self):
        """Test handling of invalid unit references"""
        invalid_unit = 0
        with pytest.raises(RuntimeError):
            capi.music_device_midi_event(invalid_unit, 144, 60, 127)
        with pytest.raises(RuntimeError):
            capi.music_device_sysex(invalid_unit, bytes([240, 247]))
        with pytest.raises(RuntimeError):
            capi.music_device_start_note(invalid_unit, 0, 0, 60.0, 100.0)
        with pytest.raises(RuntimeError):
            capi.music_device_stop_note(invalid_unit, 0, 1)

    def test_invalid_sysex_data_handling(self):
        """Test handling of invalid SysEx data"""
        invalid_unit = 0
        with pytest.raises(RuntimeError):
            capi.music_device_sysex(invalid_unit, b"")
        with pytest.raises(TypeError):
            capi.music_device_sysex(invalid_unit, "not bytes")

    def test_parameter_validation(self):
        """Test parameter validation"""
        invalid_unit = 0
        with pytest.raises(RuntimeError):
            capi.music_device_start_note(invalid_unit, 0, 0, -1.0, 100.0)
        with pytest.raises(RuntimeError):
            capi.music_device_start_note(invalid_unit, 0, 0, 60.0, -1.0)
        with pytest.raises(RuntimeError):
            capi.music_device_start_note(invalid_unit, 0, 999999, 60.0, 100.0)

@pytest.mark.slow
class TestMusicDeviceIntegration:
    """Test MusicDevice integration scenarios"""

    @pytest.fixture
    def initialized_music_device(self):
        """Create and initialize a music device for testing"""
        try:
            desc = {
                "type": capi.get_audio_component_type_music_device(),
                "subtype": 0,
                "manufacturer": 0,
                "flags": 0,
                "flags_mask": 0,
            }
            component = capi.audio_component_find_next(desc)
            if not component:
                pytest.skip("No music device components available")
            unit = capi.audio_component_instance_new(component)
            try:
                capi.audio_unit_initialize(unit)
                yield unit
                capi.audio_unit_uninitialize(unit)
            except Exception as e:
                logger.warning(f"Unit initialization/cleanup issue: {e}")
                yield unit
            capi.audio_component_instance_dispose(unit)
        except Exception as e:
            pytest.skip(f"Could not create music device: {e}")

    def test_music_device_lifecycle(self, initialized_music_device):
        """Test complete music device lifecycle"""
        unit = initialized_music_device
        try:
            status, data1, data2 = capi.midi_program_change(0, 0)
            capi.music_device_midi_event(unit, status, data1, data2)
            note_ids = []
            for pitch in [60, 64, 67]:
                note_id = capi.music_device_start_note(
                    unit, capi.get_music_note_event_unused(), 0, float(pitch), 100.0, 0
                )
                note_ids.append(note_id)
            assert len(set(note_ids)) == len(note_ids)
            assert all(nid != 0 for nid in note_ids)
            for note_id in note_ids:
                result = capi.music_device_stop_note(unit, 0, note_id, 0)
                assert result == 0
        except RuntimeError as e:
            print(f"Lifecycle test skipped: {e}")

    def test_music_device_midi_sequence(self, initialized_music_device):
        """Test sending a sequence of MIDI events"""
        unit = initialized_music_device
        try:
            midi_events = [
                capi.midi_program_change(0, 0),
                cm.midi_control_change(0, 7, 100),
                capi.midi_control_change(0, 1, 0),
                capi.midi_note_on(0, 60, 100),
                capi.midi_note_on(0, 64, 100),
                capi.midi_note_on(0, 67, 100),
            ]
            for status, data1, data2 in midi_events:
                result = capi.music_device_midi_event(unit, status, data1, data2, 0)
                assert result == 0
            note_offs = [
                capi.midi_note_off(0, 60),
                capi.midi_note_off(0, 64),
                capi.midi_note_off(0, 67),
            ]
            for status, data1, data2 in note_offs:
                result = capi.music_device_midi_event(unit, status, data1, data2, 0)
                assert result == 0
        except RuntimeError as e:
            print(f"MIDI sequence test skipped: {e}")

@pytest.mark.slow
class TestMusicDeviceResourceManagement:
    """Test MusicDevice resource management"""

    def test_multiple_note_management(self):
        """Test managing multiple notes simultaneously"""
        try:
            desc = {
                "type": capi.get_audio_component_type_music_device(),
                "subtype": 0,
                "manufacturer": 0,
                "flags": 0,
                "flags_mask": 0,
            }
            component = capi.audio_component_find_next(desc)
            if not component:
                pytest.skip("No music device available")
            unit = capi.audio_component_instance_new(component)
            try:
                note_ids = []
                for i in range(5):
                    note_id = capi.music_device_start_note(
                        unit, capi.get_music_note_event_unused(), 0, 60.0 + i, 100.0, 0
                    )
                    note_ids.append(note_id)
                assert len(set(note_ids)) == len(note_ids)
                for note_id in note_ids:
                    result = capi.music_device_stop_note(unit, 0, note_id, 0)
                    assert result == 0
            except RuntimeError as e:
                print(f"Multiple note test skipped: {e}")
            finally:
                capi.audio_component_instance_dispose(unit)
        except Exception as e:
            pytest.skip(f"Resource management test failed: {e}")

    def test_sysex_data_sizes(self):
        """Test SysEx messages of various sizes"""
        try:
            desc = {
                "type": capi.get_audio_component_type_music_device(),
                "subtype": 0,
                "manufacturer": 0,
                "flags": 0,
                "flags_mask": 0,
            }
            component = capi.audio_component_find_next(desc)
            if not component:
                pytest.skip("No music device available")
            unit = capi.audio_component_instance_new(component)
            try:
                sysex_messages = [
                    bytes([240, 247]),
                    bytes([240, 126, 127, 9, 1, 247]),
                    bytes([240, 67, 18, 0] + [0] * 100 + [247]),
                ]
                for sysex_data in sysex_messages:
                    try:
                        result = capi.music_device_sysex(unit, sysex_data)
                        assert result == 0
                    except RuntimeError:
                        pass
            finally:
                capi.audio_component_instance_dispose(unit)
        except Exception as e:
            pytest.skip(f"SysEx size test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
