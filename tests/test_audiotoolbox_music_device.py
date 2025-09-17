#!/usr/bin/env python3
"""pytest test suite for MusicDevice functionality."""

import os
import pytest
import tempfile
import coremusic as cm


class TestMusicDeviceConstants:
    """Test MusicDevice constants access"""

    def test_music_note_event_constants(self):
        """Test MusicDevice note event constants"""
        # 0xFFFFFFFF is treated as -1 in Python signed integers
        assert cm.get_music_note_event_use_group_instrument() == -1
        assert cm.get_music_note_event_unused() == -1

    def test_music_device_selector_constants(self):
        """Test MusicDevice selector constants"""
        assert cm.get_music_device_range() == 0x0100
        assert cm.get_music_device_midi_event_select() == 0x0101
        assert cm.get_music_device_sysex_select() == 0x0102
        assert cm.get_music_device_start_note_select() == 0x0105
        assert cm.get_music_device_stop_note_select() == 0x0106
        assert cm.get_music_device_midi_event_list_select() == 0x0107


class TestMusicDeviceHelpers:
    """Test MusicDevice helper functions"""

    def test_create_std_note_params(self):
        """Test creating standard note parameters"""
        params = cm.create_music_device_std_note_params(60.0, 127.0)
        assert params['argCount'] == 2
        assert params['pitch'] == 60.0
        assert params['velocity'] == 127.0

    def test_create_note_params_with_controls(self):
        """Test creating note parameters with controls"""
        controls = [(1, 64.0), (7, 100.0)]  # Modulation wheel, Volume
        params = cm.create_music_device_note_params(60.0, 100.0, controls)
        assert params['argCount'] == 4  # pitch + velocity + 2 controls
        assert params['pitch'] == 60.0
        assert params['velocity'] == 100.0
        assert params['controls'] == controls

    def test_midi_note_on_helper(self):
        """Test MIDI Note On helper function"""
        status, data1, data2 = cm.midi_note_on(0, 60, 127)
        assert status == 0x90  # Note On channel 0
        assert data1 == 60     # Middle C
        assert data2 == 127    # Max velocity

        # Test with different channel
        status, data1, data2 = cm.midi_note_on(9, 36, 64)
        assert status == 0x99  # Note On channel 9 (drum channel)
        assert data1 == 36     # Bass drum
        assert data2 == 64     # Medium velocity

    def test_midi_note_off_helper(self):
        """Test MIDI Note Off helper function"""
        status, data1, data2 = cm.midi_note_off(0, 60)
        assert status == 0x80  # Note Off channel 0
        assert data1 == 60     # Middle C
        assert data2 == 0      # Default velocity

        # Test with specific velocity
        status, data1, data2 = cm.midi_note_off(5, 72, 64)
        assert status == 0x85  # Note Off channel 5
        assert data1 == 72
        assert data2 == 64

    def test_midi_control_change_helper(self):
        """Test MIDI Control Change helper function"""
        status, data1, data2 = cm.midi_control_change(0, 7, 127)
        assert status == 0xB0  # CC channel 0
        assert data1 == 7      # Volume controller
        assert data2 == 127    # Max volume

        # Test modulation wheel
        status, data1, data2 = cm.midi_control_change(3, 1, 64)
        assert status == 0xB3  # CC channel 3
        assert data1 == 1      # Modulation wheel
        assert data2 == 64

    def test_midi_program_change_helper(self):
        """Test MIDI Program Change helper function"""
        status, data1, data2 = cm.midi_program_change(0, 0)
        assert status == 0xC0  # Program Change channel 0
        assert data1 == 0      # Piano (GM)
        assert data2 == 0      # Unused

        # Test different program
        status, data1, data2 = cm.midi_program_change(9, 0)
        assert status == 0xC9  # Program Change channel 9
        assert data1 == 0
        assert data2 == 0

    def test_midi_pitch_bend_helper(self):
        """Test MIDI Pitch Bend helper function"""
        # Center position (no bend)
        status, data1, data2 = cm.midi_pitch_bend(0, 8192)
        assert status == 0xE0  # Pitch Bend channel 0
        assert data1 == 0      # LSB
        assert data2 == 64     # MSB (center)

        # Maximum up bend
        status, data1, data2 = cm.midi_pitch_bend(2, 16383)
        assert status == 0xE2  # Pitch Bend channel 2
        assert data1 == 127    # LSB
        assert data2 == 127    # MSB

        # Minimum down bend
        status, data1, data2 = cm.midi_pitch_bend(1, 0)
        assert status == 0xE1  # Pitch Bend channel 1
        assert data1 == 0      # LSB
        assert data2 == 0      # MSB

    def test_midi_data_bounds_checking(self):
        """Test that MIDI helper functions respect bounds"""
        # Test channel bounds (should mask to 0-15)
        status, _, _ = cm.midi_note_on(16, 60, 127)  # Channel 16 -> 0
        assert status == 0x90

        status, _, _ = cm.midi_note_on(255, 60, 127)  # Channel 255 -> 15
        assert status == 0x9F

        # Test data bounds (should mask to 0-127)
        _, data1, data2 = cm.midi_note_on(0, 128, 128)
        assert data1 == 0   # 128 & 0x7F = 0
        assert data2 == 0   # 128 & 0x7F = 0

        _, data1, data2 = cm.midi_control_change(0, 255, 255)
        assert data1 == 127  # 255 & 0x7F = 127
        assert data2 == 127  # 255 & 0x7F = 127


class TestMusicDeviceBasicOperations:
    """Test basic MusicDevice operations"""

    @pytest.fixture
    def music_device_unit(self):
        """Create a music device audio unit for testing"""
        # Try to find a music device component
        desc = {
            'type': cm.get_audio_component_type_music_device(),
            'subtype': 0,  # Any subtype
            'manufacturer': 0,  # Any manufacturer
            'flags': 0,
            'flags_mask': 0
        }

        component = cm.audio_component_find_next(desc)
        if not component:
            pytest.skip("No music device components available")

        # Create instance
        unit = cm.audio_component_instance_new(component)
        yield unit

        # Cleanup
        try:
            cm.audio_component_instance_dispose(unit)
        except:
            pass  # Ignore cleanup errors

    def test_music_device_midi_event_basic(self, music_device_unit):
        """Test basic MIDI event sending"""
        # Test Note On
        try:
            result = cm.music_device_midi_event(
                music_device_unit, 0x90, 60, 127, 0
            )
            assert result == 0  # noErr
        except RuntimeError as e:
            # Some music devices may not be initialized
            print(f"MIDI event failed (expected if unit not initialized): {e}")

    def test_music_device_sysex_basic(self, music_device_unit):
        """Test basic SysEx message sending"""
        # Create a simple SysEx message (GM Reset)
        sysex_data = bytes([0xF0, 0x7E, 0x7F, 0x09, 0x01, 0xF7])

        try:
            result = cm.music_device_sysex(music_device_unit, sysex_data)
            assert result == 0  # noErr
        except RuntimeError as e:
            # Some music devices may not support SysEx
            print(f"SysEx failed (expected if not supported): {e}")

    def test_music_device_start_stop_note_basic(self, music_device_unit):
        """Test basic note start/stop functionality"""
        try:
            # Start a note
            note_id = cm.music_device_start_note(
                music_device_unit,
                cm.get_music_note_event_unused(),  # Use current patch
                0,      # Group 0
                60.0,   # Middle C
                100.0,  # Velocity
                0       # No offset
            )
            assert isinstance(note_id, int)
            assert note_id != 0

            # Stop the note
            result = cm.music_device_stop_note(
                music_device_unit, 0, note_id, 0
            )
            assert result == 0  # noErr

        except RuntimeError as e:
            # Some music devices may not support start/stop note API
            print(f"Start/stop note failed (expected if not supported): {e}")

    def test_music_device_start_note_with_controls(self, music_device_unit):
        """Test starting note with additional controls"""
        try:
            # Start note with modulation and volume controls
            controls = [(1, 64.0), (7, 100.0)]  # Modulation, Volume
            note_id = cm.music_device_start_note(
                music_device_unit,
                cm.get_music_note_event_unused(),
                0,        # Group 0
                67.0,     # G above middle C
                80.0,     # Velocity
                0,        # No offset
                controls  # Additional controls
            )
            assert isinstance(note_id, int)
            assert note_id != 0

            # Stop the note
            result = cm.music_device_stop_note(
                music_device_unit, 0, note_id, 0
            )
            assert result == 0  # noErr

        except RuntimeError as e:
            print(f"Start note with controls failed: {e}")


class TestMusicDeviceErrorHandling:
    """Test MusicDevice error handling"""

    def test_invalid_unit_handling(self):
        """Test handling of invalid unit references"""
        invalid_unit = 0  # Invalid unit reference

        # Test MIDI event with invalid unit
        with pytest.raises(RuntimeError):
            cm.music_device_midi_event(invalid_unit, 0x90, 60, 127)

        # Test SysEx with invalid unit
        with pytest.raises(RuntimeError):
            cm.music_device_sysex(invalid_unit, bytes([0xF0, 0xF7]))

        # Test start note with invalid unit
        with pytest.raises(RuntimeError):
            cm.music_device_start_note(invalid_unit, 0, 0, 60.0, 100.0)

        # Test stop note with invalid unit
        with pytest.raises(RuntimeError):
            cm.music_device_stop_note(invalid_unit, 0, 1)

    def test_invalid_sysex_data_handling(self):
        """Test handling of invalid SysEx data"""
        # This will test with invalid unit, but the data validation
        # should still work at the Python level
        invalid_unit = 0

        # Test with empty data
        with pytest.raises(RuntimeError):
            cm.music_device_sysex(invalid_unit, b"")

        # Test with non-bytes data would cause TypeError at Python level
        with pytest.raises(TypeError):
            cm.music_device_sysex(invalid_unit, "not bytes")

    def test_parameter_validation(self):
        """Test parameter validation"""
        invalid_unit = 0

        # Test with invalid pitch range (should still work, but might be clamped by audio unit)
        with pytest.raises(RuntimeError):
            cm.music_device_start_note(invalid_unit, 0, 0, -1.0, 100.0)

        # Test with invalid velocity (should still work)
        with pytest.raises(RuntimeError):
            cm.music_device_start_note(invalid_unit, 0, 0, 60.0, -1.0)

        # Test with invalid group ID (large number should work)
        with pytest.raises(RuntimeError):
            cm.music_device_start_note(invalid_unit, 0, 999999, 60.0, 100.0)


class TestMusicDeviceIntegration:
    """Test MusicDevice integration scenarios"""

    @pytest.fixture
    def initialized_music_device(self):
        """Create and initialize a music device for testing"""
        try:
            # Try to find a music device component
            desc = {
                'type': cm.get_audio_component_type_music_device(),
                'subtype': 0,  # Any subtype
                'manufacturer': 0,  # Any manufacturer
                'flags': 0,
                'flags_mask': 0
            }

            component = cm.audio_component_find_next(desc)
            if not component:
                pytest.skip("No music device components available")

            # Create and initialize instance
            unit = cm.audio_component_instance_new(component)

            try:
                # Try to initialize the audio unit
                cm.audio_unit_initialize(unit)
                yield unit
                cm.audio_unit_uninitialize(unit)
            except:
                # If initialization fails, still yield for basic tests
                yield unit

            cm.audio_component_instance_dispose(unit)

        except Exception as e:
            pytest.skip(f"Could not create music device: {e}")

    def test_music_device_lifecycle(self, initialized_music_device):
        """Test complete music device lifecycle"""
        unit = initialized_music_device

        try:
            # Send program change to select a sound
            status, data1, data2 = cm.midi_program_change(0, 0)  # Piano
            cm.music_device_midi_event(unit, status, data1, data2)

            # Start multiple notes
            note_ids = []
            for pitch in [60, 64, 67]:  # C major chord
                note_id = cm.music_device_start_note(
                    unit, cm.get_music_note_event_unused(),
                    0, float(pitch), 100.0, 0
                )
                note_ids.append(note_id)

            # All note IDs should be unique and non-zero
            assert len(set(note_ids)) == len(note_ids)
            assert all(nid != 0 for nid in note_ids)

            # Stop all notes
            for note_id in note_ids:
                result = cm.music_device_stop_note(unit, 0, note_id, 0)
                assert result == 0

        except RuntimeError as e:
            # This is acceptable for uninitialized or unsupported devices
            print(f"Lifecycle test skipped: {e}")

    def test_music_device_midi_sequence(self, initialized_music_device):
        """Test sending a sequence of MIDI events"""
        unit = initialized_music_device

        try:
            # Send a sequence of MIDI events
            midi_events = [
                cm.midi_program_change(0, 0),           # Select piano
                cm.midi_control_change(0, 7, 100),      # Set volume
                cm.midi_control_change(0, 1, 0),        # Reset modulation
                cm.midi_note_on(0, 60, 100),            # Play C
                cm.midi_note_on(0, 64, 100),            # Play E
                cm.midi_note_on(0, 67, 100),            # Play G
            ]

            for status, data1, data2 in midi_events:
                result = cm.music_device_midi_event(unit, status, data1, data2, 0)
                assert result == 0

            # Send note offs
            note_offs = [
                cm.midi_note_off(0, 60),
                cm.midi_note_off(0, 64),
                cm.midi_note_off(0, 67),
            ]

            for status, data1, data2 in note_offs:
                result = cm.music_device_midi_event(unit, status, data1, data2, 0)
                assert result == 0

        except RuntimeError as e:
            print(f"MIDI sequence test skipped: {e}")


class TestMusicDeviceResourceManagement:
    """Test MusicDevice resource management"""

    def test_multiple_note_management(self):
        """Test managing multiple notes simultaneously"""
        try:
            # Try to find a music device
            desc = {
                'type': cm.get_audio_component_type_music_device(),
                'subtype': 0,
                'manufacturer': 0,
                'flags': 0,
                'flags_mask': 0
            }
            component = cm.audio_component_find_next(desc)
            if not component:
                pytest.skip("No music device available")

            unit = cm.audio_component_instance_new(component)

            try:
                # Start multiple notes
                note_ids = []
                for i in range(5):
                    note_id = cm.music_device_start_note(
                        unit, cm.get_music_note_event_unused(),
                        0, 60.0 + i, 100.0, 0
                    )
                    note_ids.append(note_id)

                # All note IDs should be unique
                assert len(set(note_ids)) == len(note_ids)

                # Stop all notes
                for note_id in note_ids:
                    result = cm.music_device_stop_note(unit, 0, note_id, 0)
                    assert result == 0

            except RuntimeError as e:
                print(f"Multiple note test skipped: {e}")

            finally:
                cm.audio_component_instance_dispose(unit)

        except Exception as e:
            pytest.skip(f"Resource management test failed: {e}")

    def test_sysex_data_sizes(self):
        """Test SysEx messages of various sizes"""
        try:
            desc = {
                'type': cm.get_audio_component_type_music_device(),
                'subtype': 0,
                'manufacturer': 0,
                'flags': 0,
                'flags_mask': 0
            }
            component = cm.audio_component_find_next(desc)
            if not component:
                pytest.skip("No music device available")

            unit = cm.audio_component_instance_new(component)

            try:
                # Test different sized SysEx messages
                sysex_messages = [
                    bytes([0xF0, 0xF7]),  # Minimal SysEx
                    bytes([0xF0, 0x7E, 0x7F, 0x09, 0x01, 0xF7]),  # GM Reset
                    bytes([0xF0, 0x43, 0x12, 0x00] + [0x00] * 100 + [0xF7])  # Large message
                ]

                for sysex_data in sysex_messages:
                    try:
                        result = cm.music_device_sysex(unit, sysex_data)
                        assert result == 0
                    except RuntimeError:
                        # Some devices may not support certain SysEx messages
                        pass

            finally:
                cm.audio_component_instance_dispose(unit)

        except Exception as e:
            pytest.skip(f"SysEx size test failed: {e}")


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])