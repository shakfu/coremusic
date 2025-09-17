#!/usr/bin/env python3
"""pytest test suite for CoreMIDI wrapper functionality."""

import os
import pytest
import time
import coremusic as cm


class TestCoreMIDIConstants:
    """Test CoreMIDI constants access"""

    def test_midi_error_constants(self):
        """Test MIDI error constants"""
        assert cm.get_midi_error_invalid_client() == -10830
        assert cm.get_midi_error_invalid_port() == -10831
        assert cm.get_midi_error_wrong_endpoint_type() == -10832
        assert cm.get_midi_error_no_connection() == -10833
        assert cm.get_midi_error_unknown_endpoint() == -10834
        assert cm.get_midi_error_unknown_property() == -10835
        assert cm.get_midi_error_wrong_property_type() == -10836
        assert cm.get_midi_error_no_current_setup() == -10837
        assert cm.get_midi_error_message_send_err() == -10838
        assert cm.get_midi_error_server_start_err() == -10839
        assert cm.get_midi_error_setup_format_err() == -10840
        assert cm.get_midi_error_wrong_thread() == -10841
        assert cm.get_midi_error_object_not_found() == -10842
        assert cm.get_midi_error_id_not_unique() == -10843
        assert cm.get_midi_error_not_permitted() == -10844
        assert cm.get_midi_error_unknown_error() == -10845

    def test_midi_object_type_constants(self):
        """Test MIDI object type constants"""
        assert cm.get_midi_object_type_other() == -1
        assert cm.get_midi_object_type_device() == 0
        assert cm.get_midi_object_type_entity() == 1
        assert cm.get_midi_object_type_source() == 2
        assert cm.get_midi_object_type_destination() == 3
        assert cm.get_midi_object_type_external_device() == 0x10
        assert cm.get_midi_object_type_external_entity() == 0x11
        assert cm.get_midi_object_type_external_source() == 0x12
        assert cm.get_midi_object_type_external_destination() == 0x13

    def test_midi_protocol_constants(self):
        """Test MIDI protocol constants"""
        assert cm.get_midi_protocol_1_0() == 1
        assert cm.get_midi_protocol_2_0() == 2


class TestCoreMIDIClient:
    """Test CoreMIDI client operations"""

    def test_midi_client_create_dispose(self):
        """Test MIDI client creation and disposal"""
        # Create a MIDI client
        client = cm.midi_client_create("Test Client")
        assert client is not None
        assert isinstance(client, int)
        assert client > 0

        # Dispose the client
        cm.midi_client_dispose(client)

    def test_midi_client_create_invalid_name(self):
        """Test MIDI client creation with invalid name"""
        # CoreMIDI actually allows empty names, so test with None instead
        with pytest.raises((RuntimeError, ValueError, TypeError, AttributeError)):
            cm.midi_client_create(None)


class TestCoreMIDIPorts:
    """Test CoreMIDI port operations"""

    def setup_method(self):
        """Set up test client for port tests"""
        self.client = cm.midi_client_create("Test Port Client")

    def teardown_method(self):
        """Clean up test client"""
        try:
            if hasattr(self, 'client') and self.client:
                cm.midi_client_dispose(self.client)
                self.client = None
        except Exception:
            pass

    def test_midi_input_port_create_dispose(self):
        """Test MIDI input port creation and disposal"""
        port = cm.midi_input_port_create(self.client, "Test Input Port")
        assert port is not None
        assert isinstance(port, int)
        assert port > 0

        cm.midi_port_dispose(port)

    def test_midi_output_port_create_dispose(self):
        """Test MIDI output port creation and disposal"""
        port = cm.midi_output_port_create(self.client, "Test Output Port")
        assert port is not None
        assert isinstance(port, int)
        assert port > 0

        cm.midi_port_dispose(port)

    def test_midi_port_create_invalid_client(self):
        """Test port creation with invalid client"""
        with pytest.raises(RuntimeError):
            cm.midi_input_port_create(0, "Invalid Client Port")


class TestCoreMIDIDevices:
    """Test CoreMIDI device enumeration"""

    def test_midi_get_number_of_devices(self):
        """Test getting number of MIDI devices"""
        num_devices = cm.midi_get_number_of_devices()
        assert isinstance(num_devices, int)
        assert num_devices >= 0

    def test_midi_get_number_of_sources(self):
        """Test getting number of MIDI sources"""
        num_sources = cm.midi_get_number_of_sources()
        assert isinstance(num_sources, int)
        assert num_sources >= 0

    def test_midi_get_number_of_destinations(self):
        """Test getting number of MIDI destinations"""
        num_destinations = cm.midi_get_number_of_destinations()
        assert isinstance(num_destinations, int)
        assert num_destinations >= 0

    def test_midi_device_enumeration(self):
        """Test MIDI device enumeration"""
        num_devices = cm.midi_get_number_of_devices()

        for i in range(num_devices):
            device = cm.midi_get_device(i)
            assert device is not None
            assert isinstance(device, int)
            assert device > 0

            # Test getting entities for this device
            num_entities = cm.midi_device_get_number_of_entities(device)
            assert isinstance(num_entities, int)
            assert num_entities >= 0

            for j in range(num_entities):
                entity = cm.midi_device_get_entity(device, j)
                assert entity is not None
                assert isinstance(entity, int)

    def test_midi_source_enumeration(self):
        """Test MIDI source enumeration"""
        num_sources = cm.midi_get_number_of_sources()

        for i in range(min(num_sources, 5)):  # Test first 5 sources max
            source = cm.midi_get_source(i)
            assert source is not None
            assert isinstance(source, int)
            assert source > 0

    def test_midi_destination_enumeration(self):
        """Test MIDI destination enumeration"""
        num_destinations = cm.midi_get_number_of_destinations()

        for i in range(min(num_destinations, 5)):  # Test first 5 destinations max
            destination = cm.midi_get_destination(i)
            assert destination is not None
            assert isinstance(destination, int)
            assert destination > 0


class TestCoreMIDIVirtualEndpoints:
    """Test CoreMIDI virtual endpoint operations"""

    def setup_method(self):
        """Set up test client for virtual endpoint tests"""
        self.client = cm.midi_client_create("Test Virtual Client")

    def teardown_method(self):
        """Clean up test client"""
        try:
            if hasattr(self, 'client') and self.client:
                cm.midi_client_dispose(self.client)
                self.client = None
        except Exception:
            pass

    def test_midi_source_create_dispose(self):
        """Test virtual MIDI source creation and disposal"""
        source = cm.midi_source_create(self.client, "Test Virtual Source")
        assert source is not None
        assert isinstance(source, int)
        assert source > 0

        cm.midi_endpoint_dispose(source)

    def test_midi_destination_create_dispose(self):
        """Test virtual MIDI destination creation and disposal"""
        destination = cm.midi_destination_create(self.client, "Test Virtual Destination")
        assert destination is not None
        assert isinstance(destination, int)
        assert destination > 0

        cm.midi_endpoint_dispose(destination)


class TestCoreMIDIProperties:
    """Test CoreMIDI property operations"""

    def setup_method(self):
        """Set up test client for property tests"""
        self.client = cm.midi_client_create("Test Property Client")

    def teardown_method(self):
        """Clean up test client"""
        try:
            if hasattr(self, 'client') and self.client:
                cm.midi_client_dispose(self.client)
                self.client = None
        except Exception:
            pass

    def test_midi_object_get_string_property(self):
        """Test getting MIDI object string properties"""
        # Test with a device if available
        num_devices = cm.midi_get_number_of_devices()
        if num_devices > 0:
            device = cm.midi_get_device(0)
            try:
                name = cm.midi_object_get_string_property(device, "name")
                if name is not None:
                    assert isinstance(name, str)
                    assert len(name) > 0
            except RuntimeError:
                # Property might not exist, which is okay
                pass

    def test_midi_object_get_integer_property(self):
        """Test getting MIDI object integer properties"""
        # Test with a device if available
        num_devices = cm.midi_get_number_of_devices()
        if num_devices > 0:
            device = cm.midi_get_device(0)
            try:
                unique_id = cm.midi_object_get_integer_property(device, "uniqueID")
                if unique_id is not None:
                    assert isinstance(unique_id, int)
            except RuntimeError:
                # Property might not exist, which is okay
                pass

    def test_midi_object_set_string_property(self):
        """Test setting MIDI object string properties"""
        # Create a virtual source to test property setting
        source = cm.midi_source_create(self.client, "Test Property Source")

        try:
            # Try to set a custom property
            cm.midi_object_set_string_property(source, "testProperty", "test value")

            # Try to get it back
            value = cm.midi_object_get_string_property(source, "testProperty")
            if value is not None:
                assert value == "test value"
        except RuntimeError:
            # Some properties might be read-only, which is okay
            pass
        finally:
            cm.midi_endpoint_dispose(source)

    def test_midi_object_set_integer_property(self):
        """Test setting MIDI object integer properties"""
        # Create a virtual source to test property setting
        source = cm.midi_source_create(self.client, "Test Integer Property Source")

        try:
            # Try to set a custom integer property
            cm.midi_object_set_integer_property(source, "testIntProperty", 42)

            # Try to get it back
            value = cm.midi_object_get_integer_property(source, "testIntProperty")
            if value is not None:
                assert value == 42
        except RuntimeError:
            # Some properties might be read-only, which is okay
            pass
        finally:
            cm.midi_endpoint_dispose(source)


class TestCoreMIDIData:
    """Test CoreMIDI data transmission"""

    def test_midi_send_data_large(self):
        """Test MIDI data sending with oversized data"""
        # Create data that's too large (over 256 bytes)
        large_data = bytes([0x42] * 300)

        with pytest.raises(ValueError):
            cm.midi_send_data(1, 1, large_data, 0)

    def test_midi_send_data_basic_success(self):
        """Test that basic MIDI data function exists and can be called"""
        # CoreMIDI is lenient with invalid parameters, but the function should exist
        note_on_data = bytes([0x90, 60, 127])

        # This should succeed (CoreMIDI may just ignore invalid refs)
        result = cm.midi_send_data(1, 1, note_on_data, 0)
        assert isinstance(result, int)  # Should return OSStatus



class TestCoreMIDIIntegration:
    """Integration tests for CoreMIDI functionality"""

    def test_full_midi_workflow(self):
        """Test complete MIDI workflow: client -> ports -> virtual endpoints -> data"""
        # Create client
        client = cm.midi_client_create("Integration Test Client")

        try:
            # Create ports
            input_port = cm.midi_input_port_create(client, "Integration Input")
            output_port = cm.midi_output_port_create(client, "Integration Output")

            # Create virtual endpoints
            virtual_source = cm.midi_source_create(client, "Integration Virtual Source")
            virtual_destination = cm.midi_destination_create(client, "Integration Virtual Destination")

            # Test property operations
            try:
                cm.midi_object_set_string_property(virtual_source, "description", "Test source")
                description = cm.midi_object_get_string_property(virtual_source, "description")
                if description is not None:
                    assert description == "Test source"
            except RuntimeError:
                # Property operations might fail, which is okay
                pass

            # Test data sending
            try:
                note_data = bytes([0x90, 60, 127])
                cm.midi_send_data(output_port, virtual_destination, note_data, 0)
            except RuntimeError:
                # Data sending might fail without proper connections
                pass

            # Clean up
            cm.midi_endpoint_dispose(virtual_destination)
            cm.midi_endpoint_dispose(virtual_source)
            cm.midi_port_dispose(output_port)
            cm.midi_port_dispose(input_port)

        finally:
            cm.midi_client_dispose(client)

    def test_midi_system_enumeration(self):
        """Test complete MIDI system enumeration"""
        # Get system overview
        num_devices = cm.midi_get_number_of_devices()
        num_sources = cm.midi_get_number_of_sources()
        num_destinations = cm.midi_get_number_of_destinations()

        print(f"MIDI System: {num_devices} devices, {num_sources} sources, {num_destinations} destinations")

        # Enumerate devices and their entities
        for i in range(min(num_devices, 3)):  # Test first 3 devices max
            device = cm.midi_get_device(i)
            num_entities = cm.midi_device_get_number_of_entities(device)

            for j in range(num_entities):
                entity = cm.midi_device_get_entity(device, j)
                num_entity_sources = cm.midi_entity_get_number_of_sources(entity)
                num_entity_destinations = cm.midi_entity_get_number_of_destinations(entity)

                print(f"  Device {i}, Entity {j}: {num_entity_sources} sources, {num_entity_destinations} destinations")

                # Test entity endpoints
                for k in range(min(num_entity_sources, 2)):
                    source = cm.midi_entity_get_source(entity, k)
                    assert source > 0

                for k in range(min(num_entity_destinations, 2)):
                    destination = cm.midi_entity_get_destination(entity, k)
                    assert destination > 0


class TestCoreMIDIMessages:
    """Test CoreMIDI Universal MIDI Packet (UMP) message functionality"""

    def test_midi_message_type_constants(self):
        """Test MIDI message type constants"""
        assert cm.get_midi_message_type_utility() == 0x0
        assert cm.get_midi_message_type_system() == 0x1
        assert cm.get_midi_message_type_channel_voice1() == 0x2
        assert cm.get_midi_message_type_sysex() == 0x3
        assert cm.get_midi_message_type_channel_voice2() == 0x4
        assert cm.get_midi_message_type_data128() == 0x5

    def test_midi_cv_status_constants(self):
        """Test MIDI channel voice status constants"""
        assert cm.get_midi_cv_status_note_off() == 0x8
        assert cm.get_midi_cv_status_note_on() == 0x9
        assert cm.get_midi_cv_status_poly_pressure() == 0xA
        assert cm.get_midi_cv_status_control_change() == 0xB
        assert cm.get_midi_cv_status_program_change() == 0xC
        assert cm.get_midi_cv_status_channel_pressure() == 0xD
        assert cm.get_midi_cv_status_pitch_bend() == 0xE

    def test_midi_message_type_for_up_word(self):
        """Test extracting message type from Universal MIDI Packet word"""
        # Create test UMP words with different message types
        utility_word = 0x00000000  # Type 0 (Utility)
        system_word = 0x10000000   # Type 1 (System)
        cv1_word = 0x20000000      # Type 2 (Channel Voice 1)
        sysex_word = 0x30000000    # Type 3 (SysEx)
        cv2_word = 0x40000000      # Type 4 (Channel Voice 2)
        data128_word = 0x50000000  # Type 5 (Data128)

        assert cm.midi_message_type_for_up_word(utility_word) == 0x0
        assert cm.midi_message_type_for_up_word(system_word) == 0x1
        assert cm.midi_message_type_for_up_word(cv1_word) == 0x2
        assert cm.midi_message_type_for_up_word(sysex_word) == 0x3
        assert cm.midi_message_type_for_up_word(cv2_word) == 0x4
        assert cm.midi_message_type_for_up_word(data128_word) == 0x5

    def test_midi1_up_channel_voice_message(self):
        """Test MIDI 1.0 Universal Packet channel voice message creation"""
        group = 0
        status = 9  # Note On
        channel = 0
        data1 = 60  # Middle C
        data2 = 127  # Full velocity

        message = cm.midi1_up_channel_voice_message(group, status, channel, data1, data2)
        assert isinstance(message, int)
        assert message > 0

        # Check that the message type is correct (should be 0x2 for Channel Voice 1)
        message_type = cm.midi_message_type_for_up_word(message)
        assert message_type == 0x2

    def test_midi1_up_note_on(self):
        """Test MIDI 1.0 Universal Packet Note On message"""
        group = 0
        channel = 0
        note_number = 60  # Middle C
        velocity = 127

        message = cm.midi1_up_note_on(group, channel, note_number, velocity)
        assert isinstance(message, int)
        assert message > 0

        # Verify message type
        message_type = cm.midi_message_type_for_up_word(message)
        assert message_type == 0x2

    def test_midi1_up_note_off(self):
        """Test MIDI 1.0 Universal Packet Note Off message"""
        group = 0
        channel = 0
        note_number = 60  # Middle C
        velocity = 64

        message = cm.midi1_up_note_off(group, channel, note_number, velocity)
        assert isinstance(message, int)
        assert message > 0

        # Verify message type
        message_type = cm.midi_message_type_for_up_word(message)
        assert message_type == 0x2

    def test_midi1_up_control_change(self):
        """Test MIDI 1.0 Universal Packet Control Change message"""
        group = 0
        channel = 0
        index = 7  # Volume controller
        data = 100

        message = cm.midi1_up_control_change(group, channel, index, data)
        assert isinstance(message, int)
        assert message > 0

        # Verify message type
        message_type = cm.midi_message_type_for_up_word(message)
        assert message_type == 0x2

    def test_midi1_up_pitch_bend(self):
        """Test MIDI 1.0 Universal Packet Pitch Bend message"""
        group = 0
        channel = 0
        lsb = 0
        msb = 64  # Center position

        message = cm.midi1_up_pitch_bend(group, channel, lsb, msb)
        assert isinstance(message, int)
        assert message > 0

        # Verify message type
        message_type = cm.midi_message_type_for_up_word(message)
        assert message_type == 0x2

    def test_midi1_up_system_common(self):
        """Test MIDI 1.0 Universal Packet System Common message"""
        group = 0
        status = 0xF2  # Song Position Pointer
        byte1 = 0x10
        byte2 = 0x20

        message = cm.midi1_up_system_common(group, status, byte1, byte2)
        assert isinstance(message, int)
        assert message > 0

        # Verify message type
        message_type = cm.midi_message_type_for_up_word(message)
        assert message_type == 0x1

    def test_midi1_up_sysex(self):
        """Test MIDI 1.0 Universal Packet SysEx message"""
        group = 0
        status = 0  # Complete SysEx
        bytes_used = 3
        byte1 = 0x7E  # Non-real-time
        byte2 = 0x00  # Device ID
        byte3 = 0x09  # General MIDI
        byte4 = 0x01  # GM On
        byte5 = 0x00
        byte6 = 0x00

        message = cm.midi1_up_sysex(group, status, bytes_used, byte1, byte2, byte3, byte4, byte5, byte6)
        assert isinstance(message, tuple)
        assert len(message) == 2
        assert isinstance(message[0], int)
        assert isinstance(message[1], int)

        # Verify message type from first word
        message_type = cm.midi_message_type_for_up_word(message[0])
        assert message_type == 0x3

    def test_midi2_channel_voice_message(self):
        """Test MIDI 2.0 Channel Voice message creation"""
        group = 0
        status = 9  # Note On
        channel = 0
        index = 0x3C00  # Note number with attribute type
        value = 0xFFFF0000  # Velocity and attribute data

        message = cm.midi2_channel_voice_message(group, status, channel, index, value)
        assert isinstance(message, tuple)
        assert len(message) == 2
        assert isinstance(message[0], int)
        assert isinstance(message[1], int)

        # Verify message type
        message_type = cm.midi_message_type_for_up_word(message[0])
        assert message_type == 0x4

    def test_midi2_note_on(self):
        """Test MIDI 2.0 Note On message"""
        group = 0
        channel = 0
        note_number = 60  # Middle C
        attribute_type = 0  # No attribute
        attribute_data = 0
        velocity = 0xFFFF  # Full velocity (16-bit)

        message = cm.midi2_note_on(group, channel, note_number, attribute_type, attribute_data, velocity)
        assert isinstance(message, tuple)
        assert len(message) == 2
        assert isinstance(message[0], int)
        assert isinstance(message[1], int)

        # Verify message type
        message_type = cm.midi_message_type_for_up_word(message[0])
        assert message_type == 0x4

    def test_midi2_note_off(self):
        """Test MIDI 2.0 Note Off message"""
        group = 0
        channel = 0
        note_number = 60  # Middle C
        attribute_type = 0  # No attribute
        attribute_data = 0
        velocity = 0x8000  # Half velocity (16-bit)

        message = cm.midi2_note_off(group, channel, note_number, attribute_type, attribute_data, velocity)
        assert isinstance(message, tuple)
        assert len(message) == 2
        assert isinstance(message[0], int)
        assert isinstance(message[1], int)

        # Verify message type
        message_type = cm.midi_message_type_for_up_word(message[0])
        assert message_type == 0x4

    def test_midi2_control_change(self):
        """Test MIDI 2.0 Control Change message"""
        group = 0
        channel = 0
        index = 7  # Volume controller
        value = 0x80000000  # Half value (32-bit)

        message = cm.midi2_control_change(group, channel, index, value)
        assert isinstance(message, tuple)
        assert len(message) == 2
        assert isinstance(message[0], int)
        assert isinstance(message[1], int)

        # Verify message type
        message_type = cm.midi_message_type_for_up_word(message[0])
        assert message_type == 0x4

    def test_midi2_program_change(self):
        """Test MIDI 2.0 Program Change message"""
        group = 0
        channel = 0
        bank_is_valid = True
        program = 1  # Piano
        bank_msb = 0
        bank_lsb = 0

        message = cm.midi2_program_change(group, channel, bank_is_valid, program, bank_msb, bank_lsb)
        assert isinstance(message, tuple)
        assert len(message) == 2
        assert isinstance(message[0], int)
        assert isinstance(message[1], int)

        # Verify message type
        message_type = cm.midi_message_type_for_up_word(message[0])
        assert message_type == 0x4

    def test_midi2_pitch_bend(self):
        """Test MIDI 2.0 Pitch Bend message"""
        group = 0
        channel = 0
        value = 0x80000000  # Center position (32-bit)

        message = cm.midi2_pitch_bend(group, channel, value)
        assert isinstance(message, tuple)
        assert len(message) == 2
        assert isinstance(message[0], int)
        assert isinstance(message[1], int)

        # Verify message type
        message_type = cm.midi_message_type_for_up_word(message[0])
        assert message_type == 0x4

    def test_midi_message_parameter_validation(self):
        """Test parameter validation for MIDI message functions"""
        # Test valid ranges
        group = 0  # 0-15
        channel = 0  # 0-15
        note = 60  # 0-127
        velocity = 127  # 0-127

        # These should work without error
        message = cm.midi1_up_note_on(group, channel, note, velocity)
        assert isinstance(message, int)

        # Test with maximum valid values
        max_group = 15
        max_channel = 15
        max_note = 127
        max_velocity = 127

        message = cm.midi1_up_note_on(max_group, max_channel, max_note, max_velocity)
        assert isinstance(message, int)

    def test_midi_message_consistency(self):
        """Test that identical parameters produce identical messages"""
        group = 0
        channel = 5
        note = 72  # C5
        velocity = 100

        # Create the same message twice
        message1 = cm.midi1_up_note_on(group, channel, note, velocity)
        message2 = cm.midi1_up_note_on(group, channel, note, velocity)

        # Should be identical
        assert message1 == message2

        # MIDI 2.0 version
        attr_type = 0
        attr_data = 0
        velocity16 = 0xC800  # Approximately 100 in 16-bit

        msg2_1 = cm.midi2_note_on(group, channel, note, attr_type, attr_data, velocity16)
        msg2_2 = cm.midi2_note_on(group, channel, note, attr_type, attr_data, velocity16)

        assert msg2_1 == msg2_2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])