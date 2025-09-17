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


class TestCoreMIDISetup:
    """Test CoreMIDI Setup (device and entity management) functionality"""

    def setup_method(self):
        """Set up test environment for MIDISetup tests"""
        # Note: Many MIDISetup functions require special permissions or driver context
        # These tests focus on functions that can be safely called in user context
        pass

    def teardown_method(self):
        """Clean up test environment"""
        pass

    def test_midi_external_device_create(self):
        """Test creating an external MIDI device"""
        try:
            # Create an external device
            device = cm.midi_external_device_create(
                "Test External Device",
                "Test Manufacturer",
                "Test Model"
            )

            assert isinstance(device, int)
            assert device > 0

            # The device should be successfully created
            # Note: We can't easily dispose of it without proper setup management

        except RuntimeError as e:
            # This might fail if we don't have proper permissions
            # or if the system doesn't allow external device creation
            pytest.skip(f"External device creation failed (expected): {e}")

    def test_midi_external_device_create_invalid_params(self):
        """Test external device creation with invalid parameters"""
        # Test with empty strings - should still work but create device with empty names
        try:
            device = cm.midi_external_device_create("", "", "")
            assert isinstance(device, int)
            assert device > 0
        except RuntimeError:
            # This is acceptable - some systems may not allow empty names
            pass

        # Test with None parameters - should raise an error
        with pytest.raises((TypeError, AttributeError)):
            cm.midi_external_device_create(None, "Manufacturer", "Model")

    def test_midi_device_add_entity_basic(self):
        """Test adding entity to a device (if we have a valid device)"""
        # This test is challenging because we need a valid MIDIDeviceRef
        # and proper permissions to modify devices

        # First try to get an existing device
        num_devices = cm.midi_get_number_of_devices()
        if num_devices == 0:
            pytest.skip("No MIDI devices available for entity testing")

        # Get the first device
        device = cm.midi_get_device(0)

        try:
            # Try to add an entity - this will likely fail due to permissions
            # but we test that the function exists and handles errors properly
            entity = cm.midi_device_add_entity(
                device,
                "Test Entity",
                True,  # embedded
                1,     # num source endpoints
                1      # num destination endpoints
            )

            # If this succeeds, we have a new entity
            assert isinstance(entity, int)
            assert entity > 0

            # Try to remove the entity we just added
            try:
                cm.midi_device_remove_entity(device, entity)
            except RuntimeError:
                # Removal might fail, which is okay
                pass

        except RuntimeError as e:
            # Expected - most devices are read-only to user applications
            assert "failed with status" in str(e)

    def test_midi_device_new_entity_basic(self):
        """Test creating new entity with protocol support (macOS 11.0+)"""
        # This test focuses on the newer MIDIDeviceNewEntity function

        num_devices = cm.midi_get_number_of_devices()
        if num_devices == 0:
            pytest.skip("No MIDI devices available for entity testing")

        device = cm.midi_get_device(0)

        try:
            # Try to create a new entity with MIDI 1.0 protocol
            entity = cm.midi_device_new_entity(
                device,
                "Test MIDI 1.0 Entity",
                1,     # MIDI 1.0 protocol
                True,  # embedded
                1,     # num source endpoints
                1      # num destination endpoints
            )

            assert isinstance(entity, int)
            assert entity > 0

            # Clean up
            try:
                cm.midi_device_remove_entity(device, entity)
            except RuntimeError:
                pass

        except RuntimeError as e:
            # Expected - most devices are read-only to user applications
            assert "failed with status" in str(e)

        try:
            # Try to create a new entity with MIDI 2.0 protocol
            entity = cm.midi_device_new_entity(
                device,
                "Test MIDI 2.0 Entity",
                2,     # MIDI 2.0 protocol
                False, # external connectors
                2,     # num source endpoints
                2      # num destination endpoints
            )

            assert isinstance(entity, int)
            assert entity > 0

            # Clean up
            try:
                cm.midi_device_remove_entity(device, entity)
            except RuntimeError:
                pass

        except RuntimeError as e:
            # Expected - most devices are read-only to user applications
            assert "failed with status" in str(e)

    def test_midi_entity_add_or_remove_endpoints(self):
        """Test adding/removing endpoints from an entity"""
        # Get a device and entity to test with
        num_devices = cm.midi_get_number_of_devices()
        if num_devices == 0:
            pytest.skip("No MIDI devices available for endpoint testing")

        device = cm.midi_get_device(0)
        num_entities = cm.midi_device_get_number_of_entities(device)

        if num_entities == 0:
            pytest.skip("No entities available for endpoint testing")

        entity = cm.midi_device_get_entity(device, 0)

        try:
            # Try to modify endpoints - this will likely fail for existing entities
            # but we test that the function works properly
            result = cm.midi_entity_add_or_remove_endpoints(
                entity,
                2,  # desired source endpoints
                2   # desired destination endpoints
            )

            # If this succeeds, check the result
            assert isinstance(result, int)
            assert result == 0  # Success status

        except RuntimeError as e:
            # Expected - most entities are read-only to user applications
            assert "failed with status" in str(e)

    def test_midi_setup_device_management(self):
        """Test setup device management functions"""
        # These functions are typically only available to drivers
        # We test that they exist and handle invalid devices properly

        invalid_device = 999999  # Definitely invalid device ref

        # Test adding invalid device to setup
        try:
            cm.midi_setup_add_device(invalid_device)
            # If this doesn't raise an error, the function succeeded unexpectedly
            assert False, "Expected RuntimeError for invalid device"
        except RuntimeError as e:
            assert "failed with status" in str(e)

        # Test removing invalid device from setup
        try:
            cm.midi_setup_remove_device(invalid_device)
            # If this doesn't raise an error, the function succeeded unexpectedly
            assert False, "Expected RuntimeError for invalid device"
        except RuntimeError as e:
            assert "failed with status" in str(e)

        # Test adding invalid external device
        try:
            cm.midi_setup_add_external_device(invalid_device)
            assert False, "Expected RuntimeError for invalid device"
        except RuntimeError as e:
            assert "failed with status" in str(e)

        # Test removing invalid external device
        try:
            cm.midi_setup_remove_external_device(invalid_device)
            assert False, "Expected RuntimeError for invalid device"
        except RuntimeError as e:
            assert "failed with status" in str(e)

    def test_midi_setup_function_existence(self):
        """Test that all MIDISetup functions exist and are callable"""
        # Test that all the wrapped functions exist
        assert hasattr(cm, 'midi_external_device_create')
        assert callable(cm.midi_external_device_create)

        assert hasattr(cm, 'midi_device_add_entity')
        assert callable(cm.midi_device_add_entity)

        assert hasattr(cm, 'midi_device_new_entity')
        assert callable(cm.midi_device_new_entity)

        assert hasattr(cm, 'midi_device_remove_entity')
        assert callable(cm.midi_device_remove_entity)

        assert hasattr(cm, 'midi_entity_add_or_remove_endpoints')
        assert callable(cm.midi_entity_add_or_remove_endpoints)

        assert hasattr(cm, 'midi_setup_add_device')
        assert callable(cm.midi_setup_add_device)

        assert hasattr(cm, 'midi_setup_remove_device')
        assert callable(cm.midi_setup_remove_device)

        assert hasattr(cm, 'midi_setup_add_external_device')
        assert callable(cm.midi_setup_add_external_device)

        assert hasattr(cm, 'midi_setup_remove_external_device')
        assert callable(cm.midi_setup_remove_external_device)

    def test_midi_setup_parameter_validation(self):
        """Test parameter validation for MIDISetup functions"""
        # Test parameter type validation

        # String parameters should reject None
        with pytest.raises((TypeError, AttributeError)):
            cm.midi_external_device_create(None, "Manufacturer", "Model")

        with pytest.raises((TypeError, AttributeError)):
            cm.midi_external_device_create("Name", None, "Model")

        with pytest.raises((TypeError, AttributeError)):
            cm.midi_external_device_create("Name", "Manufacturer", None)

        # Integer parameters should reject strings (where inappropriate)
        with pytest.raises((TypeError, ValueError)):
            cm.midi_setup_add_device("not_a_device_ref")

        with pytest.raises((TypeError, ValueError)):
            cm.midi_device_remove_entity("not_a_device", "not_an_entity")

    def test_midi_setup_integration_workflow(self):
        """Test a complete workflow of external device management"""
        try:
            # Step 1: Create an external device
            device = cm.midi_external_device_create(
                "Integration Test Device",
                "Test Company",
                "Test Model v1.0"
            )

            assert isinstance(device, int)
            assert device > 0

            # Step 2: Try to add it to the setup
            try:
                cm.midi_setup_add_external_device(device)

                # Step 3: Try to add an entity to the device
                try:
                    entity = cm.midi_device_add_entity(
                        device,
                        "Test Entity",
                        False,  # external connectors
                        1,      # source endpoints
                        1       # destination endpoints
                    )

                    # Step 4: Try to modify endpoints
                    try:
                        cm.midi_entity_add_or_remove_endpoints(entity, 2, 2)
                    except RuntimeError:
                        pass  # Expected

                    # Step 5: Try to remove entity
                    try:
                        cm.midi_device_remove_entity(device, entity)
                    except RuntimeError:
                        pass  # Expected

                except RuntimeError:
                    pass  # Expected - entity operations might not be allowed

                # Step 6: Try to remove device from setup
                try:
                    cm.midi_setup_remove_external_device(device)
                except RuntimeError:
                    pass  # Expected - might not be allowed

            except RuntimeError:
                pass  # Expected - setup operations might not be allowed

        except RuntimeError:
            # Device creation itself might fail, which is acceptable
            pytest.skip("External device creation not supported in this environment")


class TestCoreMIDIDriver:
    """Test CoreMIDI Driver functionality"""

    def setup_method(self):
        """Set up test environment for MIDIDriver tests"""
        # Note: Many MIDIDriver functions are primarily for driver development
        # These tests focus on functions that can be safely called in user context
        pass

    def teardown_method(self):
        """Clean up test environment"""
        pass

    def test_midi_device_create_basic(self):
        """Test creating a MIDI device using the driver API"""
        try:
            # Create a device using the driver API (with NULL owner = non-driver)
            device = cm.midi_device_create(
                "Driver Test Device",
                "Test Driver Manufacturer",
                "Test Driver Model"
            )

            assert isinstance(device, int)
            assert device > 0

            # Try to dispose the device (only works if not added to setup)
            try:
                cm.midi_device_dispose(device)
            except RuntimeError:
                # This is acceptable - device might have been automatically added to setup
                pass

        except RuntimeError as e:
            # Device creation might fail in some environments
            pytest.skip(f"MIDI device creation failed (expected): {e}")

    def test_midi_device_create_invalid_params(self):
        """Test MIDI device creation with invalid parameters"""
        # Test with None parameters - should raise an error
        with pytest.raises((TypeError, AttributeError)):
            cm.midi_device_create(None, "Manufacturer", "Model")

        with pytest.raises((TypeError, AttributeError)):
            cm.midi_device_create("Name", None, "Model")

        with pytest.raises((TypeError, AttributeError)):
            cm.midi_device_create("Name", "Manufacturer", None)

    def test_midi_device_dispose_invalid(self):
        """Test disposing an invalid device"""
        invalid_device = 999999  # Definitely invalid device ref

        try:
            cm.midi_device_dispose(invalid_device)
            # If this doesn't raise an error, the function succeeded unexpectedly
            assert False, "Expected RuntimeError for invalid device"
        except RuntimeError as e:
            assert "failed with status" in str(e)

    def test_midi_endpoint_ref_cons_basic(self):
        """Test setting and getting endpoint reference constants"""
        # Get an existing endpoint to test with
        num_sources = cm.midi_get_number_of_sources()
        if num_sources == 0:
            pytest.skip("No MIDI sources available for refCon testing")

        source = cm.midi_get_source(0)

        # Test setting refCons
        ref1_value = 12345
        ref2_value = 67890

        try:
            # Set reference constants
            result = cm.midi_endpoint_set_ref_cons(source, ref1_value, ref2_value)
            assert isinstance(result, int)
            assert result == 0  # Success

            # Get reference constants back
            ref1_retrieved, ref2_retrieved = cm.midi_endpoint_get_ref_cons(source)
            assert ref1_retrieved == ref1_value
            assert ref2_retrieved == ref2_value

            # Test with default values (0)
            cm.midi_endpoint_set_ref_cons(source)
            ref1_default, ref2_default = cm.midi_endpoint_get_ref_cons(source)
            assert ref1_default == 0
            assert ref2_default == 0

        except RuntimeError as e:
            # RefCon operations might fail on some endpoints
            pytest.skip(f"RefCon operations failed (expected): {e}")

    def test_midi_endpoint_ref_cons_invalid_endpoint(self):
        """Test refCon operations with invalid endpoint"""
        invalid_endpoint = 999999

        try:
            cm.midi_endpoint_set_ref_cons(invalid_endpoint, 123, 456)
            assert False, "Expected RuntimeError for invalid endpoint"
        except RuntimeError as e:
            assert "failed with status" in str(e)

        try:
            cm.midi_endpoint_get_ref_cons(invalid_endpoint)
            assert False, "Expected RuntimeError for invalid endpoint"
        except RuntimeError as e:
            assert "failed with status" in str(e)

    def test_midi_get_driver_io_runloop(self):
        """Test getting the driver I/O run loop"""
        try:
            runloop = cm.midi_get_driver_io_runloop()
            assert isinstance(runloop, int)
            assert runloop != 0  # Should be a valid CFRunLoopRef

        except Exception as e:
            # This might fail in some environments
            pytest.skip(f"Driver I/O runloop access failed: {e}")

    def test_midi_get_driver_device_list_invalid(self):
        """Test getting device list for invalid driver"""
        invalid_driver = 999999

        try:
            # This should return 0 or a null device list for invalid driver
            dev_list = cm.midi_get_driver_device_list(invalid_driver)
            # The result might be 0 (null) which is acceptable
            assert isinstance(dev_list, int)

        except Exception as e:
            # This operation might fail, which is acceptable
            pass

    def test_midi_driver_enable_monitoring_invalid(self):
        """Test enabling monitoring for invalid driver"""
        invalid_driver = 999999

        try:
            cm.midi_driver_enable_monitoring(invalid_driver, True)
            assert False, "Expected RuntimeError for invalid driver"
        except RuntimeError as e:
            assert "failed with status" in str(e)

        try:
            cm.midi_driver_enable_monitoring(invalid_driver, False)
            assert False, "Expected RuntimeError for invalid driver"
        except RuntimeError as e:
            assert "failed with status" in str(e)

    def test_midi_device_list_operations_basic(self):
        """Test basic device list operations (if we can create devices)"""
        # This test is complex because we need valid devices and device lists
        # We'll test with mock/invalid values to ensure the functions exist and handle errors

        # Test with invalid device list
        invalid_dev_list = 999999

        # Test getting number of devices from invalid list
        try:
            num_devices = cm.midi_device_list_get_number_of_devices(invalid_dev_list)
            # This might return 0 for invalid list, which is acceptable
            assert isinstance(num_devices, int)
            assert num_devices >= 0

        except Exception:
            # Function might fail, which is acceptable
            pass

        # Test getting device from invalid list
        try:
            device = cm.midi_device_list_get_device(invalid_dev_list, 0)
            # This should fail since the list is invalid
            assert False, "Expected error for invalid device list"
        except (RuntimeError, IndexError):
            # Expected failure
            pass

        # Test adding device to invalid list
        try:
            cm.midi_device_list_add_device(invalid_dev_list, 123456)
            assert False, "Expected RuntimeError for invalid device list"
        except RuntimeError as e:
            assert "failed with status" in str(e)

        # Test disposing invalid list
        try:
            cm.midi_device_list_dispose(invalid_dev_list)
            assert False, "Expected RuntimeError for invalid device list"
        except RuntimeError as e:
            assert "failed with status" in str(e)

    def test_midi_driver_function_existence(self):
        """Test that all MIDIDriver functions exist and are callable"""
        # Test that all the wrapped functions exist
        assert hasattr(cm, 'midi_device_create')
        assert callable(cm.midi_device_create)

        assert hasattr(cm, 'midi_device_dispose')
        assert callable(cm.midi_device_dispose)

        assert hasattr(cm, 'midi_device_list_get_number_of_devices')
        assert callable(cm.midi_device_list_get_number_of_devices)

        assert hasattr(cm, 'midi_device_list_get_device')
        assert callable(cm.midi_device_list_get_device)

        assert hasattr(cm, 'midi_device_list_add_device')
        assert callable(cm.midi_device_list_add_device)

        assert hasattr(cm, 'midi_device_list_dispose')
        assert callable(cm.midi_device_list_dispose)

        assert hasattr(cm, 'midi_endpoint_set_ref_cons')
        assert callable(cm.midi_endpoint_set_ref_cons)

        assert hasattr(cm, 'midi_endpoint_get_ref_cons')
        assert callable(cm.midi_endpoint_get_ref_cons)

        assert hasattr(cm, 'midi_get_driver_io_runloop')
        assert callable(cm.midi_get_driver_io_runloop)

        assert hasattr(cm, 'midi_get_driver_device_list')
        assert callable(cm.midi_get_driver_device_list)

        assert hasattr(cm, 'midi_driver_enable_monitoring')
        assert callable(cm.midi_driver_enable_monitoring)

    def test_midi_driver_parameter_validation(self):
        """Test parameter validation for MIDIDriver functions"""
        # Test parameter type validation

        # String parameters should reject None
        with pytest.raises((TypeError, AttributeError)):
            cm.midi_device_create(None, "Manufacturer", "Model")

        # Integer parameters should reject strings (where inappropriate)
        with pytest.raises((TypeError, ValueError)):
            cm.midi_device_dispose("not_a_device_ref")

        with pytest.raises((TypeError, ValueError)):
            cm.midi_endpoint_set_ref_cons("not_an_endpoint", 123, 456)

        # Test index bounds checking
        invalid_dev_list = 999999
        with pytest.raises((IndexError, RuntimeError)):
            cm.midi_device_list_get_device(invalid_dev_list, -1)

    def test_midi_driver_integration_workflow(self):
        """Test a complete workflow of driver-style device management"""
        try:
            # Step 1: Create a device using driver API
            device = cm.midi_device_create(
                "Integration Driver Device",
                "Integration Test Company",
                "Integration Test Model"
            )

            assert isinstance(device, int)
            assert device > 0

            # Step 2: Test that we can't dispose it once it's in the system
            # (This will likely fail since the device is automatically added)
            try:
                cm.midi_device_dispose(device)
                # If disposal succeeds, the device wasn't automatically added
                pass
            except RuntimeError:
                # Expected - device was automatically added to system
                pass

            # Step 3: Try to get an endpoint from the device to test refCons
            try:
                # Get entities from the device
                num_entities = cm.midi_device_get_number_of_entities(device)

                if num_entities > 0:
                    entity = cm.midi_device_get_entity(device, 0)

                    # Try to get a source from the entity
                    num_sources = cm.midi_entity_get_number_of_sources(entity)
                    if num_sources > 0:
                        source = cm.midi_entity_get_source(entity, 0)

                        # Test refCons on this endpoint
                        cm.midi_endpoint_set_ref_cons(source, 0x1234, 0x5678)
                        ref1, ref2 = cm.midi_endpoint_get_ref_cons(source)
                        assert ref1 == 0x1234
                        assert ref2 == 0x5678

            except (RuntimeError, IndexError):
                # Expected - newly created devices might not have entities/endpoints
                pass

        except RuntimeError:
            # Device creation itself might fail, which is acceptable
            pytest.skip("Driver-style device creation not supported in this environment")


class TestCoreMIDIThruConnection:
    """Test CoreMIDI Thru Connection functionality"""

    def setup_method(self):
        """Set up test environment for MIDIThruConnection tests"""
        # Store created connections for cleanup
        self.test_connections = []
        self.test_owner_id = "com.test.pytest.thruconnection"

    def teardown_method(self):
        """Clean up test environment"""
        # Clean up any connections we created
        for connection in self.test_connections:
            try:
                cm.midi_thru_connection_dispose(connection)
            except RuntimeError:
                pass  # Connection might already be disposed

    def test_midi_thru_connection_params_initialize(self):
        """Test initializing thru connection parameters"""
        params = cm.midi_thru_connection_params_initialize()

        # Check that we get a dictionary with expected keys
        assert isinstance(params, dict)
        assert 'version' in params
        assert 'sources' in params
        assert 'destinations' in params
        assert 'channelMap' in params
        assert 'filterOutSysEx' in params
        assert 'filterOutMTC' in params
        assert 'filterOutBeatClock' in params

        # Check channel map is initialized correctly (0-15)
        assert len(params['channelMap']) == 16
        assert params['channelMap'] == list(range(16))

        # Check transform structures
        assert isinstance(params['noteNumber'], dict)
        assert 'transform' in params['noteNumber']
        assert 'param' in params['noteNumber']

        # Check initial values are sensible
        assert params['version'] == 0
        assert params['sources'] == []
        assert params['destinations'] == []
        assert params['lowVelocity'] == 0
        assert params['highVelocity'] == 0

    def test_midi_thru_connection_create_basic(self):
        """Test creating a basic thru connection"""
        try:
            # Create a basic connection with default parameters
            connection = cm.midi_thru_connection_create()

            assert isinstance(connection, int)
            assert connection > 0

            # Store for cleanup
            self.test_connections.append(connection)

            # Test disposal
            cm.midi_thru_connection_dispose(connection)
            self.test_connections.remove(connection)

        except RuntimeError as e:
            # Thru connection creation might fail due to permissions or lack of endpoints
            pytest.skip(f"Thru connection creation failed (expected): {e}")

    def test_midi_thru_connection_create_persistent(self):
        """Test creating a persistent thru connection"""
        try:
            # Create a persistent connection
            connection = cm.midi_thru_connection_create(self.test_owner_id)

            assert isinstance(connection, int)
            assert connection > 0

            self.test_connections.append(connection)

            # Dispose it
            cm.midi_thru_connection_dispose(connection)
            self.test_connections.remove(connection)

        except RuntimeError as e:
            pytest.skip(f"Persistent thru connection creation failed (expected): {e}")

    def test_midi_thru_connection_create_with_params(self):
        """Test creating a thru connection with custom parameters"""
        try:
            # Get default parameters and modify them
            params = cm.midi_thru_connection_params_initialize()

            # Modify some filter settings
            params['filterOutSysEx'] = 1
            params['filterOutMTC'] = 1
            params['filterOutBeatClock'] = 1

            # Create connection with custom parameters
            connection = cm.midi_thru_connection_create(
                persistent_owner_id=self.test_owner_id,
                connection_params=params
            )

            assert isinstance(connection, int)
            assert connection > 0

            self.test_connections.append(connection)

            # Test getting parameters back
            retrieved_params = cm.midi_thru_connection_get_params(connection)

            # Check that our filter settings were applied
            assert retrieved_params['filterOutSysEx'] == 1
            assert retrieved_params['filterOutMTC'] == 1
            assert retrieved_params['filterOutBeatClock'] == 1

            # Dispose
            cm.midi_thru_connection_dispose(connection)
            self.test_connections.remove(connection)

        except RuntimeError as e:
            pytest.skip(f"Thru connection with custom params failed (expected): {e}")

    def test_midi_thru_connection_get_params_invalid(self):
        """Test getting parameters from invalid connection"""
        invalid_connection = 999999

        try:
            cm.midi_thru_connection_get_params(invalid_connection)
            assert False, "Expected RuntimeError for invalid connection"
        except RuntimeError as e:
            assert "failed with status" in str(e)

    def test_midi_thru_connection_set_params_invalid(self):
        """Test setting parameters on invalid connection"""
        invalid_connection = 999999
        params = cm.midi_thru_connection_params_initialize()

        try:
            cm.midi_thru_connection_set_params(invalid_connection, params)
            assert False, "Expected RuntimeError for invalid connection"
        except RuntimeError as e:
            assert "failed with status" in str(e)

    def test_midi_thru_connection_dispose_invalid(self):
        """Test disposing invalid connection"""
        invalid_connection = 999999

        try:
            cm.midi_thru_connection_dispose(invalid_connection)
            assert False, "Expected RuntimeError for invalid connection"
        except RuntimeError as e:
            assert "failed with status" in str(e)

    def test_midi_thru_connection_find_empty(self):
        """Test finding connections with non-existent owner"""
        try:
            connections = cm.midi_thru_connection_find("com.nonexistent.owner")
            assert isinstance(connections, list)
            # Should return empty list for non-existent owner
            assert len(connections) == 0

        except RuntimeError as e:
            # This might fail in some environments
            pytest.skip(f"Connection find failed: {e}")

    def test_midi_thru_connection_find_existing(self):
        """Test finding connections after creating them"""
        try:
            # Create a persistent connection
            connection = cm.midi_thru_connection_create(self.test_owner_id)
            self.test_connections.append(connection)

            # Find connections for our owner
            connections = cm.midi_thru_connection_find(self.test_owner_id)

            assert isinstance(connections, list)
            assert len(connections) >= 1
            assert connection in connections

            # Clean up
            cm.midi_thru_connection_dispose(connection)
            self.test_connections.remove(connection)

        except RuntimeError as e:
            pytest.skip(f"Connection find test failed: {e}")

    def test_midi_thru_connection_constants(self):
        """Test thru connection constants"""
        # Transform type constants
        assert cm.get_midi_transform_none() == 0
        assert cm.get_midi_transform_filter_out() == 1
        assert cm.get_midi_transform_map_control() == 2
        assert cm.get_midi_transform_add() == 8
        assert cm.get_midi_transform_scale() == 9
        assert cm.get_midi_transform_min_value() == 10
        assert cm.get_midi_transform_max_value() == 11
        assert cm.get_midi_transform_map_value() == 12

        # Control type constants
        assert cm.get_midi_control_type_7bit() == 0
        assert cm.get_midi_control_type_14bit() == 1
        assert cm.get_midi_control_type_7bit_rpn() == 2
        assert cm.get_midi_control_type_14bit_rpn() == 3
        assert cm.get_midi_control_type_7bit_nrpn() == 4
        assert cm.get_midi_control_type_14bit_nrpn() == 5

        # Max endpoints constant
        assert cm.get_midi_thru_connection_max_endpoints() == 8

    def test_midi_thru_connection_function_existence(self):
        """Test that all MIDIThruConnection functions exist and are callable"""
        # Test that all the wrapped functions exist
        assert hasattr(cm, 'midi_thru_connection_params_initialize')
        assert callable(cm.midi_thru_connection_params_initialize)

        assert hasattr(cm, 'midi_thru_connection_create')
        assert callable(cm.midi_thru_connection_create)

        assert hasattr(cm, 'midi_thru_connection_dispose')
        assert callable(cm.midi_thru_connection_dispose)

        assert hasattr(cm, 'midi_thru_connection_get_params')
        assert callable(cm.midi_thru_connection_get_params)

        assert hasattr(cm, 'midi_thru_connection_set_params')
        assert callable(cm.midi_thru_connection_set_params)

        assert hasattr(cm, 'midi_thru_connection_find')
        assert callable(cm.midi_thru_connection_find)

        # Constant functions
        assert hasattr(cm, 'get_midi_transform_none')
        assert callable(cm.get_midi_transform_none)

        assert hasattr(cm, 'get_midi_control_type_7bit')
        assert callable(cm.get_midi_control_type_7bit)

    def test_midi_thru_connection_parameter_validation(self):
        """Test parameter validation for MIDIThruConnection functions"""
        # Test parameter type validation

        # Integer parameters should reject strings
        with pytest.raises((TypeError, ValueError)):
            cm.midi_thru_connection_dispose("not_a_connection_ref")

        with pytest.raises((TypeError, ValueError)):
            cm.midi_thru_connection_get_params("not_a_connection_ref")

        # Dictionary parameters should be validated
        with pytest.raises((TypeError, AttributeError)):
            cm.midi_thru_connection_set_params(123456, "not_a_dict")

        # String parameters should reject None where required
        with pytest.raises((TypeError, AttributeError)):
            cm.midi_thru_connection_find(None)

    def test_midi_thru_connection_params_structure(self):
        """Test the structure of thru connection parameters"""
        params = cm.midi_thru_connection_params_initialize()

        # Test that we can modify parameters
        params['filterOutSysEx'] = 1
        params['lowVelocity'] = 10
        params['highVelocity'] = 120
        params['lowNote'] = 21  # A0
        params['highNote'] = 108  # C8

        # Test channel mapping modification
        params['channelMap'][0] = 1  # Route channel 1 to channel 2
        params['channelMap'][15] = 0xFF  # Filter out channel 16

        # Test transform modification
        params['velocity']['transform'] = cm.get_midi_transform_add()
        params['velocity']['param'] = 10

        # These modifications should not cause errors
        assert params['filterOutSysEx'] == 1
        assert params['lowVelocity'] == 10
        assert params['channelMap'][0] == 1
        assert params['velocity']['transform'] == cm.get_midi_transform_add()

    def test_midi_thru_connection_with_endpoints(self):
        """Test thru connection with actual endpoints (if available)"""
        # Check if we have any MIDI sources or destinations
        num_sources = cm.midi_get_number_of_sources()
        num_destinations = cm.midi_get_number_of_destinations()

        if num_sources == 0 and num_destinations == 0:
            pytest.skip("No MIDI endpoints available for endpoint testing")

        try:
            params = cm.midi_thru_connection_params_initialize()

            # Add sources if available
            if num_sources > 0:
                source = cm.midi_get_source(0)
                params['sources'] = [{'endpointRef': source, 'uniqueID': 0}]

            # Add destinations if available
            if num_destinations > 0:
                dest = cm.midi_get_destination(0)
                params['destinations'] = [{'endpointRef': dest, 'uniqueID': 0}]

            # Try to create connection with real endpoints
            connection = cm.midi_thru_connection_create(
                persistent_owner_id=self.test_owner_id,
                connection_params=params
            )

            assert isinstance(connection, int)
            assert connection > 0

            self.test_connections.append(connection)

            # Test getting parameters back
            retrieved_params = cm.midi_thru_connection_get_params(connection)

            # Check that endpoints were preserved
            if num_sources > 0:
                assert len(retrieved_params['sources']) >= 1
                assert retrieved_params['sources'][0]['endpointRef'] == source

            if num_destinations > 0:
                assert len(retrieved_params['destinations']) >= 1
                assert retrieved_params['destinations'][0]['endpointRef'] == dest

            # Clean up
            cm.midi_thru_connection_dispose(connection)
            self.test_connections.remove(connection)

        except RuntimeError as e:
            pytest.skip(f"Thru connection with endpoints failed: {e}")

    def test_midi_thru_connection_integration_workflow(self):
        """Test a complete workflow of thru connection management"""
        try:
            # Step 1: Initialize parameters
            params = cm.midi_thru_connection_params_initialize()

            # Step 2: Customize parameters
            params['filterOutSysEx'] = 1
            params['filterOutMTC'] = 1
            params['lowVelocity'] = 1  # Filter out very low velocities

            # Step 3: Create connection
            connection = cm.midi_thru_connection_create(
                persistent_owner_id=self.test_owner_id,
                connection_params=params
            )

            self.test_connections.append(connection)

            # Step 4: Verify parameters
            retrieved_params = cm.midi_thru_connection_get_params(connection)
            assert retrieved_params['filterOutSysEx'] == 1
            assert retrieved_params['filterOutMTC'] == 1
            assert retrieved_params['lowVelocity'] == 1

            # Step 5: Modify parameters
            retrieved_params['filterOutBeatClock'] = 1
            cm.midi_thru_connection_set_params(connection, retrieved_params)

            # Step 6: Verify modification
            final_params = cm.midi_thru_connection_get_params(connection)
            assert final_params['filterOutBeatClock'] == 1

            # Step 7: Find the connection
            found_connections = cm.midi_thru_connection_find(self.test_owner_id)
            assert connection in found_connections

            # Step 8: Clean up
            cm.midi_thru_connection_dispose(connection)
            self.test_connections.remove(connection)

        except RuntimeError as e:
            pytest.skip(f"Thru connection integration test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])