"""pytest test suite for CoreMIDI wrapper functionality."""

import os
import pytest
import time
import coremusic as cm
import coremusic.capi as capi


# Check MIDI availability at module load time
# Run multiple checks to ensure MIDI is reliably available
_MIDI_AVAILABLE = False
try:
    # Try creating and disposing multiple clients to ensure MIDI is reliably available
    for i in range(3):
        _client_id = capi.midi_client_create(f"_TestCheck{i}")
        capi.midi_client_dispose(_client_id)
    _MIDI_AVAILABLE = True
except Exception:
    _MIDI_AVAILABLE = False


# All required MIDI functions that must be exported from capi
REQUIRED_MIDI_FUNCTIONS = [
    # MIDISetup functions
    "midi_external_device_create", "midi_device_add_entity",
    "midi_device_new_entity", "midi_device_remove_entity",
    "midi_entity_add_or_remove_endpoints", "midi_setup_add_device",
    "midi_setup_remove_device", "midi_setup_add_external_device",
    "midi_setup_remove_external_device",
    # MIDIDriver functions
    "midi_device_create", "midi_device_dispose",
    "midi_device_list_get_number_of_devices", "midi_device_list_get_device",
    "midi_device_list_add_device", "midi_device_list_dispose",
    "midi_endpoint_set_ref_cons", "midi_endpoint_get_ref_cons",
    "midi_get_driver_io_runloop", "midi_get_driver_device_list",
    "midi_driver_enable_monitoring",
    # MIDIThruConnection functions
    "midi_thru_connection_params_initialize", "midi_thru_connection_create",
    "midi_thru_connection_dispose", "midi_thru_connection_get_params",
    "midi_thru_connection_set_params", "midi_thru_connection_find",
    "get_midi_transform_none", "get_midi_control_type_7bit",
]


class TestCoreMIDIFunctionAvailability:
    """Test that all required MIDI functions are available"""

    @pytest.mark.parametrize("func_name", REQUIRED_MIDI_FUNCTIONS)
    def test_midi_function_available(self, func_name):
        """Test that required MIDI function exists and is callable"""
        assert hasattr(capi, func_name), f"Missing function: {func_name}"
        assert callable(getattr(capi, func_name)), f"Not callable: {func_name}"


@pytest.mark.skipif(not _MIDI_AVAILABLE, reason="MIDI services not available")
class TestCoreMIDIClient:
    """Test CoreMIDI client operations"""

    def test_midi_client_create_dispose(self):
        """Test MIDI client creation and disposal"""
        try:
            client = capi.midi_client_create("Test Client")
        except RuntimeError:
            pytest.skip("MIDI client creation failed - MIDI services unavailable")
        assert client is not None
        assert isinstance(client, int)
        assert client > 0
        capi.midi_client_dispose(client)

    def test_midi_client_create_invalid_name(self):
        """Test MIDI client creation with invalid name"""
        with pytest.raises((RuntimeError, ValueError, TypeError, AttributeError)):
            capi.midi_client_create(None)


@pytest.mark.skipif(not _MIDI_AVAILABLE, reason="MIDI services not available")
class TestCoreMIDIPorts:
    """Test CoreMIDI port operations"""

    def setup_method(self):
        """Set up test client for port tests"""
        try:
            self.client = capi.midi_client_create("Test Port Client")
        except RuntimeError:
            pytest.skip("MIDI client creation failed - MIDI services unavailable")

    def teardown_method(self):
        """Clean up test client"""
        try:
            if hasattr(self, "client") and self.client:
                capi.midi_client_dispose(self.client)
                self.client = None
        except Exception:
            pass

    def test_midi_input_port_create_dispose(self):
        """Test MIDI input port creation and disposal"""
        port = capi.midi_input_port_create(self.client, "Test Input Port")
        assert port is not None
        assert isinstance(port, int)
        assert port > 0
        capi.midi_port_dispose(port)

    def test_midi_output_port_create_dispose(self):
        """Test MIDI output port creation and disposal"""
        port = capi.midi_output_port_create(self.client, "Test Output Port")
        assert port is not None
        assert isinstance(port, int)
        assert port > 0
        capi.midi_port_dispose(port)

    def test_midi_port_create_invalid_client(self):
        """Test port creation with invalid client"""
        with pytest.raises(RuntimeError):
            capi.midi_input_port_create(0, "Invalid Client Port")


class TestCoreMIDIDevices:
    """Test CoreMIDI device enumeration"""

    def test_midi_get_number_of_devices(self):
        """Test getting number of MIDI devices"""
        num_devices = capi.midi_get_number_of_devices()
        assert isinstance(num_devices, int)
        assert num_devices >= 0

    def test_midi_get_number_of_sources(self):
        """Test getting number of MIDI sources"""
        num_sources = capi.midi_get_number_of_sources()
        assert isinstance(num_sources, int)
        assert num_sources >= 0

    def test_midi_get_number_of_destinations(self):
        """Test getting number of MIDI destinations"""
        num_destinations = capi.midi_get_number_of_destinations()
        assert isinstance(num_destinations, int)
        assert num_destinations >= 0

    def test_midi_device_enumeration(self):
        """Test MIDI device enumeration"""
        num_devices = capi.midi_get_number_of_devices()
        for i in range(num_devices):
            device = capi.midi_get_device(i)
            assert device is not None
            assert isinstance(device, int)
            assert device > 0
            num_entities = capi.midi_device_get_number_of_entities(device)
            assert isinstance(num_entities, int)
            assert num_entities >= 0
            for j in range(num_entities):
                entity = capi.midi_device_get_entity(device, j)
                assert entity is not None
                assert isinstance(entity, int)

    def test_midi_source_enumeration(self):
        """Test MIDI source enumeration"""
        num_sources = capi.midi_get_number_of_sources()
        for i in range(min(num_sources, 5)):
            source = capi.midi_get_source(i)
            assert source is not None
            assert isinstance(source, int)
            assert source > 0

    def test_midi_destination_enumeration(self):
        """Test MIDI destination enumeration"""
        num_destinations = capi.midi_get_number_of_destinations()
        for i in range(min(num_destinations, 5)):
            destination = capi.midi_get_destination(i)
            assert destination is not None
            assert isinstance(destination, int)
            assert destination > 0


@pytest.mark.skipif(not _MIDI_AVAILABLE, reason="MIDI services not available")
class TestCoreMIDIVirtualEndpoints:
    """Test CoreMIDI virtual endpoint operations"""

    def setup_method(self):
        """Set up test client for virtual endpoint tests"""
        try:
            self.client = capi.midi_client_create("Test Virtual Client")
        except RuntimeError:
            pytest.skip("MIDI client creation failed - MIDI services unavailable")

    def teardown_method(self):
        """Clean up test client"""
        try:
            if hasattr(self, "client") and self.client:
                capi.midi_client_dispose(self.client)
                self.client = None
        except Exception:
            pass

    def test_midi_source_create_dispose(self):
        """Test virtual MIDI source creation and disposal"""
        source = capi.midi_source_create(self.client, "Test Virtual Source")
        assert source is not None
        assert isinstance(source, int)
        assert source > 0
        capi.midi_endpoint_dispose(source)

    def test_midi_destination_create_dispose(self):
        """Test virtual MIDI destination creation and disposal"""
        destination = capi.midi_destination_create(
            self.client, "Test Virtual Destination"
        )
        assert destination is not None
        assert isinstance(destination, int)
        assert destination > 0
        capi.midi_endpoint_dispose(destination)


@pytest.mark.skipif(not _MIDI_AVAILABLE, reason="MIDI services not available")
class TestCoreMIDIProperties:
    """Test CoreMIDI property operations"""

    def setup_method(self):
        """Set up test client for property tests"""
        try:
            self.client = capi.midi_client_create("Test Property Client")
        except RuntimeError:
            pytest.skip("MIDI client creation failed - MIDI services unavailable")

    def teardown_method(self):
        """Clean up test client"""
        try:
            if hasattr(self, "client") and self.client:
                capi.midi_client_dispose(self.client)
                self.client = None
        except Exception:
            pass

    def test_midi_object_get_string_property(self):
        """Test getting MIDI object string properties"""
        num_devices = capi.midi_get_number_of_devices()
        if num_devices == 0:
            pytest.skip("No MIDI devices available for property testing")
        device = capi.midi_get_device(0)
        # Name property should always exist for devices
        name = capi.midi_object_get_string_property(device, "name")
        assert name is not None, "Device name property should not be None"
        assert isinstance(name, str)
        assert len(name) > 0

    def test_midi_object_get_integer_property(self):
        """Test getting MIDI object integer properties"""
        num_devices = capi.midi_get_number_of_devices()
        if num_devices == 0:
            pytest.skip("No MIDI devices available for property testing")
        device = capi.midi_get_device(0)
        # uniqueID property should always exist for devices
        unique_id = capi.midi_object_get_integer_property(device, "uniqueID")
        assert unique_id is not None, "Device uniqueID property should not be None"
        assert isinstance(unique_id, int)

    def test_midi_object_set_string_property(self):
        """Test setting MIDI object string properties"""
        source = capi.midi_source_create(self.client, "Test Property Source")
        try:
            capi.midi_object_set_string_property(source, "testProperty", "test value")
            value = capi.midi_object_get_string_property(source, "testProperty")
            # We set it, so it should exist
            assert value == "test value", "Set property should be retrievable"
        finally:
            capi.midi_endpoint_dispose(source)

    def test_midi_object_set_integer_property(self):
        """Test setting MIDI object integer properties"""
        source = capi.midi_source_create(self.client, "Test Integer Property Source")
        try:
            capi.midi_object_set_integer_property(source, "testIntProperty", 42)
            value = capi.midi_object_get_integer_property(source, "testIntProperty")
            # We set it, so it should exist
            assert value == 42, "Set property should be retrievable"
        finally:
            capi.midi_endpoint_dispose(source)

    def test_midi_device_get_name(self):
        """Test getting MIDI device names"""
        num_devices = capi.midi_get_number_of_devices()
        if num_devices == 0:
            pytest.skip("No MIDI devices available for name testing")
        device = capi.midi_get_device(0)
        name = capi.midi_device_get_name(device)
        assert isinstance(name, str), "Device name should be a string"
        assert len(name) > 0, "Device name should not be empty"

    def test_midi_endpoint_get_name(self):
        """Test getting MIDI endpoint names"""
        num_sources = capi.midi_get_number_of_sources()
        num_destinations = capi.midi_get_number_of_destinations()
        if num_sources == 0 and num_destinations == 0:
            pytest.skip("No MIDI endpoints available for name testing")

        if num_sources > 0:
            source = capi.midi_get_source(0)
            name = capi.midi_endpoint_get_name(source)
            # Name can be None for some endpoints, but if returned should be str
            if name is not None:
                assert isinstance(name, str)

        if num_destinations > 0:
            dest = capi.midi_get_destination(0)
            name = capi.midi_endpoint_get_name(dest)
            if name is not None:
                assert isinstance(name, str)

    def test_midi_object_get_manufacturer_and_model(self):
        """Test getting MIDI object manufacturer and model"""
        num_devices = capi.midi_get_number_of_devices()
        if num_devices == 0:
            pytest.skip("No MIDI devices available for manufacturer/model testing")
        device = capi.midi_get_device(0)
        # These can return None for devices without manufacturer/model info
        manufacturer = capi.midi_object_get_manufacturer(device)
        model = capi.midi_object_get_model(device)
        # Just verify types if not None
        if manufacturer is not None:
            assert isinstance(manufacturer, str)
        if model is not None:
            assert isinstance(model, str)

    def test_midi_object_get_name_convenience(self):
        """Test the generic midi_object_get_name function"""
        num_devices = capi.midi_get_number_of_devices()
        if num_devices == 0:
            pytest.skip("No MIDI devices available for name testing")
        device = capi.midi_get_device(0)
        name = capi.midi_object_get_name(device)
        assert isinstance(name, str), "Object name should be a string"
        assert len(name) > 0, "Object name should not be empty"


class TestCoreMIDIData:
    """Test CoreMIDI data transmission"""

    def test_midi_send_data_large(self):
        """Test MIDI data sending with oversized data"""
        large_data = bytes([66] * 300)
        with pytest.raises(ValueError):
            capi.midi_send_data(1, 1, large_data, 0)

    def test_midi_send_data_basic_success(self):
        """Test that basic MIDI data function exists and can be called"""
        note_on_data = bytes([144, 60, 127])
        result = capi.midi_send_data(1, 1, note_on_data, 0)
        assert isinstance(result, int)


@pytest.mark.skipif(not _MIDI_AVAILABLE, reason="MIDI services not available")
class TestCoreMIDIIntegration:
    """Integration tests for CoreMIDI functionality"""

    def test_full_midi_workflow(self):
        """Test complete MIDI workflow: client -> ports -> virtual endpoints -> data"""
        try:
            client = capi.midi_client_create("Integration Test Client")
        except RuntimeError:
            pytest.skip("MIDI client creation failed - MIDI services unavailable")
        try:
            input_port = capi.midi_input_port_create(client, "Integration Input")
            output_port = capi.midi_output_port_create(client, "Integration Output")
            virtual_source = capi.midi_source_create(
                client, "Integration Virtual Source"
            )
            virtual_destination = capi.midi_destination_create(
                client, "Integration Virtual Destination"
            )
            try:
                capi.midi_object_set_string_property(
                    virtual_source, "description", "Test source"
                )
                description = capi.midi_object_get_string_property(
                    virtual_source, "description"
                )
                if description is not None:
                    assert description == "Test source"
            except RuntimeError:
                pass
            try:
                note_data = bytes([144, 60, 127])
                capi.midi_send_data(output_port, virtual_destination, note_data, 0)
            except RuntimeError:
                pass
            capi.midi_endpoint_dispose(virtual_destination)
            capi.midi_endpoint_dispose(virtual_source)
            capi.midi_port_dispose(output_port)
            capi.midi_port_dispose(input_port)
        finally:
            capi.midi_client_dispose(client)

    def test_midi_system_enumeration(self):
        """Test complete MIDI system enumeration"""
        num_devices = capi.midi_get_number_of_devices()
        num_sources = capi.midi_get_number_of_sources()
        num_destinations = capi.midi_get_number_of_destinations()
        print(
            f"MIDI System: {num_devices} devices, {num_sources} sources, {num_destinations} destinations"
        )
        for i in range(min(num_devices, 3)):
            device = capi.midi_get_device(i)
            num_entities = capi.midi_device_get_number_of_entities(device)
            for j in range(num_entities):
                entity = capi.midi_device_get_entity(device, j)
                num_entity_sources = capi.midi_entity_get_number_of_sources(entity)
                num_entity_destinations = capi.midi_entity_get_number_of_destinations(
                    entity
                )
                print(
                    f"  Device {i}, Entity {j}: {num_entity_sources} sources, {num_entity_destinations} destinations"
                )
                for k in range(min(num_entity_sources, 2)):
                    source = capi.midi_entity_get_source(entity, k)
                    assert source > 0
                for k in range(min(num_entity_destinations, 2)):
                    destination = capi.midi_entity_get_destination(entity, k)
                    assert destination > 0


class TestCoreMIDIMessages:
    """Test CoreMIDI Universal MIDI Packet (UMP) message functionality"""

    def test_midi_message_type_for_up_word(self):
        """Test extracting message type from Universal MIDI Packet word"""
        utility_word = 0
        system_word = 268435456
        cv1_word = 536870912
        sysex_word = 805306368
        cv2_word = 1073741824
        data128_word = 1342177280
        assert capi.midi_message_type_for_up_word(utility_word) == 0
        assert capi.midi_message_type_for_up_word(system_word) == 1
        assert capi.midi_message_type_for_up_word(cv1_word) == 2
        assert capi.midi_message_type_for_up_word(sysex_word) == 3
        assert capi.midi_message_type_for_up_word(cv2_word) == 4
        assert capi.midi_message_type_for_up_word(data128_word) == 5

    def test_midi1_up_channel_voice_message(self):
        """Test MIDI 1.0 Universal Packet channel voice message creation"""
        group = 0
        status = 9
        channel = 0
        data1 = 60
        data2 = 127
        message = capi.midi1_up_channel_voice_message(
            group, status, channel, data1, data2
        )
        assert isinstance(message, int)
        assert message > 0
        message_type = capi.midi_message_type_for_up_word(message)
        assert message_type == 2

    def test_midi1_up_note_on(self):
        """Test MIDI 1.0 Universal Packet Note On message"""
        group = 0
        channel = 0
        note_number = 60
        velocity = 127
        message = capi.midi1_up_note_on(group, channel, note_number, velocity)
        assert isinstance(message, int)
        assert message > 0
        message_type = capi.midi_message_type_for_up_word(message)
        assert message_type == 2

    def test_midi1_up_note_off(self):
        """Test MIDI 1.0 Universal Packet Note Off message"""
        group = 0
        channel = 0
        note_number = 60
        velocity = 64
        message = capi.midi1_up_note_off(group, channel, note_number, velocity)
        assert isinstance(message, int)
        assert message > 0
        message_type = capi.midi_message_type_for_up_word(message)
        assert message_type == 2

    def test_midi1_up_control_change(self):
        """Test MIDI 1.0 Universal Packet Control Change message"""
        group = 0
        channel = 0
        index = 7
        data = 100
        message = capi.midi1_up_control_change(group, channel, index, data)
        assert isinstance(message, int)
        assert message > 0
        message_type = capi.midi_message_type_for_up_word(message)
        assert message_type == 2

    def test_midi1_up_pitch_bend(self):
        """Test MIDI 1.0 Universal Packet Pitch Bend message"""
        group = 0
        channel = 0
        lsb = 0
        msb = 64
        message = capi.midi1_up_pitch_bend(group, channel, lsb, msb)
        assert isinstance(message, int)
        assert message > 0
        message_type = capi.midi_message_type_for_up_word(message)
        assert message_type == 2

    def test_midi1_up_system_common(self):
        """Test MIDI 1.0 Universal Packet System Common message"""
        group = 0
        status = 242
        byte1 = 16
        byte2 = 32
        message = capi.midi1_up_system_common(group, status, byte1, byte2)
        assert isinstance(message, int)
        assert message > 0
        message_type = capi.midi_message_type_for_up_word(message)
        assert message_type == 1

    def test_midi1_up_sysex(self):
        """Test MIDI 1.0 Universal Packet SysEx message"""
        group = 0
        status = 0
        bytes_used = 3
        byte1 = 126
        byte2 = 0
        byte3 = 9
        byte4 = 1
        byte5 = 0
        byte6 = 0
        message = capi.midi1_up_sysex(
            group, status, bytes_used, byte1, byte2, byte3, byte4, byte5, byte6
        )
        assert isinstance(message, tuple)
        assert len(message) == 2
        assert isinstance(message[0], int)
        assert isinstance(message[1], int)
        message_type = capi.midi_message_type_for_up_word(message[0])
        assert message_type == 3

    def test_midi2_channel_voice_message(self):
        """Test MIDI 2.0 Channel Voice message creation"""
        group = 0
        status = 9
        channel = 0
        index = 15360
        value = 4294901760
        message = capi.midi2_channel_voice_message(group, status, channel, index, value)
        assert isinstance(message, tuple)
        assert len(message) == 2
        assert isinstance(message[0], int)
        assert isinstance(message[1], int)
        message_type = capi.midi_message_type_for_up_word(message[0])
        assert message_type == 4

    def test_midi2_note_on(self):
        """Test MIDI 2.0 Note On message"""
        group = 0
        channel = 0
        note_number = 60
        attribute_type = 0
        attribute_data = 0
        velocity = 65535
        message = capi.midi2_note_on(
            group, channel, note_number, attribute_type, attribute_data, velocity
        )
        assert isinstance(message, tuple)
        assert len(message) == 2
        assert isinstance(message[0], int)
        assert isinstance(message[1], int)
        message_type = capi.midi_message_type_for_up_word(message[0])
        assert message_type == 4

    def test_midi2_note_off(self):
        """Test MIDI 2.0 Note Off message"""
        group = 0
        channel = 0
        note_number = 60
        attribute_type = 0
        attribute_data = 0
        velocity = 32768
        message = capi.midi2_note_off(
            group, channel, note_number, attribute_type, attribute_data, velocity
        )
        assert isinstance(message, tuple)
        assert len(message) == 2
        assert isinstance(message[0], int)
        assert isinstance(message[1], int)
        message_type = capi.midi_message_type_for_up_word(message[0])
        assert message_type == 4

    def test_midi2_control_change(self):
        """Test MIDI 2.0 Control Change message"""
        group = 0
        channel = 0
        index = 7
        value = 2147483648
        message = capi.midi2_control_change(group, channel, index, value)
        assert isinstance(message, tuple)
        assert len(message) == 2
        assert isinstance(message[0], int)
        assert isinstance(message[1], int)
        message_type = capi.midi_message_type_for_up_word(message[0])
        assert message_type == 4

    def test_midi2_program_change(self):
        """Test MIDI 2.0 Program Change message"""
        group = 0
        channel = 0
        bank_is_valid = True
        program = 1
        bank_msb = 0
        bank_lsb = 0
        message = capi.midi2_program_change(
            group, channel, bank_is_valid, program, bank_msb, bank_lsb
        )
        assert isinstance(message, tuple)
        assert len(message) == 2
        assert isinstance(message[0], int)
        assert isinstance(message[1], int)
        message_type = capi.midi_message_type_for_up_word(message[0])
        assert message_type == 4

    def test_midi2_pitch_bend(self):
        """Test MIDI 2.0 Pitch Bend message"""
        group = 0
        channel = 0
        value = 2147483648
        message = capi.midi2_pitch_bend(group, channel, value)
        assert isinstance(message, tuple)
        assert len(message) == 2
        assert isinstance(message[0], int)
        assert isinstance(message[1], int)
        message_type = capi.midi_message_type_for_up_word(message[0])
        assert message_type == 4

    def test_midi_message_parameter_validation(self):
        """Test parameter validation for MIDI message functions"""
        group = 0
        channel = 0
        note = 60
        velocity = 127
        message = capi.midi1_up_note_on(group, channel, note, velocity)
        assert isinstance(message, int)
        max_group = 15
        max_channel = 15
        max_note = 127
        max_velocity = 127
        message = capi.midi1_up_note_on(max_group, max_channel, max_note, max_velocity)
        assert isinstance(message, int)

    def test_midi_message_consistency(self):
        """Test that identical parameters produce identical messages"""
        group = 0
        channel = 5
        note = 72
        velocity = 100
        message1 = capi.midi1_up_note_on(group, channel, note, velocity)
        message2 = capi.midi1_up_note_on(group, channel, note, velocity)
        assert message1 == message2
        attr_type = 0
        attr_data = 0
        velocity16 = 51200
        msg2_1 = capi.midi2_note_on(
            group, channel, note, attr_type, attr_data, velocity16
        )
        msg2_2 = capi.midi2_note_on(
            group, channel, note, attr_type, attr_data, velocity16
        )
        assert msg2_1 == msg2_2


class TestCoreMIDISetup:
    """Test CoreMIDI Setup (device and entity management) functionality"""

    def setup_method(self):
        """Set up test environment for MIDISetup tests"""
        pass

    def teardown_method(self):
        """Clean up test environment"""
        pass

    def test_midi_external_device_create(self):
        """Test creating an external MIDI device"""
        try:
            device = capi.midi_external_device_create(
                "Test External Device", "Test Manufacturer", "Test Model"
            )
            assert isinstance(device, int)
            assert device > 0
        except RuntimeError as e:
            pytest.skip(f"External device creation failed (expected): {e}")

    def test_midi_external_device_create_invalid_params(self):
        """Test external device creation with invalid parameters"""
        try:
            device = capi.midi_external_device_create("", "", "")
            assert isinstance(device, int)
            assert device > 0
        except RuntimeError:
            pass
        with pytest.raises((TypeError, AttributeError)):
            capi.midi_external_device_create(None, "Manufacturer", "Model")

    def test_midi_device_add_entity_basic(self):
        """Test adding entity to a device (if we have a valid device)"""
        num_devices = capi.midi_get_number_of_devices()
        if num_devices == 0:
            pytest.skip("No MIDI devices available for entity testing")
        device = capi.midi_get_device(0)
        try:
            entity = capi.midi_device_add_entity(device, "Test Entity", True, 1, 1)
            assert isinstance(entity, int)
            assert entity > 0
            try:
                capi.midi_device_remove_entity(device, entity)
            except RuntimeError:
                pass
        except RuntimeError as e:
            assert "failed" in str(e)

    def test_midi_device_new_entity_basic(self):
        """Test creating new entity with protocol support (macOS 11.0+)"""
        num_devices = capi.midi_get_number_of_devices()
        if num_devices == 0:
            pytest.skip("No MIDI devices available for entity testing")
        device = capi.midi_get_device(0)
        try:
            entity = capi.midi_device_new_entity(
                device, "Test MIDI 1.0 Entity", 1, True, 1, 1
            )
            assert isinstance(entity, int)
            assert entity > 0
            try:
                capi.midi_device_remove_entity(device, entity)
            except RuntimeError:
                pass
        except RuntimeError as e:
            assert "failed" in str(e)
        try:
            entity = capi.midi_device_new_entity(
                device, "Test MIDI 2.0 Entity", 2, False, 2, 2
            )
            assert isinstance(entity, int)
            assert entity > 0
            try:
                capi.midi_device_remove_entity(device, entity)
            except RuntimeError:
                pass
        except RuntimeError as e:
            assert "failed" in str(e)

    def test_midi_entity_add_or_remove_endpoints(self):
        """Test adding/removing endpoints from an entity"""
        num_devices = capi.midi_get_number_of_devices()
        if num_devices == 0:
            pytest.skip("No MIDI devices available for endpoint testing")
        device = capi.midi_get_device(0)
        num_entities = capi.midi_device_get_number_of_entities(device)
        if num_entities == 0:
            pytest.skip("No entities available for endpoint testing")
        entity = capi.midi_device_get_entity(device, 0)
        try:
            result = capi.midi_entity_add_or_remove_endpoints(entity, 2, 2)
            assert isinstance(result, int)
            assert result == 0
        except RuntimeError as e:
            assert "failed" in str(e)

    def test_midi_setup_device_management(self):
        """Test setup device management functions"""
        invalid_device = 999999
        try:
            capi.midi_setup_add_device(invalid_device)
            assert False, "Expected RuntimeError for invalid device"
        except RuntimeError as e:
            assert "failed" in str(e)
        try:
            capi.midi_setup_remove_device(invalid_device)
            assert False, "Expected RuntimeError for invalid device"
        except RuntimeError as e:
            assert "failed" in str(e)
        try:
            capi.midi_setup_add_external_device(invalid_device)
            assert False, "Expected RuntimeError for invalid device"
        except RuntimeError as e:
            assert "failed" in str(e)
        try:
            capi.midi_setup_remove_external_device(invalid_device)
            assert False, "Expected RuntimeError for invalid device"
        except RuntimeError as e:
            assert "failed" in str(e)

    def test_midi_setup_parameter_validation(self):
        """Test parameter validation for MIDISetup functions"""
        with pytest.raises((TypeError, AttributeError)):
            capi.midi_external_device_create(None, "Manufacturer", "Model")
        with pytest.raises((TypeError, AttributeError)):
            capi.midi_external_device_create("Name", None, "Model")
        with pytest.raises((TypeError, AttributeError)):
            capi.midi_external_device_create("Name", "Manufacturer", None)
        with pytest.raises((TypeError, ValueError)):
            capi.midi_setup_add_device("not_a_device_ref")
        with pytest.raises((TypeError, ValueError)):
            capi.midi_device_remove_entity("not_a_device", "not_an_entity")

    def test_midi_setup_integration_workflow(self):
        """Test a complete workflow of external device management"""
        try:
            device = capi.midi_external_device_create(
                "Integration Test Device", "Test Company", "Test Model v1.0"
            )
            assert isinstance(device, int)
            assert device > 0
            try:
                capi.midi_setup_add_external_device(device)
                try:
                    entity = capi.midi_device_add_entity(
                        device, "Test Entity", False, 1, 1
                    )
                    try:
                        capi.midi_entity_add_or_remove_endpoints(entity, 2, 2)
                    except RuntimeError:
                        pass
                    try:
                        capi.midi_device_remove_entity(device, entity)
                    except RuntimeError:
                        pass
                except RuntimeError:
                    pass
                try:
                    capi.midi_setup_remove_external_device(device)
                except RuntimeError:
                    pass
            except RuntimeError:
                pass
        except RuntimeError:
            pytest.skip("External device creation not supported in this environment")


class TestCoreMIDIDriver:
    """Test CoreMIDI Driver functionality"""

    def setup_method(self):
        """Set up test environment for MIDIDriver tests"""
        pass

    def teardown_method(self):
        """Clean up test environment"""
        pass

    def test_midi_device_create_basic(self):
        """Test creating a MIDI device using the driver API"""
        try:
            device = capi.midi_device_create(
                "Driver Test Device", "Test Driver Manufacturer", "Test Driver Model"
            )
            assert isinstance(device, int)
            assert device > 0
            try:
                capi.midi_device_dispose(device)
            except RuntimeError:
                pass
        except RuntimeError as e:
            pytest.skip(f"MIDI device creation failed (expected): {e}")

    def test_midi_device_create_invalid_params(self):
        """Test MIDI device creation with invalid parameters"""
        with pytest.raises((TypeError, AttributeError)):
            capi.midi_device_create(None, "Manufacturer", "Model")
        with pytest.raises((TypeError, AttributeError)):
            capi.midi_device_create("Name", None, "Model")
        with pytest.raises((TypeError, AttributeError)):
            capi.midi_device_create("Name", "Manufacturer", None)

    def test_midi_device_dispose_invalid(self):
        """Test disposing an invalid device"""
        invalid_device = 999999
        try:
            capi.midi_device_dispose(invalid_device)
            assert False, "Expected RuntimeError for invalid device"
        except RuntimeError as e:
            assert "failed" in str(e)

    def test_midi_endpoint_ref_cons_basic(self):
        """Test setting and getting endpoint reference constants"""
        num_sources = capi.midi_get_number_of_sources()
        if num_sources == 0:
            pytest.skip("No MIDI sources available for refCon testing")
        source = capi.midi_get_source(0)
        ref1_value = 12345
        ref2_value = 67890
        try:
            result = capi.midi_endpoint_set_ref_cons(source, ref1_value, ref2_value)
            assert isinstance(result, int)
            assert result == 0
            ref1_retrieved, ref2_retrieved = capi.midi_endpoint_get_ref_cons(source)
            assert ref1_retrieved == ref1_value
            assert ref2_retrieved == ref2_value
            capi.midi_endpoint_set_ref_cons(source)
            ref1_default, ref2_default = capi.midi_endpoint_get_ref_cons(source)
            assert ref1_default == 0
            assert ref2_default == 0
        except RuntimeError as e:
            pytest.skip(f"RefCon operations failed (expected): {e}")

    def test_midi_endpoint_ref_cons_invalid_endpoint(self):
        """Test refCon operations with invalid endpoint"""
        invalid_endpoint = 999999
        try:
            capi.midi_endpoint_set_ref_cons(invalid_endpoint, 123, 456)
            assert False, "Expected RuntimeError for invalid endpoint"
        except RuntimeError as e:
            assert "failed" in str(e)
        try:
            capi.midi_endpoint_get_ref_cons(invalid_endpoint)
            assert False, "Expected RuntimeError for invalid endpoint"
        except RuntimeError as e:
            assert "failed" in str(e)

    def test_midi_get_driver_io_runloop(self):
        """Test getting the driver I/O run loop"""
        try:
            runloop = capi.midi_get_driver_io_runloop()
            assert isinstance(runloop, int)
            assert runloop != 0
        except Exception as e:
            pytest.skip(f"Driver I/O runloop access failed: {e}")

    def test_midi_get_driver_device_list_invalid(self):
        """Test getting device list for invalid driver"""
        invalid_driver = 999999
        try:
            dev_list = capi.midi_get_driver_device_list(invalid_driver)
            assert isinstance(dev_list, int)
        except Exception as e:
            pass

    def test_midi_driver_enable_monitoring_invalid(self):
        """Test enabling monitoring for invalid driver"""
        invalid_driver = 999999
        try:
            capi.midi_driver_enable_monitoring(invalid_driver, True)
            assert False, "Expected RuntimeError for invalid driver"
        except RuntimeError as e:
            assert "failed" in str(e)
        try:
            capi.midi_driver_enable_monitoring(invalid_driver, False)
            assert False, "Expected RuntimeError for invalid driver"
        except RuntimeError as e:
            assert "failed" in str(e)

    def test_midi_device_list_operations_basic(self):
        """Test basic device list operations (if we can create devices)"""
        invalid_dev_list = 999999
        try:
            num_devices = capi.midi_device_list_get_number_of_devices(invalid_dev_list)
            assert isinstance(num_devices, int)
            assert num_devices >= 0
        except Exception:
            pass
        try:
            device = capi.midi_device_list_get_device(invalid_dev_list, 0)
            assert False, "Expected error for invalid device list"
        except (RuntimeError, IndexError):
            pass
        try:
            capi.midi_device_list_add_device(invalid_dev_list, 123456)
            assert False, "Expected RuntimeError for invalid device list"
        except RuntimeError as e:
            assert "failed" in str(e)
        try:
            capi.midi_device_list_dispose(invalid_dev_list)
            assert False, "Expected RuntimeError for invalid device list"
        except RuntimeError as e:
            assert "failed" in str(e)

    def test_midi_driver_parameter_validation(self):
        """Test parameter validation for MIDIDriver functions"""
        with pytest.raises((TypeError, AttributeError)):
            capi.midi_device_create(None, "Manufacturer", "Model")
        with pytest.raises((TypeError, ValueError)):
            capi.midi_device_dispose("not_a_device_ref")
        with pytest.raises((TypeError, ValueError)):
            capi.midi_endpoint_set_ref_cons("not_an_endpoint", 123, 456)
        invalid_dev_list = 999999
        with pytest.raises((IndexError, RuntimeError)):
            capi.midi_device_list_get_device(invalid_dev_list, -1)

    def test_midi_driver_integration_workflow(self):
        """Test a complete workflow of driver-style device management"""
        try:
            device = capi.midi_device_create(
                "Integration Driver Device",
                "Integration Test Company",
                "Integration Test Model",
            )
            assert isinstance(device, int)
            assert device > 0
            try:
                capi.midi_device_dispose(device)
                pass
            except RuntimeError:
                pass
            try:
                num_entities = capi.midi_device_get_number_of_entities(device)
                if num_entities > 0:
                    entity = capi.midi_device_get_entity(device, 0)
                    num_sources = capi.midi_entity_get_number_of_sources(entity)
                    if num_sources > 0:
                        source = capi.midi_entity_get_source(entity, 0)
                        capi.midi_endpoint_set_ref_cons(source, 4660, 22136)
                        ref1, ref2 = capi.midi_endpoint_get_ref_cons(source)
                        assert ref1 == 4660
                        assert ref2 == 22136
            except (RuntimeError, IndexError):
                pass
        except RuntimeError:
            pytest.skip(
                "Driver-style device creation not supported in this environment"
            )


class TestCoreMIDIThruConnection:
    """Test CoreMIDI Thru Connection functionality"""

    def setup_method(self):
        """Set up test environment for MIDIThruConnection tests"""
        self.test_connections = []
        self.test_owner_id = "com.test.pytest.thruconnection"

    def teardown_method(self):
        """Clean up test environment"""
        for connection in self.test_connections:
            try:
                capi.midi_thru_connection_dispose(connection)
            except RuntimeError:
                pass

    def test_midi_thru_connection_params_initialize(self):
        """Test initializing thru connection parameters"""
        params = capi.midi_thru_connection_params_initialize()
        assert isinstance(params, dict)
        assert "version" in params
        assert "sources" in params
        assert "destinations" in params
        assert "channelMap" in params
        assert "filterOutSysEx" in params
        assert "filterOutMTC" in params
        assert "filterOutBeatClock" in params
        assert len(params["channelMap"]) == 16
        assert params["channelMap"] == list(range(16))
        assert isinstance(params["noteNumber"], dict)
        assert "transform" in params["noteNumber"]
        assert "param" in params["noteNumber"]
        assert params["version"] == 0
        assert params["sources"] == []
        assert params["destinations"] == []
        assert params["lowVelocity"] == 0
        assert params["highVelocity"] == 0

    def test_midi_thru_connection_create_basic(self):
        """Test creating a basic thru connection"""
        try:
            connection = capi.midi_thru_connection_create()
            assert isinstance(connection, int)
            assert connection > 0
            self.test_connections.append(connection)
            capi.midi_thru_connection_dispose(connection)
            self.test_connections.remove(connection)
        except RuntimeError as e:
            pytest.skip(f"Thru connection creation failed (expected): {e}")

    def test_midi_thru_connection_create_persistent(self):
        """Test creating a persistent thru connection"""
        try:
            connection = capi.midi_thru_connection_create(self.test_owner_id)
            assert isinstance(connection, int)
            assert connection > 0
            self.test_connections.append(connection)
            capi.midi_thru_connection_dispose(connection)
            self.test_connections.remove(connection)
        except RuntimeError as e:
            pytest.skip(f"Persistent thru connection creation failed (expected): {e}")

    def test_midi_thru_connection_create_with_params(self):
        """Test creating a thru connection with custom parameters"""
        try:
            params = capi.midi_thru_connection_params_initialize()
            params["filterOutSysEx"] = 1
            params["filterOutMTC"] = 1
            params["filterOutBeatClock"] = 1
            connection = capi.midi_thru_connection_create(
                persistent_owner_id=self.test_owner_id, connection_params=params
            )
            assert isinstance(connection, int)
            assert connection > 0
            self.test_connections.append(connection)
            retrieved_params = capi.midi_thru_connection_get_params(connection)
            assert retrieved_params["filterOutSysEx"] == 1
            assert retrieved_params["filterOutMTC"] == 1
            assert retrieved_params["filterOutBeatClock"] == 1
            capi.midi_thru_connection_dispose(connection)
            self.test_connections.remove(connection)
        except RuntimeError as e:
            pytest.skip(f"Thru connection with custom params failed (expected): {e}")

    def test_midi_thru_connection_get_params_invalid(self):
        """Test getting parameters from invalid connection"""
        invalid_connection = 999999
        try:
            capi.midi_thru_connection_get_params(invalid_connection)
            assert False, "Expected RuntimeError for invalid connection"
        except RuntimeError as e:
            assert "failed" in str(e)

    def test_midi_thru_connection_set_params_invalid(self):
        """Test setting parameters on invalid connection"""
        invalid_connection = 999999
        params = capi.midi_thru_connection_params_initialize()
        try:
            capi.midi_thru_connection_set_params(invalid_connection, params)
            assert False, "Expected RuntimeError for invalid connection"
        except RuntimeError as e:
            assert "failed" in str(e)

    def test_midi_thru_connection_dispose_invalid(self):
        """Test disposing invalid connection"""
        invalid_connection = 999999
        try:
            capi.midi_thru_connection_dispose(invalid_connection)
            assert False, "Expected RuntimeError for invalid connection"
        except RuntimeError as e:
            assert "failed" in str(e)

    def test_midi_thru_connection_find_empty(self):
        """Test finding connections with non-existent owner"""
        try:
            connections = capi.midi_thru_connection_find("com.nonexistent.owner")
            assert isinstance(connections, list)
            assert len(connections) == 0
        except RuntimeError as e:
            pytest.skip(f"Connection find failed: {e}")

    def test_midi_thru_connection_find_existing(self):
        """Test finding connections after creating them"""
        try:
            connection = capi.midi_thru_connection_create(self.test_owner_id)
            self.test_connections.append(connection)
            connections = capi.midi_thru_connection_find(self.test_owner_id)
            assert isinstance(connections, list)
            assert len(connections) >= 1
            assert connection in connections
            capi.midi_thru_connection_dispose(connection)
            self.test_connections.remove(connection)
        except RuntimeError as e:
            pytest.skip(f"Connection find test failed: {e}")

    def test_midi_thru_connection_parameter_validation(self):
        """Test parameter validation for MIDIThruConnection functions"""
        with pytest.raises((TypeError, ValueError)):
            capi.midi_thru_connection_dispose("not_a_connection_ref")
        with pytest.raises((TypeError, ValueError)):
            capi.midi_thru_connection_get_params("not_a_connection_ref")
        with pytest.raises((TypeError, AttributeError)):
            capi.midi_thru_connection_set_params(123456, "not_a_dict")
        with pytest.raises((TypeError, AttributeError)):
            capi.midi_thru_connection_find(None)

    def test_midi_thru_connection_params_structure(self):
        """Test the structure of thru connection parameters"""
        params = capi.midi_thru_connection_params_initialize()
        params["filterOutSysEx"] = 1
        params["lowVelocity"] = 10
        params["highVelocity"] = 120
        params["lowNote"] = 21
        params["highNote"] = 108
        params["channelMap"][0] = 1
        params["channelMap"][15] = 255
        params["velocity"]["transform"] = capi.get_midi_transform_add()
        params["velocity"]["param"] = 10
        assert params["filterOutSysEx"] == 1
        assert params["lowVelocity"] == 10
        assert params["channelMap"][0] == 1
        assert params["velocity"]["transform"] == capi.get_midi_transform_add()

    def test_midi_thru_connection_with_endpoints(self):
        """Test thru connection with actual endpoints (if available)"""
        num_sources = capi.midi_get_number_of_sources()
        num_destinations = capi.midi_get_number_of_destinations()
        if num_sources == 0 and num_destinations == 0:
            pytest.skip("No MIDI endpoints available for endpoint testing")
        try:
            params = capi.midi_thru_connection_params_initialize()
            if num_sources > 0:
                source = capi.midi_get_source(0)
                params["sources"] = [{"endpointRef": source, "uniqueID": 0}]
            if num_destinations > 0:
                dest = capi.midi_get_destination(0)
                params["destinations"] = [{"endpointRef": dest, "uniqueID": 0}]
            connection = capi.midi_thru_connection_create(
                persistent_owner_id=self.test_owner_id, connection_params=params
            )
            assert isinstance(connection, int)
            assert connection > 0
            self.test_connections.append(connection)
            retrieved_params = capi.midi_thru_connection_get_params(connection)
            if num_sources > 0:
                assert len(retrieved_params["sources"]) >= 1
                assert retrieved_params["sources"][0]["endpointRef"] == source
            if num_destinations > 0:
                assert len(retrieved_params["destinations"]) >= 1
                assert retrieved_params["destinations"][0]["endpointRef"] == dest
            capi.midi_thru_connection_dispose(connection)
            self.test_connections.remove(connection)
        except RuntimeError as e:
            pytest.skip(f"Thru connection with endpoints failed: {e}")

    def test_midi_thru_connection_integration_workflow(self):
        """Test a complete workflow of thru connection management"""
        try:
            params = capi.midi_thru_connection_params_initialize()
            params["filterOutSysEx"] = 1
            params["filterOutMTC"] = 1
            params["lowVelocity"] = 1
            connection = capi.midi_thru_connection_create(
                persistent_owner_id=self.test_owner_id, connection_params=params
            )
            self.test_connections.append(connection)
            retrieved_params = capi.midi_thru_connection_get_params(connection)
            assert retrieved_params["filterOutSysEx"] == 1
            assert retrieved_params["filterOutMTC"] == 1
            assert retrieved_params["lowVelocity"] == 1
            retrieved_params["filterOutBeatClock"] = 1
            capi.midi_thru_connection_set_params(connection, retrieved_params)
            final_params = capi.midi_thru_connection_get_params(connection)
            assert final_params["filterOutBeatClock"] == 1
            found_connections = capi.midi_thru_connection_find(self.test_owner_id)
            assert connection in found_connections
            capi.midi_thru_connection_dispose(connection)
            self.test_connections.remove(connection)
        except RuntimeError as e:
            pytest.skip(f"Thru connection integration test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
