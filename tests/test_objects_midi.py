"""Tests for MIDI object-oriented classes."""

import pytest
import time
import coremusic as cm
import coremusic.capi as capi


# Check MIDI availability at module load time
_MIDI_AVAILABLE = False
try:
    _client_id = capi.midi_client_create("_TestCheck")
    capi.midi_client_dispose(_client_id)
    _MIDI_AVAILABLE = True
except Exception:
    _MIDI_AVAILABLE = False


# Skip all MIDI tests if MIDI hardware/services not available
pytestmark = pytest.mark.skipif(
    not _MIDI_AVAILABLE,
    reason="MIDI services not available (requires macOS audio infrastructure)"
)


@pytest.fixture(autouse=True)
def skip_if_midi_fails():
    """Autouse fixture to skip tests if MIDI operations fail at runtime."""
    try:
        test_client = capi.midi_client_create("_RuntimeCheck")
        capi.midi_client_dispose(test_client)
    except Exception:
        pytest.skip("MIDI services unavailable at runtime")


class TestMIDIClient:
    """Test MIDIClient object-oriented wrapper"""

    def test_midi_client_creation(self):
        """Test MIDIClient creation"""
        client = cm.MIDIClient("Test Client")
        assert isinstance(client, cm.MIDIClient)
        assert isinstance(client, cm.CoreAudioObject)
        assert client.name == "Test Client"
        assert client.object_id != 0
        assert len(client._ports) == 0
        assert not client.is_disposed
        client.dispose()

    def test_midi_client_create_input_port(self):
        """Test MIDIClient input port creation"""
        client = cm.MIDIClient("Test Client")
        try:
            port = client.create_input_port("Test Input Port")
            assert isinstance(port, cm.MIDIInputPort)
            assert isinstance(port, cm.MIDIPort)
            assert isinstance(port, cm.CoreAudioObject)
            assert port.name == "Test Input Port"
            assert port.object_id != 0
            assert port._client is client
            assert len(client._ports) == 1
            assert client._ports[0] is port
        finally:
            client.dispose()

    def test_midi_client_create_output_port(self):
        """Test MIDIClient output port creation"""
        client = cm.MIDIClient("Test Client")
        try:
            port = client.create_output_port("Test Output Port")
            assert isinstance(port, cm.MIDIOutputPort)
            assert isinstance(port, cm.MIDIPort)
            assert port.name == "Test Output Port"
            assert port.object_id != 0
            assert port._client is client
            assert len(client._ports) == 1
            assert client._ports[0] is port
        finally:
            client.dispose()

    def test_midi_client_multiple_ports(self):
        """Test MIDIClient with multiple ports"""
        client = cm.MIDIClient("Multi Port Client")
        try:
            input_port1 = client.create_input_port("Input 1")
            input_port2 = client.create_input_port("Input 2")
            output_port = client.create_output_port("Output 1")
            assert len(client._ports) == 3
            assert input_port1 in client._ports
            assert input_port2 in client._ports
            assert output_port in client._ports
            assert input_port1.object_id != input_port2.object_id
            assert input_port1.object_id != output_port.object_id
            assert input_port2.object_id != output_port.object_id
        finally:
            client.dispose()

    def test_midi_client_disposal(self):
        """Test MIDIClient disposal"""
        client = cm.MIDIClient("Disposal Test Client")
        input_port = client.create_input_port("Test Port")
        assert not client.is_disposed
        assert not input_port.is_disposed
        assert len(client._ports) == 1
        client.dispose()
        assert client.is_disposed
        assert input_port.is_disposed
        assert len(client._ports) == 0

    def test_midi_client_operations_on_disposed_object(self):
        """Test operations on disposed MIDIClient"""
        client = cm.MIDIClient("Test Client")
        client.dispose()
        with pytest.raises(RuntimeError, match="has been disposed"):
            client.create_input_port("Test Port")
        with pytest.raises(RuntimeError, match="has been disposed"):
            client.create_output_port("Test Port")


class TestMIDIPort:
    """Test MIDIPort base class functionality"""

    def test_midi_port_creation(self):
        """Test MIDIPort creation"""
        port = cm.MIDIPort("Test Port")
        assert isinstance(port, cm.MIDIPort)
        assert isinstance(port, cm.CoreAudioObject)
        assert port.name == "Test Port"
        assert port._client is None

    def test_midi_port_disposal(self):
        """Test MIDIPort disposal"""
        client = cm.MIDIClient("Test Client")
        try:
            port = client.create_input_port("Test Port")
            assert not port.is_disposed
            port.dispose()
            assert port.is_disposed
        finally:
            client.dispose()


class TestMIDIInputPort:
    """Test MIDIInputPort functionality"""

    def test_midi_input_port_creation(self):
        """Test MIDIInputPort creation through client"""
        client = cm.MIDIClient("Test Client")
        try:
            port = client.create_input_port("Input Port")
            assert isinstance(port, cm.MIDIInputPort)
            assert isinstance(port, cm.MIDIPort)
            assert port.name == "Input Port"
        finally:
            client.dispose()

    def test_midi_input_port_connect_source(self):
        """Test MIDIInputPort source connection"""
        client = cm.MIDIClient("Test Client")
        try:
            port = client.create_input_port("Input Port")

            class MockEndpoint:
                def __init__(self):
                    self.object_id = 12345

            mock_source = MockEndpoint()
            try:
                port.connect_source(mock_source)
            except cm.MIDIError:
                pass
        finally:
            client.dispose()

    def test_midi_input_port_disconnect_source(self):
        """Test MIDIInputPort source disconnection"""
        client = cm.MIDIClient("Test Client")
        try:
            port = client.create_input_port("Input Port")

            class MockEndpoint:
                def __init__(self):
                    self.object_id = 12345

            mock_source = MockEndpoint()
            try:
                port.disconnect_source(mock_source)
            except cm.MIDIError:
                pass
        finally:
            client.dispose()

    def test_midi_input_port_operations_on_disposed_object(self):
        """Test operations on disposed MIDIInputPort"""
        client = cm.MIDIClient("Test Client")
        port = client.create_input_port("Test Port")
        client.dispose()

        class MockEndpoint:
            def __init__(self):
                self.object_id = 12345

        mock_source = MockEndpoint()
        with pytest.raises(RuntimeError, match="has been disposed"):
            port.connect_source(mock_source)
        with pytest.raises(RuntimeError, match="has been disposed"):
            port.disconnect_source(mock_source)


class TestMIDIOutputPort:
    """Test MIDIOutputPort functionality"""

    def test_midi_output_port_creation(self):
        """Test MIDIOutputPort creation through client"""
        client = cm.MIDIClient("Test Client")
        try:
            port = client.create_output_port("Output Port")
            assert isinstance(port, cm.MIDIOutputPort)
            assert isinstance(port, cm.MIDIPort)
            assert port.name == "Output Port"
        finally:
            client.dispose()

    def test_midi_output_port_send_data(self):
        """Test MIDIOutputPort data sending"""
        client = cm.MIDIClient("Test Client")
        try:
            port = client.create_output_port("Output Port")

            class MockEndpoint:
                def __init__(self):
                    self.object_id = 54321

            mock_destination = MockEndpoint()
            test_data = b"\x90@\x7f"
            try:
                port.send_data(mock_destination, test_data)
                port.send_data(mock_destination, test_data, timestamp=1000)
            except cm.MIDIError:
                pass
        finally:
            client.dispose()

    def test_midi_output_port_operations_on_disposed_object(self):
        """Test operations on disposed MIDIOutputPort"""
        client = cm.MIDIClient("Test Client")
        port = client.create_output_port("Test Port")
        client.dispose()

        class MockEndpoint:
            def __init__(self):
                self.object_id = 54321

        mock_destination = MockEndpoint()
        test_data = b"\x90@\x7f"
        with pytest.raises(RuntimeError, match="has been disposed"):
            port.send_data(mock_destination, test_data)


class TestMIDIIntegration:
    """Integration tests for MIDI functionality"""

    def test_midi_vs_functional_api_consistency(self):
        """Test MIDI OO API vs functional API consistency"""
        func_client_id = capi.midi_client_create("Functional Client")
        try:
            func_input_port_id = capi.midi_input_port_create(
                func_client_id, "Functional Input"
            )
            func_output_port_id = capi.midi_output_port_create(
                func_client_id, "Functional Output"
            )
            capi.midi_port_dispose(func_input_port_id)
            capi.midi_port_dispose(func_output_port_id)
        finally:
            capi.midi_client_dispose(func_client_id)
        oo_client = cm.MIDIClient("OO Client")
        try:
            oo_input_port = oo_client.create_input_port("OO Input")
            oo_output_port = oo_client.create_output_port("OO Output")
            assert func_client_id != 0
            assert func_input_port_id != 0
            assert func_output_port_id != 0
            assert oo_client.object_id != 0
            assert oo_input_port.object_id != 0
            assert oo_output_port.object_id != 0
        finally:
            oo_client.dispose()

    def test_midi_client_full_workflow(self):
        """Test complete MIDI workflow"""
        client = cm.MIDIClient("Workflow Client")
        try:
            input_port = client.create_input_port("Workflow Input")
            output_port = client.create_output_port("Workflow Output")
            assert len(client._ports) == 2

            class MockEndpoint:
                def __init__(self, id_val):
                    self.object_id = id_val

            mock_source = MockEndpoint(11111)
            mock_destination = MockEndpoint(22222)
            try:
                input_port.connect_source(mock_source)
                output_port.send_data(mock_destination, b"\x90@\x7f")
                input_port.disconnect_source(mock_source)
            except cm.MIDIError:
                pass
        finally:
            client.dispose()

    def test_midi_multiple_clients(self):
        """Test multiple MIDI clients"""
        client1 = cm.MIDIClient("Client 1")
        client2 = cm.MIDIClient("Client 2")
        try:
            assert client1.object_id != client2.object_id
            assert client1.name != client2.name
            port1 = client1.create_input_port("Port 1")
            port2 = client2.create_output_port("Port 2")
            assert port1._client is client1
            assert port2._client is client2
            assert port1.object_id != port2.object_id
        finally:
            client1.dispose()
            client2.dispose()

    def test_midi_resource_management(self):
        """Test MIDI resource management"""
        for i in range(3):
            client = cm.MIDIClient(f"Resource Test Client {i}")
            for j in range(2):
                client.create_input_port(f"Input {j}")
                client.create_output_port(f"Output {j}")
            assert len(client._ports) == 4
            client.dispose()
            assert client.is_disposed

    def test_midi_error_handling(self):
        """Test MIDI error handling"""
        try:
            empty_client = cm.MIDIClient("")
            empty_client.dispose()
        except cm.MIDIError:
            pass
        clients = []
        try:
            for i in range(10):
                client = cm.MIDIClient(f"Stress Test Client {i}")
                clients.append(client)
        except cm.MIDIError:
            pass
        finally:
            for client in clients:
                if not client.is_disposed:
                    client.dispose()


class TestMIDIPortPolymorphism:
    """Test MIDI port polymorphism and inheritance"""

    def test_midi_port_inheritance(self):
        """Test MIDI port class inheritance"""
        client = cm.MIDIClient("Inheritance Test")
        try:
            input_port = client.create_input_port("Input")
            output_port = client.create_output_port("Output")
            assert isinstance(input_port, cm.MIDIInputPort)
            assert isinstance(input_port, cm.MIDIPort)
            assert isinstance(input_port, cm.CoreAudioObject)
            assert isinstance(output_port, cm.MIDIOutputPort)
            assert isinstance(output_port, cm.MIDIPort)
            assert isinstance(output_port, cm.CoreAudioObject)
            ports = [input_port, output_port]
            for port in ports:
                assert hasattr(port, "name")
                assert hasattr(port, "dispose")
                assert hasattr(port, "is_disposed")
        finally:
            client.dispose()

    def test_midi_port_list_management(self):
        """Test MIDI client port list management"""
        client = cm.MIDIClient("List Test")
        try:
            port1 = client.create_input_port("Input 1")
            port2 = client.create_output_port("Output 1")
            port3 = client.create_input_port("Input 2")
            assert len(client._ports) == 3
            assert all(isinstance(port, cm.MIDIPort) for port in client._ports)
            port2.dispose()
            assert port2.is_disposed
            assert len(client._ports) == 2
        finally:
            client.dispose()
