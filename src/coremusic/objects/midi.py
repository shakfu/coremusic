"""MIDI classes for coremusic.

This module provides classes for working with MIDI:
- MIDIPort: Base class for MIDI ports
- MIDIInputPort: MIDI input port for receiving MIDI data
- MIDIOutputPort: MIDI output port for sending MIDI data
- MIDIClient: MIDI client for managing MIDI operations
"""

from __future__ import annotations

from typing import Any, List, Optional

from .. import capi
from .exceptions import MIDIError

__all__ = [
    "MIDIPort",
    "MIDIInputPort",
    "MIDIOutputPort",
    "MIDIClient",
]


class MIDIPort(capi.CoreAudioObject):
    """Base class for MIDI ports"""

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._client: Optional["MIDIClient"] = None  # Reference to parent MIDIClient

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        status = "disposed" if self.is_disposed else "active"
        return f"{self.__class__.__name__}({self._name!r}, {status})"

    def dispose(self) -> None:
        """Dispose of the MIDI port"""
        if not self.is_disposed:
            try:
                capi.midi_port_dispose(self.object_id)
            except Exception:
                # Best effort disposal - some MIDI operations may fail in test environments
                pass
            finally:
                # Remove from client's port list if we have a client reference
                if self._client and hasattr(self._client, "_ports"):
                    try:
                        self._client._ports.remove(self)
                    except ValueError:
                        pass  # Already removed
                super().dispose()


class MIDIInputPort(MIDIPort):
    """MIDI input port for receiving MIDI data"""

    def connect_source(self, source: Any) -> None:
        """Connect to a MIDI source"""
        self._ensure_not_disposed()
        try:
            capi.midi_port_connect_source(self.object_id, source.object_id)
        except Exception as e:
            raise MIDIError(f"Failed to connect source: {e}")

    def disconnect_source(self, source: Any) -> None:
        """Disconnect from a MIDI source"""
        self._ensure_not_disposed()
        try:
            capi.midi_port_disconnect_source(self.object_id, source.object_id)
        except Exception as e:
            raise MIDIError(f"Failed to disconnect source: {e}")


class MIDIOutputPort(MIDIPort):
    """MIDI output port for sending MIDI data"""

    def send_data(self, destination: Any, data: bytes, timestamp: int = 0) -> None:
        """Send MIDI data to a destination endpoint

        Args:
            destination: MIDIEndpoint to send data to
            data: MIDI message bytes (following MIDI protocol specification)
            timestamp: MIDI timestamp (0 for immediate, or future timestamp)

        Raises:
            MIDIError: If sending fails

        Example::

            import coremusic as cm

            client = cm.MIDIClient("MyApp")
            output_port = client.create_output_port("Output")

            # Get destination (e.g., virtual destination or hardware endpoint)
            destination = client.create_virtual_destination("Synth")

            # Send Note On (middle C, velocity 100)
            note_on = bytes([0x90, 0x3C, 0x64])  # Status, note, velocity
            output_port.send_data(destination, note_on)

            # Send Control Change (CC 7 = volume to 127)
            cc_volume = bytes([0xB0, 0x07, 0x7F])  # Status, controller, value
            output_port.send_data(destination, cc_volume)

            # Send Note Off
            note_off = bytes([0x80, 0x3C, 0x00])
            output_port.send_data(destination, note_off)
        """
        self._ensure_not_disposed()
        try:
            capi.midi_send(self.object_id, destination.object_id, data, timestamp)
        except Exception as e:
            raise MIDIError(f"Failed to send data: {e}")


class MIDIClient(capi.CoreAudioObject):
    """MIDI client for managing MIDI operations"""

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._ports: List[MIDIPort] = []
        try:
            client_id = capi.midi_client_create(name)
            self._set_object_id(client_id)
        except Exception as e:
            raise MIDIError(f"Failed to create MIDI client: {e}")

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        if self.is_disposed:
            return f"MIDIClient({self._name!r}, disposed)"
        return f"MIDIClient({self._name!r}, ports={len(self._ports)})"

    def create_input_port(self, name: str) -> MIDIInputPort:
        """Create a MIDI input port"""
        self._ensure_not_disposed()
        try:
            port_id = capi.midi_input_port_create(self.object_id, name)
            port = MIDIInputPort(name)
            port._set_object_id(port_id)
            port._client = self
            self._ports.append(port)
            return port
        except Exception as e:
            raise MIDIError(f"Failed to create input port: {e}")

    def create_output_port(self, name: str) -> MIDIOutputPort:
        """Create a MIDI output port"""
        self._ensure_not_disposed()
        try:
            port_id = capi.midi_output_port_create(self.object_id, name)
            port = MIDIOutputPort(name)
            port._set_object_id(port_id)
            port._client = self
            self._ports.append(port)
            return port
        except Exception as e:
            raise MIDIError(f"Failed to create output port: {e}")

    def dispose(self) -> None:
        """Dispose of the MIDI client and all its ports"""
        if not self.is_disposed:
            # Dispose all ports first
            for port in self._ports[
                :
            ]:  # Copy list to avoid modification during iteration
                if not port.is_disposed:
                    try:
                        port.dispose()
                    except Exception:
                        pass  # Best effort cleanup

            try:
                capi.midi_client_dispose(self.object_id)
            except Exception:
                # Best effort disposal - some MIDI operations may fail in test environments
                pass
            finally:
                # Clear port references and call base dispose
                self._ports.clear()
                super().dispose()
