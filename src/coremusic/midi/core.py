"""MIDI classes for coremusic.

This module provides classes for working with MIDI:
- MIDIPort: Base class for MIDI ports
- MIDIInputPort: MIDI input port for receiving MIDI data
- MIDIOutputPort: MIDI output port for sending MIDI data
- MIDIClient: MIDI client for managing MIDI operations
- MIDIMessageSplitter: Split incoming packet bytes into individual messages
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from coremusic import capi
from coremusic.exceptions import MIDIError

__all__ = [
    "MIDIPort",
    "MIDIInputPort",
    "MIDIOutputPort",
    "MIDIClient",
    "MIDIMessageSplitter",
    "split_midi_messages",
]


# Number of data bytes following each channel voice status (upper nibble).
_CHANNEL_DATA_BYTES = {
    0x80: 2,  # Note Off
    0x90: 2,  # Note On
    0xA0: 2,  # Poly Aftertouch
    0xB0: 2,  # Control Change
    0xC0: 1,  # Program Change
    0xD0: 1,  # Channel Aftertouch
    0xE0: 2,  # Pitch Bend
}

# Number of data bytes following each system common status.
_SYSTEM_DATA_BYTES = {
    0xF1: 1,  # MTC Quarter Frame
    0xF2: 2,  # Song Position Pointer
    0xF3: 1,  # Song Select
}


class MIDIMessageSplitter:
    """Split raw CoreMIDI packet bytes into individual MIDI messages.

    A CoreMIDI packet is not one message. The framework packs as many events as
    fit into a packet, and conversely spreads a large system-exclusive dump over
    several consecutive packets. Real-time bytes may also be interleaved into
    the middle of another message. Splitting correctly therefore needs state
    that persists across packets, which is what this class holds.

    Feed each packet payload to :meth:`push` and it returns the messages that
    became complete. Running status is resolved, and a system-exclusive message
    is emitted only once its terminating 0xF7 arrives.

    Example::

        splitter = MIDIMessageSplitter()
        for host_time, payload in capi.midi_input_poll(port):
            for message in splitter.push(payload):
                print(message.hex())
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Discard any partially received message and clear running status."""
        self._running_status = 0
        self._pending = bytearray()
        self._expected = 0
        self._in_sysex = False

    @property
    def in_sysex(self) -> bool:
        """Whether a system-exclusive message is still being received."""
        return self._in_sysex

    def push(self, data: bytes | bytearray | Iterable[int]) -> list[bytes]:
        """Consume packet bytes and return the messages completed by them."""
        messages: list[bytes] = []

        for byte in data:
            if byte >= 0xF8:
                # Real-time messages are single bytes and may appear anywhere,
                # including inside a system-exclusive message.
                messages.append(bytes([byte]))
                continue

            if byte >= 0x80:
                if self._in_sysex:
                    self._in_sysex = False
                    if byte == 0xF7:
                        # Normal end of system-exclusive.
                        self._pending.append(byte)
                        messages.append(bytes(self._pending))
                        self._pending = bytearray()
                        continue
                    # Anything else aborts the dump. Emit what arrived so the
                    # data is not silently lost, then let the new status byte
                    # start its own message below.
                    messages.append(bytes(self._pending))
                    self._pending = bytearray()

                if byte == 0xF0:
                    self._running_status = 0
                    self._in_sysex = True
                    self._pending = bytearray([byte])
                    self._expected = 0
                elif byte == 0xF7:
                    # Stray end-of-exclusive with no dump in progress.
                    self._pending = bytearray()
                    self._expected = 0
                elif byte < 0xF0:
                    self._running_status = byte
                    self._expected = _CHANNEL_DATA_BYTES[byte & 0xF0]
                    self._pending = bytearray([byte])
                    if self._expected == 0:
                        messages.append(bytes(self._pending))
                        self._pending = bytearray()
                else:
                    # System common clears running status.
                    self._running_status = 0
                    self._expected = _SYSTEM_DATA_BYTES.get(byte, 0)
                    self._pending = bytearray([byte])
                    if self._expected == 0:
                        messages.append(bytes(self._pending))
                        self._pending = bytearray()
                continue

            # Data byte.
            if self._in_sysex:
                self._pending.append(byte)
            elif self._expected > 0:
                self._pending.append(byte)
                self._expected -= 1
                if self._expected == 0:
                    messages.append(bytes(self._pending))
                    self._pending = bytearray()
            elif self._running_status:
                # Running status: the previous status byte is implied.
                self._pending = bytearray([self._running_status, byte])
                self._expected = _CHANNEL_DATA_BYTES[self._running_status & 0xF0] - 1
                if self._expected == 0:
                    messages.append(bytes(self._pending))
                    self._pending = bytearray()
            # Otherwise an orphan data byte with no context: drop it.

        return messages


def split_midi_messages(data: bytes | bytearray) -> list[bytes]:
    """Split one packet's bytes into complete MIDI messages.

    Convenience wrapper around :class:`MIDIMessageSplitter` for callers holding
    a single self-contained packet. Any message left incomplete at the end of
    the data is discarded; use the splitter directly to receive
    system-exclusive dumps that span several packets.
    """
    return MIDIMessageSplitter().push(data)


class MIDIPort(capi.CoreAudioObject):
    """Base class for MIDI ports"""

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._client: MIDIClient | None = None  # Reference to parent MIDIClient

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
    """MIDI input port for receiving MIDI data.

    Unless the port was created with a callback, incoming packets are buffered
    and retrieved with :meth:`poll`::

        client = MIDIClient("MyApp")
        port = client.create_input_port("Input")
        port.connect_source(source)

        while True:
            if port.wait(0.1):
                for host_time, payload in port.poll():
                    print(host_time, payload.hex())
    """

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

    def poll(self, max_events: int = 0) -> list[tuple[int, bytes]]:
        """Drain the buffered incoming packets.

        Args:
            max_events: Maximum number of packets to return, or 0 for all

        Returns:
            List of (host_time, data) tuples in arrival order. Convert
            host_time to seconds with ``capi.midi_host_time_to_seconds``. A
            packet may hold more than one MIDI message; use
            :class:`MIDIMessageSplitter` to separate them. Always empty for a
            port created with a callback.
        """
        self._ensure_not_disposed()
        return capi.midi_input_poll(self.object_id, max_events)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until at least one packet is buffered or the timeout expires.

        Returns:
            True if packets are available, False if the timeout expired
        """
        self._ensure_not_disposed()
        return capi.midi_input_wait(self.object_id, timeout)

    @property
    def pending(self) -> int:
        """Number of packets currently buffered."""
        self._ensure_not_disposed()
        return capi.midi_input_pending(self.object_id)

    @property
    def dropped(self) -> int:
        """Number of packets discarded because the buffer was full.

        A non-zero value means this port is not being polled fast enough, or
        that it was created with too small a ``queue_size``.
        """
        self._ensure_not_disposed()
        return capi.midi_input_dropped(self.object_id)


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

            from coremusic.midi import MIDIClient

            client = MIDIClient("MyApp")
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
            capi.midi_send_data(self.object_id, destination.object_id, data, timestamp)
        except Exception as e:
            raise MIDIError(f"Failed to send data: {e}")


class MIDIClient(capi.CoreAudioObject):
    """MIDI client for managing MIDI operations"""

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._ports: list[MIDIPort] = []
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

    def create_input_port(
        self,
        name: str,
        callback: Callable[[bytes, int], None] | None = None,
        queue_size: int = 4096,
    ) -> MIDIInputPort:
        """Create a MIDI input port.

        Args:
            name: Name for the port
            callback: Optional ``callback(data, host_time)`` invoked for each
                incoming packet on the CoreMIDI receive thread. It must be
                short and must not block. When omitted, packets are buffered
                for :meth:`MIDIInputPort.poll`.
            queue_size: Maximum buffered packets when no callback is given
        """
        self._ensure_not_disposed()
        try:
            port_id = capi.midi_input_port_create(
                self.object_id, name, callback, queue_size
            )
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
