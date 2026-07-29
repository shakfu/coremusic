"""MIDI classes for coremusic.

This module provides classes for working with MIDI:
- MIDIPort: Base class for MIDI ports
- MIDIInputPort: MIDI input port for receiving MIDI data
- MIDIOutputPort: MIDI output port for sending MIDI data
- MIDIEndpoint: A MIDI source or destination endpoint
- MIDIClient: MIDI client for managing MIDI operations
- MIDIMessageSplitter: Split incoming packet bytes into individual messages

Module-level helpers list the endpoints published by the system
(:func:`get_destinations`, :func:`get_sources`, :func:`find_destination`,
:func:`find_source`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from coremusic import capi
from coremusic.exceptions import FRAMEWORK_ERRORS, MIDIError

__all__ = [
    "MIDIClient",
    "MIDIEndpoint",
    "MIDIInputPort",
    "MIDIMessageSplitter",
    "MIDIOutputPort",
    "MIDIPort",
    "find_destination",
    "find_source",
    "get_destinations",
    "get_sources",
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


def _endpoint_id(endpoint: Any) -> int:
    """Resolve an endpoint argument to a raw MIDIEndpointRef.

    Accepts a :class:`MIDIEndpoint` (or anything exposing ``object_id``) as
    well as a plain integer handle from the functional ``capi`` layer.
    """
    if isinstance(endpoint, int):
        return endpoint
    object_id = getattr(endpoint, "object_id", None)
    if not isinstance(object_id, int):
        raise MIDIError(
            f"Expected a MIDIEndpoint or an endpoint id, got {type(endpoint).__name__}"
        )
    return object_id


class MIDIEndpoint(capi.CoreAudioObject):
    """A MIDI endpoint: either a source (produces MIDI) or a destination
    (consumes MIDI).

    Endpoints come from two places:

    - The system, via :func:`get_sources` / :func:`get_destinations`. These
      belong to other applications or hardware and are never disposed here.
    - This process, via :meth:`MIDIClient.create_virtual_source` and
      :meth:`MIDIClient.create_virtual_destination`. These are owned, and
      :meth:`dispose` destroys them.

    A virtual destination created without a callback buffers incoming packets;
    retrieve them with :meth:`poll`, as for :class:`MIDIInputPort`.
    """

    def __init__(self, endpoint_id: int, name: str | None = None, owned: bool = False):
        super().__init__()
        self._set_object_id(endpoint_id)
        self._owned = owned
        self._client: MIDIClient | None = None
        if name is None:
            try:
                name = capi.midi_endpoint_get_name(endpoint_id)
            except FRAMEWORK_ERRORS:
                name = None
        self._name = name or ""

    @property
    def name(self) -> str:
        """Name of the endpoint as published to CoreMIDI"""
        return self._name

    @property
    def is_owned(self) -> bool:
        """True for virtual endpoints created by this process"""
        return self._owned

    def __repr__(self) -> str:
        status = "disposed" if self.is_disposed else "active"
        kind = "virtual" if self._owned else "system"
        return f"MIDIEndpoint({self._name!r}, {kind}, {status})"

    def __enter__(self) -> MIDIEndpoint:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.dispose()

    def send(self, data: bytes, timestamp: int = 0) -> None:
        """Distribute MIDI data from this virtual source to its connections.

        This is the counterpart of :meth:`MIDIOutputPort.send_data`: it makes
        the endpoint behave as if a device had just produced the given bytes.
        Only meaningful for an endpoint from
        :meth:`MIDIClient.create_virtual_source`.

        Args:
            data: MIDI message bytes
            timestamp: MIDI timestamp (0 for immediate)

        Raises:
            MIDIError: If distribution fails
        """
        self._ensure_not_disposed()
        try:
            capi.midi_received(self.object_id, data, timestamp)
        except Exception as e:
            raise MIDIError(f"Failed to send from source: {e}") from e

    def poll(self, max_events: int = 0) -> list[tuple[int, bytes]]:
        """Drain packets buffered by this virtual destination.

        Args:
            max_events: Maximum number of packets to return, or 0 for all

        Returns:
            List of (host_time, data) tuples in arrival order. Always empty for
            a destination created with a callback.
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
        """Number of packets discarded because the buffer was full."""
        self._ensure_not_disposed()
        return capi.midi_input_dropped(self.object_id)

    def dispose(self) -> None:
        """Dispose of a virtual endpoint.

        A system endpoint is owned by another process, so this only marks the
        wrapper as disposed and leaves the endpoint alone.
        """
        if self.is_disposed:
            return
        if self._owned:
            try:
                capi.midi_endpoint_dispose(self.object_id)
            except FRAMEWORK_ERRORS:
                # Best effort disposal - the client may already be gone
                pass
        if self._client is not None and hasattr(self._client, "_endpoints"):
            try:
                self._client._endpoints.remove(self)
            except ValueError:
                pass  # Already removed
        super().dispose()


def get_sources() -> list[MIDIEndpoint]:
    """List the MIDI sources currently published by the system.

    Returns:
        Source endpoints, in CoreMIDI index order. These are inputs to this
        process: connect one to a :class:`MIDIInputPort` to receive from it.
    """
    return [
        MIDIEndpoint(capi.midi_get_source(i))
        for i in range(capi.midi_get_number_of_sources())
    ]


def get_destinations() -> list[MIDIEndpoint]:
    """List the MIDI destinations currently published by the system.

    Returns:
        Destination endpoints, in CoreMIDI index order. These are outputs from
        this process: pass one to :meth:`MIDIOutputPort.send_data`.
    """
    return [
        MIDIEndpoint(capi.midi_get_destination(i))
        for i in range(capi.midi_get_number_of_destinations())
    ]


def _find_endpoint(endpoints: list[MIDIEndpoint], name: str) -> MIDIEndpoint | None:
    for endpoint in endpoints:
        if endpoint.name == name:
            return endpoint
    lowered = name.lower()
    for endpoint in endpoints:
        if lowered in endpoint.name.lower():
            return endpoint
    return None


def find_source(name: str) -> MIDIEndpoint | None:
    """Find a MIDI source by name.

    Args:
        name: Exact name, or a substring matched case-insensitively when no
            exact match exists

    Returns:
        The first matching source, or None
    """
    return _find_endpoint(get_sources(), name)


def find_destination(name: str) -> MIDIEndpoint | None:
    """Find a MIDI destination by name.

    Args:
        name: Exact name, or a substring matched case-insensitively when no
            exact match exists

    Returns:
        The first matching destination, or None
    """
    return _find_endpoint(get_destinations(), name)


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

    def __enter__(self) -> MIDIPort:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.dispose()

    def dispose(self) -> None:
        """Dispose of the MIDI port"""
        if not self.is_disposed:
            try:
                capi.midi_port_dispose(self.object_id)
            except FRAMEWORK_ERRORS:
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

    def connect_source(self, source: MIDIEndpoint | int) -> None:
        """Connect to a MIDI source

        Args:
            source: A :class:`MIDIEndpoint` from :func:`get_sources`, or a raw
                endpoint id from the functional ``capi`` layer
        """
        self._ensure_not_disposed()
        source_id = _endpoint_id(source)
        try:
            capi.midi_port_connect_source(self.object_id, source_id)
        except Exception as e:
            raise MIDIError(f"Failed to connect source: {e}") from e

    def disconnect_source(self, source: MIDIEndpoint | int) -> None:
        """Disconnect from a MIDI source"""
        self._ensure_not_disposed()
        source_id = _endpoint_id(source)
        try:
            capi.midi_port_disconnect_source(self.object_id, source_id)
        except Exception as e:
            raise MIDIError(f"Failed to disconnect source: {e}") from e

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

    def send_data(
        self, destination: MIDIEndpoint | int, data: bytes, timestamp: int = 0
    ) -> None:
        """Send MIDI data to a destination endpoint

        Args:
            destination: A :class:`MIDIEndpoint` to send data to, or a raw
                endpoint id from the functional ``capi`` layer
            data: MIDI message bytes (following MIDI protocol specification)
            timestamp: MIDI timestamp (0 for immediate, or future timestamp)

        Raises:
            MIDIError: If sending fails

        Example::

            from coremusic.constants import MIDIControlChange
            from coremusic.midi import (
                MIDIClient,
                control_change,
                get_destinations,
                note_off,
                note_on,
            )

            client = MIDIClient("MyApp")
            output_port = client.create_output_port("Output")

            # Pick a destination: a hardware or software endpoint published by
            # the system, or a virtual one owned by this process.
            destinations = get_destinations()
            if destinations:
                destination = destinations[0]
            else:
                destination = client.create_virtual_destination("Synth")

            # Send Note On (middle C, velocity 100)
            output_port.send_data(destination, note_on("C4", 100))

            # Send Control Change (volume to 127)
            output_port.send_data(
                destination, control_change(MIDIControlChange.VOLUME, 127)
            )

            # Send Note Off
            output_port.send_data(destination, note_off("C4"))

            client.dispose()
        """
        self._ensure_not_disposed()
        dest_id = _endpoint_id(destination)
        try:
            capi.midi_send_data(self.object_id, dest_id, data, timestamp)
        except Exception as e:
            raise MIDIError(f"Failed to send data: {e}") from e


class MIDIClient(capi.CoreAudioObject):
    """MIDI client for managing MIDI operations"""

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._ports: list[MIDIPort] = []
        self._endpoints: list[MIDIEndpoint] = []
        try:
            client_id = capi.midi_client_create(name)
            self._set_object_id(client_id)
        except Exception as e:
            raise MIDIError(f"Failed to create MIDI client: {e}") from e

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        if self.is_disposed:
            return f"MIDIClient({self._name!r}, disposed)"
        return f"MIDIClient({self._name!r}, ports={len(self._ports)})"

    def __enter__(self) -> MIDIClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.dispose()

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
            raise MIDIError(f"Failed to create input port: {e}") from e

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
            raise MIDIError(f"Failed to create output port: {e}") from e

    def create_virtual_source(self, name: str) -> MIDIEndpoint:
        """Create a virtual MIDI source owned by this client.

        The source appears to other applications as a MIDI input they can
        connect to. Produce MIDI from it with :meth:`MIDIEndpoint.send`.

        Args:
            name: Name published to CoreMIDI

        Returns:
            The virtual source endpoint, disposed with this client
        """
        self._ensure_not_disposed()
        try:
            endpoint_id = capi.midi_source_create(self.object_id, name)
        except Exception as e:
            raise MIDIError(f"Failed to create virtual source: {e}") from e
        endpoint = MIDIEndpoint(endpoint_id, name, owned=True)
        endpoint._client = self
        self._endpoints.append(endpoint)
        return endpoint

    def create_virtual_destination(
        self,
        name: str,
        callback: Callable[[bytes, int], None] | None = None,
        queue_size: int = 4096,
    ) -> MIDIEndpoint:
        """Create a virtual MIDI destination owned by this client.

        The destination appears to other applications as a MIDI output they
        can send to.

        Args:
            name: Name published to CoreMIDI
            callback: Optional ``callback(data, host_time)`` invoked for each
                incoming packet on the CoreMIDI receive thread. It must be
                short and must not block. When omitted, packets are buffered
                for :meth:`MIDIEndpoint.poll`.
            queue_size: Maximum buffered packets when no callback is given

        Returns:
            The virtual destination endpoint, disposed with this client
        """
        self._ensure_not_disposed()
        try:
            endpoint_id = capi.midi_destination_create(
                self.object_id, name, callback, queue_size
            )
        except Exception as e:
            raise MIDIError(f"Failed to create virtual destination: {e}") from e
        endpoint = MIDIEndpoint(endpoint_id, name, owned=True)
        endpoint._client = self
        self._endpoints.append(endpoint)
        return endpoint

    def dispose(self) -> None:
        """Dispose of the MIDI client, its ports and its virtual endpoints"""
        if not self.is_disposed:
            # Dispose virtual endpoints first - CoreMIDI would drop them with
            # the client anyway, but this keeps the wrappers consistent.
            for endpoint in self._endpoints[:]:
                if not endpoint.is_disposed:
                    try:
                        endpoint.dispose()
                    except FRAMEWORK_ERRORS:
                        pass  # Best effort cleanup

            # Dispose all ports next
            for port in self._ports[
                :
            ]:  # Copy list to avoid modification during iteration
                if not port.is_disposed:
                    try:
                        port.dispose()
                    except FRAMEWORK_ERRORS:
                        pass  # Best effort cleanup

            try:
                capi.midi_client_dispose(self.object_id)
            except FRAMEWORK_ERRORS:
                # Best effort disposal - some MIDI operations may fail in test environments
                pass
            finally:
                # Clear port references and call base dispose
                self._ports.clear()
                super().dispose()
