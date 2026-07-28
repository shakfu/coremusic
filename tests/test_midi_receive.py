"""Tests for MIDI input reception.

Regression coverage for issue #1: monitoring a MIDI input port crashed with
SIGSEGV as soon as a packet arrived, because input ports and virtual
destinations were created with a NULL CoreMIDI read proc.
"""

import threading
import time

import pytest
from conftest import midi_or_skip

from coremusic import capi
from coremusic.midi import MIDIClient, MIDIMessageSplitter, split_midi_messages

# Timeout for a packet to make the round trip through the CoreMIDI server.
DELIVERY_TIMEOUT = 5.0


_MIDI_AVAILABLE = False
try:
    _client_id = capi.midi_client_create("_ReceiveTestCheck")
    capi.midi_client_dispose(_client_id)
    _MIDI_AVAILABLE = True
except Exception:
    _MIDI_AVAILABLE = False

requires_midi = pytest.mark.skipif(
    not _MIDI_AVAILABLE, reason="MIDI services not available"
)


def collect(obj_id, expected, timeout=DELIVERY_TIMEOUT):
    """Poll until `expected` packets have arrived or the timeout expires."""
    deadline = time.monotonic() + timeout
    events = []
    while len(events) < expected and time.monotonic() < deadline:
        capi.midi_input_wait(obj_id, 0.05)
        events.extend(capi.midi_input_poll(obj_id))
    return events


def payloads(events):
    """Split every polled packet into individual MIDI messages."""
    splitter = MIDIMessageSplitter()
    messages = []
    for _host_time, data in events:
        messages.extend(splitter.push(data))
    return messages


def collect_messages(obj_id, expected, timeout=DELIVERY_TIMEOUT):
    """Poll until `expected` MIDI *messages* arrive, or the timeout expires.

    `collect` counts packets, which is not the same thing: CoreMIDI decides how
    many messages travel in a packet, so two sends may arrive coalesced into one
    or split across two. Waiting for one packet and then asserting two messages
    passes only while the coalescing happens to go your way - which is how
    `test_input_port_receives_from_virtual_source` passed locally for months and
    failed on a loaded CI runner.

    One splitter is kept across polls so a message spanning packets is still
    assembled correctly.
    """
    deadline = time.monotonic() + timeout
    splitter = MIDIMessageSplitter()
    messages = []
    while len(messages) < expected and time.monotonic() < deadline:
        capi.midi_input_wait(obj_id, 0.05)
        for _host_time, data in capi.midi_input_poll(obj_id):
            messages.extend(splitter.push(data))
    return messages


@pytest.fixture
def client():
    client_id = midi_or_skip(
        lambda: capi.midi_client_create("coremusic-receive-tests")
    )
    yield client_id
    capi.midi_client_dispose(client_id)


# ============================================================================
# Message splitting (no MIDI services required)
# ============================================================================


class TestMIDIMessageSplitter:
    """A CoreMIDI packet is not one MIDI message."""

    def test_single_message(self):
        assert split_midi_messages(bytes([0x90, 0x40, 0x64])) == [b"\x90\x40\x64"]

    def test_coalesced_messages(self):
        # CoreMIDI packs several same-timestamp events into one packet.
        data = bytes([0x90, 0x40, 0x64, 0x90, 0x41, 0x64, 0x80, 0x40, 0x00])
        assert split_midi_messages(data) == [
            b"\x90\x40\x64",
            b"\x90\x41\x64",
            b"\x80\x40\x00",
        ]

    def test_running_status(self):
        data = bytes([0x90, 0x40, 0x64, 0x41, 0x64, 0x42, 0x64])
        assert split_midi_messages(data) == [
            b"\x90\x40\x64",
            b"\x90\x41\x64",
            b"\x90\x42\x64",
        ]

    def test_one_byte_data_messages(self):
        # Program change and channel aftertouch carry a single data byte.
        data = bytes([0xC0, 0x05, 0xD0, 0x40])
        assert split_midi_messages(data) == [b"\xc0\x05", b"\xd0\x40"]

    def test_system_common(self):
        data = bytes([0xF2, 0x00, 0x10, 0xF3, 0x02, 0xF6])
        assert split_midi_messages(data) == [b"\xf2\x00\x10", b"\xf3\x02", b"\xf6"]

    def test_realtime_interleaved_mid_message(self):
        # A clock byte may land between the status and data bytes.
        data = bytes([0x90, 0xF8, 0x40, 0x64])
        assert split_midi_messages(data) == [b"\xf8", b"\x90\x40\x64"]

    def test_sysex_complete(self):
        data = bytes.fromhex("F07F7F080200011540694BF7")
        assert split_midi_messages(data) == [data]

    def test_sysex_split_across_packets(self):
        splitter = MIDIMessageSplitter()
        assert splitter.push(bytes([0xF0, 0x7F, 0x7F])) == []
        assert splitter.in_sysex
        assert splitter.push(bytes([0x08, 0x02, 0xF7])) == [b"\xf0\x7f\x7f\x08\x02\xf7"]
        assert not splitter.in_sysex

    def test_realtime_inside_sysex(self):
        splitter = MIDIMessageSplitter()
        messages = splitter.push(bytes([0xF0, 0x01, 0xF8, 0x02, 0xF7]))
        assert messages == [b"\xf8", b"\xf0\x01\x02\xf7"]

    def test_aborted_sysex_is_not_lost(self):
        data = bytes([0xF0, 0x01, 0x02, 0x90, 0x40, 0x64])
        assert split_midi_messages(data) == [b"\xf0\x01\x02", b"\x90\x40\x64"]

    def test_orphan_data_bytes_dropped(self):
        assert split_midi_messages(bytes([0x40, 0x64])) == []

    def test_incomplete_trailer_dropped(self):
        assert split_midi_messages(bytes([0x90, 0x40, 0x64, 0x90, 0x41])) == [
            b"\x90\x40\x64"
        ]

    def test_reset_clears_state(self):
        splitter = MIDIMessageSplitter()
        splitter.push(bytes([0xF0, 0x01]))
        splitter.reset()
        assert not splitter.in_sysex
        assert splitter.push(bytes([0xF7])) == []


# ============================================================================
# Receiving through CoreMIDI
# ============================================================================


@requires_midi
class TestVirtualDestinationReceive:
    """A virtual destination must survive and deliver incoming packets."""

    def test_send_to_virtual_destination_does_not_crash(self, client):
        # Issue #1: this used to jump to a NULL read proc and kill the process.
        dest = capi.midi_destination_create(client, "cm-test-dest-crash")
        out = capi.midi_output_port_create(client, "cm-test-out-crash")
        capi.midi_send_data(out, dest, bytes([0x90, 0x40, 0x64]))
        time.sleep(0.2)
        capi.midi_port_dispose(out)
        capi.midi_endpoint_dispose(dest)

    def test_poll_returns_sent_data(self, client):
        dest = capi.midi_destination_create(client, "cm-test-dest-poll")
        out = capi.midi_output_port_create(client, "cm-test-out-poll")
        try:
            capi.midi_send_data(out, dest, bytes([0x92, 0x15, 0x45]))
            assert payloads(collect(dest, 1)) == [b"\x92\x15\x45"]
        finally:
            capi.midi_port_dispose(out)
            capi.midi_endpoint_dispose(dest)

    def test_poll_returns_sysex(self, client):
        sysex = bytes.fromhex("F07F7F080200011540694BF7")
        dest = capi.midi_destination_create(client, "cm-test-dest-sysex")
        out = capi.midi_output_port_create(client, "cm-test-out-sysex")
        try:
            capi.midi_send_data(out, dest, sysex)
            assert payloads(collect(dest, 1)) == [sysex]
        finally:
            capi.midi_port_dispose(out)
            capi.midi_endpoint_dispose(dest)

    def test_callback_receives_data(self, client):
        received = []
        arrived = threading.Event()

        def on_midi(data, host_time):
            received.append((data, host_time))
            arrived.set()

        dest = capi.midi_destination_create(client, "cm-test-dest-cb", on_midi)
        out = capi.midi_output_port_create(client, "cm-test-out-cb")
        try:
            capi.midi_send_data(out, dest, bytes([0x90, 0x40, 0x64]))
            assert arrived.wait(DELIVERY_TIMEOUT)
            assert received[0][0] == b"\x90\x40\x64"
        finally:
            capi.midi_port_dispose(out)
            capi.midi_endpoint_dispose(dest)

    def test_callback_exception_does_not_crash(self, client):
        calls = []

        def on_midi(data, host_time):
            calls.append(data)
            raise ValueError("callback failure")

        dest = capi.midi_destination_create(client, "cm-test-dest-raise", on_midi)
        out = capi.midi_output_port_create(client, "cm-test-out-raise")
        try:
            capi.midi_send_data(out, dest, bytes([0x90, 0x40, 0x64]))
            deadline = time.monotonic() + DELIVERY_TIMEOUT
            while not calls and time.monotonic() < deadline:
                time.sleep(0.05)
            assert calls
            # A second message must still be delivered after the failure.
            capi.midi_send_data(out, dest, bytes([0x80, 0x40, 0x00]))
            time.sleep(0.3)
            assert len(calls) >= 2
        finally:
            capi.midi_port_dispose(out)
            capi.midi_endpoint_dispose(dest)

    def test_callback_port_polls_empty(self, client):
        dest = capi.midi_destination_create(
            client, "cm-test-dest-cbpoll", lambda data, ts: None
        )
        try:
            assert capi.midi_input_poll(dest) == []
            assert capi.midi_input_pending(dest) == 0
            assert capi.midi_input_wait(dest, 0.01) is False
        finally:
            capi.midi_endpoint_dispose(dest)


@requires_midi
class TestInputPortReceive:
    """The path from issue #1: an input port connected to a live source."""

    def test_input_port_receives_from_virtual_source(self, client):
        source = capi.midi_source_create(client, "cm-test-src")
        port = capi.midi_input_port_create(client, "cm-test-input")
        try:
            capi.midi_port_connect_source(port, source)
            # Give the CoreMIDI server time to establish the connection.
            time.sleep(0.2)

            capi.midi_received(source, bytes([0x92, 0x15, 0x45]))
            capi.midi_received(source, bytes([0x82, 0x15, 0x20]))

            messages = collect_messages(port, 2)
            assert b"\x92\x15\x45" in messages
            assert b"\x82\x15\x20" in messages
        finally:
            capi.midi_port_dispose(port)
            capi.midi_endpoint_dispose(source)

    def test_timestamps_are_host_time(self, client):
        source = capi.midi_source_create(client, "cm-test-src-ts")
        port = capi.midi_input_port_create(client, "cm-test-input-ts")
        try:
            capi.midi_port_connect_source(port, source)
            time.sleep(0.2)

            sent_at = capi.midi_current_host_time()
            capi.midi_received(source, bytes([0x90, 0x40, 0x64]), sent_at)

            events = collect(port, 1)
            assert events
            host_time = events[0][0]
            # The timestamp must survive as a full 64-bit host time, not be
            # truncated to 32 bits.
            assert host_time == sent_at
            assert capi.midi_host_time_to_seconds(host_time) > 0.0
        finally:
            capi.midi_port_dispose(port)
            capi.midi_endpoint_dispose(source)

    def test_wait_times_out_when_idle(self, client):
        port = capi.midi_input_port_create(client, "cm-test-input-idle")
        try:
            start = time.monotonic()
            assert capi.midi_input_wait(port, 0.1) is False
            assert time.monotonic() - start >= 0.05
        finally:
            capi.midi_port_dispose(port)

    def test_max_events_limits_the_drain(self, client):
        source = capi.midi_source_create(client, "cm-test-src-max")
        port = capi.midi_input_port_create(client, "cm-test-input-max")
        try:
            capi.midi_port_connect_source(port, source)
            time.sleep(0.2)

            # Space the sends out so CoreMIDI cannot merge them into one packet.
            for i in range(3):
                capi.midi_received(source, bytes([0x90, 0x40 + i, 0x64]))
                time.sleep(0.05)

            # Wait for arrival without draining the queue.
            deadline = time.monotonic() + DELIVERY_TIMEOUT
            while capi.midi_input_pending(port) < 3 and time.monotonic() < deadline:
                time.sleep(0.05)

            total = capi.midi_input_pending(port)
            assert total >= 2, "expected separate packets for spaced-out sends"
            assert len(capi.midi_input_poll(port, 1)) == 1
            assert capi.midi_input_pending(port) == total - 1
        finally:
            capi.midi_port_dispose(port)
            capi.midi_endpoint_dispose(source)


@requires_midi
class TestReceiverLifetime:
    """Receivers must be released with the object that owns them."""

    def test_poll_unknown_object_raises(self):
        with pytest.raises(ValueError):
            capi.midi_input_poll(0)

    def test_port_dispose_releases_receiver(self, client):
        port = capi.midi_input_port_create(client, "cm-test-input-dispose")
        capi.midi_port_dispose(port)
        with pytest.raises(ValueError):
            capi.midi_input_poll(port)

    def test_client_dispose_releases_child_receivers(self):
        client_id = midi_or_skip(
            lambda: capi.midi_client_create("cm-test-client-dispose")
        )
        port = capi.midi_input_port_create(client_id, "cm-test-input-child")
        dest = capi.midi_destination_create(client_id, "cm-test-dest-child")
        capi.midi_client_dispose(client_id)
        with pytest.raises(ValueError):
            capi.midi_input_poll(port)
        with pytest.raises(ValueError):
            capi.midi_input_poll(dest)

    def test_failed_creation_does_not_leak(self, client):
        # An invalid client must not leave a dangling receiver behind.
        with pytest.raises(RuntimeError):
            capi.midi_input_port_create(0, "cm-test-input-bad")

    def test_invalid_callback_rejected(self, client):
        with pytest.raises(TypeError):
            capi.midi_input_port_create(client, "cm-test-input-notcallable", 42)

    def test_invalid_queue_size_rejected(self, client):
        with pytest.raises(ValueError):
            capi.midi_input_port_create(client, "cm-test-input-badqueue", None, 0)


@requires_midi
class TestMIDIInputPortObject:
    """The object layer exposes the same receive API."""

    def test_poll_through_input_port_object(self):
        client = midi_or_skip(lambda: MIDIClient("cm-test-object-client"))
        try:
            port = client.create_input_port("cm-test-object-input")
            source_id = capi.midi_source_create(client.object_id, "cm-test-object-src")
            try:
                capi.midi_port_connect_source(port.object_id, source_id)
                time.sleep(0.2)
                capi.midi_received(source_id, bytes([0x90, 0x40, 0x64]))

                assert port.wait(DELIVERY_TIMEOUT)
                events = port.poll()
                assert payloads(events) == [b"\x90\x40\x64"]
                assert port.pending == 0
                assert port.dropped == 0
            finally:
                capi.midi_endpoint_dispose(source_id)
        finally:
            client.dispose()

    def test_output_port_send_data_round_trip(self):
        """MIDIOutputPort.send_data used to call a capi function that did not exist."""

        class Endpoint:
            def __init__(self, object_id):
                self.object_id = object_id

        client = midi_or_skip(lambda: MIDIClient("cm-test-object-send-client"))
        try:
            out = client.create_output_port("cm-test-object-out")
            dest_id = capi.midi_destination_create(
                client.object_id, "cm-test-object-send-dest"
            )
            try:
                out.send_data(Endpoint(dest_id), bytes([0x90, 0x40, 0x64]))
                assert payloads(collect(dest_id, 1)) == [b"\x90\x40\x64"]
            finally:
                capi.midi_endpoint_dispose(dest_id)
        finally:
            client.dispose()

    def test_callback_through_input_port_object(self):
        arrived = threading.Event()
        received = []

        client = midi_or_skip(lambda: MIDIClient("cm-test-object-cb-client"))
        try:
            port = client.create_input_port(
                "cm-test-object-cb-input",
                lambda data, host_time: (received.append(data), arrived.set()),
            )
            source_id = capi.midi_source_create(
                client.object_id, "cm-test-object-cb-src"
            )
            try:
                capi.midi_port_connect_source(port.object_id, source_id)
                time.sleep(0.2)
                capi.midi_received(source_id, bytes([0x90, 0x40, 0x64]))
                assert arrived.wait(DELIVERY_TIMEOUT)
                assert received[0] == b"\x90\x40\x64"
            finally:
                capi.midi_endpoint_dispose(source_id)
        finally:
            client.dispose()
