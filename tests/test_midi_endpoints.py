"""Tests for the object-layer MIDI endpoint API.

Regression coverage for issue #2: the documented way to send MIDI could not be
run, because the object layer had no way to obtain a destination endpoint.
MIDIOutputPort.send_data() required an object with an `object_id`, but neither
MIDIClient.create_virtual_destination() nor any endpoint discovery helper
existed - both only lived in the functional `capi` layer.
"""

import subprocess
import sys
import time

import pytest
from conftest import midi_or_skip

from coremusic import capi
from coremusic.exceptions import MIDIError
from coremusic.midi import (
    MIDIClient,
    MIDIEndpoint,
    MIDIMessageSplitter,
    find_destination,
    find_source,
    get_destinations,
    get_sources,
    note_on,
)

# Timeout for a packet to make the round trip through the CoreMIDI server.
DELIVERY_TIMEOUT = 5.0

# Messages for the transport tests. The send and the assertion name the same
# constant, so the two cannot drift apart.
NOTE_ON_C4 = note_on("C4", 127)  # b"\x90\x3c\x7f"
NOTE_ON_E4 = note_on("E4", 100)  # b"\x90\x40\x64"
NOTE_ON_F4 = note_on("F4", 100)  # b"\x90\x41\x64"
NOTE_ON_FS4 = note_on("F#4", 100)  # b"\x90\x42\x64"
NOTE_ON_G4 = note_on("G4", 100)  # b"\x90\x43\x64"


_MIDI_AVAILABLE = False
try:
    _client_id = capi.midi_client_create("_EndpointTestCheck")
    capi.midi_client_dispose(_client_id)
    _MIDI_AVAILABLE = True
except Exception:
    _MIDI_AVAILABLE = False

requires_midi = pytest.mark.skipif(
    not _MIDI_AVAILABLE, reason="MIDI services not available"
)


def collect(endpoint, expected, timeout=DELIVERY_TIMEOUT):
    """Poll an endpoint until `expected` packets arrive or the timeout expires."""
    deadline = time.monotonic() + timeout
    events = []
    while len(events) < expected and time.monotonic() < deadline:
        endpoint.wait(0.05)
        events.extend(endpoint.poll())
    return events


def payloads(events):
    """Split every polled packet into individual MIDI messages."""
    splitter = MIDIMessageSplitter()
    messages = []
    for _host_time, data in events:
        messages.extend(splitter.push(data))
    return messages


def make_client(name):
    """Create a client, retrying and then skipping if the server refuses."""
    return midi_or_skip(lambda: MIDIClient(name))


@pytest.fixture
def client():
    cli = make_client("coremusic-endpoint-tests")
    yield cli
    cli.dispose()


@requires_midi
class TestVirtualDestination:
    """create_virtual_destination() gives send_data() something to aim at."""

    def test_readme_send_example(self, client):
        """The README MIDI example, end to end."""
        output_port = client.create_output_port("Output")
        destination = client.create_virtual_destination("cm-test-readme-dest")

        output_port.send_data(destination, NOTE_ON_C4)

        assert payloads(collect(destination, 1)) == [NOTE_ON_C4]

    def test_endpoint_attributes(self, client):
        destination = client.create_virtual_destination("cm-test-attrs-dest")

        assert isinstance(destination, MIDIEndpoint)
        assert destination.name == "cm-test-attrs-dest"
        assert destination.is_owned
        assert not destination.is_disposed
        assert destination.object_id != 0
        assert "cm-test-attrs-dest" in repr(destination)

    def test_send_accepts_raw_endpoint_id(self, client):
        """The functional layer's integer handles stay usable."""
        output_port = client.create_output_port("cm-test-rawid-out")
        destination = client.create_virtual_destination("cm-test-rawid-dest")

        output_port.send_data(destination.object_id, NOTE_ON_E4)

        assert payloads(collect(destination, 1)) == [NOTE_ON_E4]

    def test_send_rejects_non_endpoint(self, client):
        output_port = client.create_output_port("cm-test-badarg-out")

        with pytest.raises(MIDIError):
            output_port.send_data("not-an-endpoint", NOTE_ON_E4)

    def test_callback_destination(self, client):
        received = []
        destination = client.create_virtual_destination(
            "cm-test-cb-dest", lambda data, host_time: received.append(data)
        )
        output_port = client.create_output_port("cm-test-cb-out")

        output_port.send_data(destination, NOTE_ON_E4)

        deadline = time.monotonic() + DELIVERY_TIMEOUT
        while not received and time.monotonic() < deadline:
            time.sleep(0.05)
        assert received == [NOTE_ON_E4]
        # A callback destination buffers nothing.
        assert destination.pending == 0

    def test_pending_and_dropped(self, client):
        """The counters must track buffered packets, not just read as zero.

        Asserting `pending == 0` on a freshly created destination is satisfied
        by counters hardwired to 0, so it cannot tell a working implementation
        from a broken one. Send real traffic and watch the count move.
        """
        destination = client.create_virtual_destination("cm-test-counters-dest")
        output_port = client.create_output_port("cm-test-counters-out")

        assert destination.pending == 0
        assert destination.dropped == 0

        output_port.send_data(destination, NOTE_ON_E4)

        deadline = time.monotonic() + DELIVERY_TIMEOUT
        while destination.pending == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert destination.pending == 1, "buffered packet was not counted"
        assert destination.dropped == 0

        # Draining returns the packet and clears the counter.
        assert [data for _ts, data in destination.poll()] == [NOTE_ON_E4]
        assert destination.pending == 0

    def test_disposed_with_client(self):
        cli = make_client("cm-test-dispose-client")
        destination = cli.create_virtual_destination("cm-test-dispose-dest")

        cli.dispose()

        assert destination.is_disposed
        with pytest.raises(RuntimeError):
            destination.poll()

    def test_explicit_dispose_detaches_from_client(self, client):
        destination = client.create_virtual_destination("cm-test-detach-dest")

        destination.dispose()

        assert destination.is_disposed
        assert destination not in client._endpoints

    def test_context_manager(self, client):
        with client.create_virtual_destination("cm-test-ctx-dest") as destination:
            assert not destination.is_disposed
        assert destination.is_disposed


@requires_midi
class TestVirtualSource:
    """create_virtual_source() is the receive-side counterpart."""

    def test_round_trip_to_input_port(self, client):
        source = client.create_virtual_source("cm-test-src")
        input_port = client.create_input_port("cm-test-src-in")
        input_port.connect_source(source)
        # Give CoreMIDI time to establish the connection.
        time.sleep(0.2)

        source.send(NOTE_ON_E4)

        assert input_port.wait(DELIVERY_TIMEOUT)
        assert payloads(input_port.poll()) == [NOTE_ON_E4]

    def test_connect_source_accepts_raw_endpoint_id(self, client):
        source = client.create_virtual_source("cm-test-src-rawid")
        input_port = client.create_input_port("cm-test-src-rawid-in")
        input_port.connect_source(source.object_id)
        time.sleep(0.2)

        source.send(NOTE_ON_F4)

        assert input_port.wait(DELIVERY_TIMEOUT)
        assert payloads(input_port.poll()) == [NOTE_ON_F4]

    def test_connect_source_rejects_non_endpoint(self, client):
        input_port = client.create_input_port("cm-test-src-badarg-in")

        with pytest.raises(MIDIError):
            input_port.connect_source(object())

    def test_disconnect_source(self, client):
        source = client.create_virtual_source("cm-test-src-disconnect")
        input_port = client.create_input_port("cm-test-src-disconnect-in")
        input_port.connect_source(source)
        time.sleep(0.2)

        input_port.disconnect_source(source)
        time.sleep(0.2)
        source.send(NOTE_ON_FS4)

        assert not input_port.wait(0.5)
        assert input_port.poll() == []


@requires_midi
class TestEndpointDiscovery:
    """get_/find_ helpers expose the system's endpoints as objects."""

    def test_virtual_endpoints_are_discoverable(self, client):
        name = "cm-test-discovery-dest"
        client.create_virtual_destination(name)
        # CoreMIDI publishes the endpoint asynchronously.
        time.sleep(0.2)

        assert name in [d.name for d in get_destinations()]

    def test_virtual_source_is_discoverable(self, client):
        name = "cm-test-discovery-src"
        client.create_virtual_source(name)
        time.sleep(0.2)

        assert name in [s.name for s in get_sources()]

    def test_discovered_endpoints_are_not_owned(self, client):
        name = "cm-test-notowned-dest"
        client.create_virtual_destination(name)
        time.sleep(0.2)

        found = find_destination(name)
        assert found is not None
        assert not found.is_owned

        # Disposing a discovered wrapper must not destroy the endpoint.
        found.dispose()
        time.sleep(0.2)
        assert name in [d.name for d in get_destinations()]

    def test_find_destination_matches_substring(self, client):
        client.create_virtual_destination("cm-test-substring-dest")
        time.sleep(0.2)

        assert find_destination("SUBSTRING-DEST") is not None

    def test_find_source_matches_substring(self, client):
        client.create_virtual_source("cm-test-substring-src")
        time.sleep(0.2)

        assert find_source("SUBSTRING-SRC") is not None

    def test_find_returns_none_when_absent(self):
        assert find_destination("cm-test-no-such-endpoint-xyz") is None
        assert find_source("cm-test-no-such-endpoint-xyz") is None

    def test_send_to_discovered_destination(self, client):
        name = "cm-test-discovered-send-dest"
        owned = client.create_virtual_destination(name)
        time.sleep(0.2)

        discovered = find_destination(name)
        assert discovered is not None

        output_port = client.create_output_port("cm-test-discovered-send-out")
        output_port.send_data(discovered, NOTE_ON_G4)

        assert payloads(collect(owned, 1)) == [NOTE_ON_G4]


@requires_midi
class TestDisposeWithPacketsInFlight:
    """Disposal must not deadlock against the CoreMIDI receive thread.

    Regression coverage: `midi_client_dispose()` held the GIL across
    `MIDIClientDispose`, which waits for in-flight read proc calls to return.
    The read proc blocks on `with gil:` to deliver its packet, so sending to a
    virtual destination and then disposing hung the process, permanently.
    Sending is what arms it - dispose with nothing in flight always worked -
    so a Link session or any other load that delayed delivery made it look
    intermittent.

    A hang here fails as a timeout in CI rather than a normal assertion, which
    is the whole reason the subprocess wrapper below exists.
    """

    SCRIPT = """
import time

import coremusic.capi as capi


def make_client(name):
    # The server refuses a new client now and then, especially after the rest
    # of the suite has churned through several dozen. conftest.midi_or_skip
    # handles that in-process; a child process needs its own retry.
    for attempt in range(5):
        try:
            return capi.midi_client_create(name)
        except RuntimeError:
            if attempt == 4:
                raise
            time.sleep(0.3)


# Each round is its own client: the window is narrow enough that a single
# attempt only deadlocked about four times in five, and repeating closes it.
for i in range(3):
    client = make_client(f"cm-test-inflight-{{i}}")
    port = capi.midi_output_port_create(client, "out")
    dest = capi.midi_destination_create(client, f"cm-test-inflight-dest-{{i}}")

    # Let the server finish publishing the endpoint, so the sends below really
    # do reach the read proc. Without this nothing is in flight to deadlock
    # against.
    time.sleep(0.3)

    for _ in range(50):
        capi.midi_send_data(port, dest, bytes([0xF8]))

    # Packets are still on their way to our own read proc at this point
    capi.{call}

print("disposed")
"""

    def run_disposal(self, call):
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT.format(call=call)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0 and "MIDIClientCreate failed" in result.stderr:
            # The retry above gave up: the server is refusing clients outright,
            # which says nothing about the deadlock this test covers.
            pytest.skip(f"MIDI services unavailable: {result.stderr.strip()[-120:]}")
        assert result.returncode == 0, result.stderr
        assert "disposed" in result.stdout

    def test_client_dispose_with_packets_in_flight(self):
        self.run_disposal("midi_client_dispose(client)")

    def test_port_dispose_with_packets_in_flight(self):
        self.run_disposal("midi_port_dispose(port)")

    def test_endpoint_dispose_with_packets_in_flight(self):
        self.run_disposal("midi_endpoint_dispose(dest)")

    def test_object_layer_dispose_with_packets_in_flight(self):
        """The same path through MIDIClient.dispose()."""
        client = make_client("cm-test-object-inflight")
        destination = client.create_virtual_destination("cm-test-object-inflight-dest")
        port = client.create_output_port("cm-test-object-inflight-out")
        time.sleep(1.0)

        for _ in range(50):
            port.send_data(destination, b"\\xf8")

        client.dispose()
        assert client.is_disposed


@requires_midi
class TestClientContextManager:
    def test_client_disposes_on_exit(self):
        with make_client("cm-test-client-ctx") as cli:
            port = cli.create_output_port("cm-test-client-ctx-out")
            assert not cli.is_disposed
        assert cli.is_disposed
        assert port.is_disposed

    def test_port_context_manager(self, client):
        with client.create_output_port("cm-test-port-ctx") as port:
            assert not port.is_disposed
        assert port.is_disposed


@requires_midi
class TestDeadConnectionDiagnostic:
    """MIDIClientCreate explains the failure that cannot be retried.

    MIDIServer exits a few seconds after its last client disconnects, which
    invalidates the CoreMIDI connection of every process still running. Nothing
    in that process can create a client again, so the bare "Unknown error code
    -2" that CoreMIDI reports is the least helpful moment to leave a caller
    without an explanation.

    The live failure cannot be provoked from here: the session fixture in
    conftest holds a client open precisely so the server never exits. These
    check the message the call would raise.
    """

    def test_dead_connection_status_is_explained(self):
        for status in capi._MIDI_DEAD_CONNECTION_STATUSES:
            message = capi._midi_client_create_message(status)
            assert "MIDIClientCreate failed" in message
            assert "MIDIServer" in message
            assert "Keep one MIDI client alive" in message
            assert "restarted" in message

    def test_other_statuses_are_not_given_the_hint(self):
        """paramErr and friends have nothing to do with a dead connection."""
        message = capi._midi_client_create_message(-50)
        assert "MIDIClientCreate failed" in message
        assert "MIDIServer" not in message

    def test_hint_matches_the_documented_statuses(self):
        assert capi._MIDI_DEAD_CONNECTION_STATUSES == (-2, -304)
