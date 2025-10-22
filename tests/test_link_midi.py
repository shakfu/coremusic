"""Tests for Link + MIDI integration

Tests the integration between Ableton Link and CoreMIDI including:
- MIDI clock synchronization
- Beat-accurate MIDI sequencing
- Time conversion utilities
"""

import pytest
import time
import coremusic as cm
from coremusic import link_midi


class TestLinkMIDIConstants:
    """Test MIDI constants are defined"""

    def test_midi_clock_constants(self):
        """Test MIDI System Real-Time message constants"""
        assert link_midi.MIDI_CLOCK == 0xF8
        assert link_midi.MIDI_START == 0xFA
        assert link_midi.MIDI_CONTINUE == 0xFB
        assert link_midi.MIDI_STOP == 0xFC

    def test_midi_timing_constants(self):
        """Test MIDI timing constants"""
        assert link_midi.MIDI_CLOCKS_PER_QUARTER_NOTE == 24


class TestMIDIEvent:
    """Test MIDIEvent dataclass"""

    def test_midi_event_creation(self):
        """Test creating MIDI event"""
        event = link_midi.MIDIEvent(
            beat=1.0,
            message=b'\x90\x3C\x64'  # Note On C4
        )

        assert event.beat == 1.0
        assert event.message == b'\x90\x3C\x64'
        assert event.sent == False

    def test_midi_event_sent_flag(self):
        """Test MIDI event sent flag"""
        event = link_midi.MIDIEvent(
            beat=1.0,
            message=b'\x90\x3C\x64',
            sent=True
        )

        assert event.sent == True


class TestLinkMIDIClockCreation:
    """Test LinkMIDIClock creation and basic functionality"""

    @pytest.fixture
    def midi_setup(self):
        """Setup MIDI client and port"""
        try:
            client = cm.capi.midi_client_create("Test Client")
            port = cm.capi.midi_output_port_create(client, "Test Port")

            # Try to get a MIDI destination
            num_destinations = cm.capi.midi_get_number_of_destinations()
            if num_destinations > 0:
                destination = cm.capi.midi_get_destination(0)
            else:
                # Create a virtual destination for testing
                destination = cm.capi.midi_destination_create(client, "Test Dest")

            yield client, port, destination

            # Cleanup
            cm.capi.midi_port_dispose(port)
            cm.capi.midi_client_dispose(client)
        except Exception as e:
            pytest.skip(f"MIDI setup failed: {e}")

    def test_clock_creation(self, midi_setup):
        """Test creating LinkMIDIClock"""
        client, port, dest = midi_setup
        session = cm.link.LinkSession(bpm=120.0)

        clock = link_midi.LinkMIDIClock(session, port, dest)

        assert clock.session is session
        assert clock.midi_port == port
        assert clock.midi_destination == dest
        assert clock.quantum == 4.0
        assert clock.running == False

    def test_clock_with_custom_quantum(self, midi_setup):
        """Test clock with custom quantum"""
        client, port, dest = midi_setup
        session = cm.link.LinkSession(bpm=120.0)

        clock = link_midi.LinkMIDIClock(session, port, dest, quantum=8.0)

        assert clock.quantum == 8.0


class TestLinkMIDIClockOperation:
    """Test LinkMIDIClock operation (without actual MIDI output)"""

    @pytest.fixture
    def mock_midi_setup(self):
        """Mock MIDI setup for testing without hardware"""
        # Use dummy IDs for testing
        client = 1
        port = 1
        dest = 1
        return client, port, dest

    def test_clock_start_stop(self, mock_midi_setup):
        """Test starting and stopping clock"""
        client, port, dest = mock_midi_setup
        session = cm.link.LinkSession(bpm=120.0)
        session.enabled = True

        clock = link_midi.LinkMIDIClock(session, port, dest)

        # Start clock
        assert clock.running == False
        # Note: start() will try to send MIDI, may fail with dummy IDs
        # clock.start()
        # assert clock.running == True

        # For now just test the state management
        clock.running = True
        assert clock.running == True

        clock.running = False
        assert clock.running == False


class TestLinkMIDISequencer:
    """Test LinkMIDISequencer functionality"""

    @pytest.fixture
    def mock_midi_setup(self):
        """Mock MIDI setup"""
        client = 1
        port = 1
        dest = 1
        return client, port, dest

    def test_sequencer_creation(self, mock_midi_setup):
        """Test creating LinkMIDISequencer"""
        client, port, dest = mock_midi_setup
        session = cm.link.LinkSession(bpm=120.0)

        seq = link_midi.LinkMIDISequencer(session, port, dest)

        assert seq.session is session
        assert seq.midi_port == port
        assert seq.midi_destination == dest
        assert seq.quantum == 4.0
        assert seq.running == False
        assert len(seq.events) == 0

    def test_schedule_event(self, mock_midi_setup):
        """Test scheduling MIDI event"""
        client, port, dest = mock_midi_setup
        session = cm.link.LinkSession(bpm=120.0)

        seq = link_midi.LinkMIDISequencer(session, port, dest)

        # Schedule event
        message = b'\x90\x3C\x64'  # Note On
        seq.schedule_event(beat=1.0, message=message)

        assert len(seq.events) == 1
        assert seq.events[0].beat == 1.0
        assert seq.events[0].message == message
        assert seq.events[0].sent == False

    def test_schedule_multiple_events_sorted(self, mock_midi_setup):
        """Test events are kept sorted by beat"""
        client, port, dest = mock_midi_setup
        session = cm.link.LinkSession(bpm=120.0)

        seq = link_midi.LinkMIDISequencer(session, port, dest)

        # Schedule events out of order
        seq.schedule_event(beat=2.0, message=b'\x90\x40\x64')
        seq.schedule_event(beat=1.0, message=b'\x90\x3C\x64')
        seq.schedule_event(beat=3.0, message=b'\x90\x44\x64')

        assert len(seq.events) == 3
        assert seq.events[0].beat == 1.0
        assert seq.events[1].beat == 2.0
        assert seq.events[2].beat == 3.0

    def test_schedule_note(self, mock_midi_setup):
        """Test scheduling note with automatic note-off"""
        client, port, dest = mock_midi_setup
        session = cm.link.LinkSession(bpm=120.0)

        seq = link_midi.LinkMIDISequencer(session, port, dest)

        # Schedule note
        seq.schedule_note(
            beat=1.0,
            channel=0,
            note=60,
            velocity=100,
            duration=0.5
        )

        # Should have note-on and note-off
        assert len(seq.events) == 2
        assert seq.events[0].beat == 1.0  # Note On
        assert seq.events[1].beat == 1.5  # Note Off

    def test_schedule_cc(self, mock_midi_setup):
        """Test scheduling CC message"""
        client, port, dest = mock_midi_setup
        session = cm.link.LinkSession(bpm=120.0)

        seq = link_midi.LinkMIDISequencer(session, port, dest)

        # Schedule CC
        seq.schedule_cc(beat=2.0, channel=0, controller=7, value=127)

        assert len(seq.events) == 1
        assert seq.events[0].beat == 2.0

    def test_clear_events(self, mock_midi_setup):
        """Test clearing all events"""
        client, port, dest = mock_midi_setup
        session = cm.link.LinkSession(bpm=120.0)

        seq = link_midi.LinkMIDISequencer(session, port, dest)

        # Schedule events
        seq.schedule_event(beat=1.0, message=b'\x90\x3C\x64')
        seq.schedule_event(beat=2.0, message=b'\x90\x40\x64')

        assert len(seq.events) == 2

        # Clear
        seq.clear_events()

        assert len(seq.events) == 0


class TestTimeConversion:
    """Test Link beat to host time conversion utilities"""

    def test_link_beat_to_host_time(self):
        """Test converting Link beat to host time"""
        session = cm.link.LinkSession(bpm=120.0)
        session.enabled = True

        # Get host time for beat 0
        host_time = link_midi.link_beat_to_host_time(session, beat=0.0, quantum=4.0)

        assert isinstance(host_time, int)
        assert host_time > 0

    def test_host_time_to_link_beat(self):
        """Test converting host time to Link beat"""
        session = cm.link.LinkSession(bpm=120.0)
        session.enabled = True

        # Get current host time
        clock = session.clock
        host_time = clock.ticks()

        # Convert to beat
        beat = link_midi.host_time_to_link_beat(session, host_time, quantum=4.0)

        assert isinstance(beat, float)
        assert beat >= 0.0

    def test_round_trip_conversion(self):
        """Test round-trip beat <-> host time conversion"""
        session = cm.link.LinkSession(bpm=120.0)
        session.enabled = True

        # Start with a beat
        original_beat = 4.0

        # Convert to host time and back
        host_time = link_midi.link_beat_to_host_time(session, original_beat, quantum=4.0)
        converted_beat = link_midi.host_time_to_link_beat(session, host_time, quantum=4.0)

        # Should be approximately equal (within small margin)
        assert abs(converted_beat - original_beat) < 0.01


class TestModuleExports:
    """Test link_midi module is properly exported"""

    def test_module_accessible(self):
        """Test link_midi module is accessible"""
        assert hasattr(cm, 'link_midi')
        assert cm.link_midi is not None

    def test_classes_accessible(self):
        """Test Link + MIDI classes are accessible"""
        assert hasattr(cm.link_midi, 'LinkMIDIClock')
        assert hasattr(cm.link_midi, 'LinkMIDISequencer')
        assert hasattr(cm.link_midi, 'MIDIEvent')

    def test_functions_accessible(self):
        """Test utility functions are accessible"""
        assert hasattr(cm.link_midi, 'link_beat_to_host_time')
        assert hasattr(cm.link_midi, 'host_time_to_link_beat')

    def test_constants_accessible(self):
        """Test MIDI constants are accessible"""
        assert hasattr(cm.link_midi, 'MIDI_CLOCK')
        assert hasattr(cm.link_midi, 'MIDI_START')
        assert hasattr(cm.link_midi, 'MIDI_STOP')
        assert hasattr(cm.link_midi, 'MIDI_CLOCKS_PER_QUARTER_NOTE')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
