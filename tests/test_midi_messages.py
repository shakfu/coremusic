#!/usr/bin/env python3
"""Tests for the MIDI channel voice message builders."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from coremusic import capi
from coremusic.constants import MIDIControlChange, MIDIStatus
from coremusic.midi import (
    DEFAULT_VELOCITY,
    PITCH_BEND_CENTER,
    PITCH_BEND_MAX,
    MIDIEvent,
    all_notes_off,
    all_sound_off,
    channel_aftertouch,
    control_change,
    note_off,
    note_on,
    pitch_bend,
    poly_aftertouch,
    program_change,
)
from coremusic.music.theory import Note, note_name_to_midi

# Every builder, as (callable, minimal positional args) for the shared checks.
BUILDERS = [
    (note_on, (60, 100)),
    (note_off, (60, 0)),
    (control_change, (7, 100)),
    (program_change, (48,)),
    (pitch_bend, (8192,)),
    (poly_aftertouch, (60, 64)),
    (channel_aftertouch, (64,)),
    (all_notes_off, ()),
    (all_sound_off, ()),
]


class TestExactBytes:
    """Each builder against a hand-verified byte string."""

    def test_note_on(self):
        assert note_on(60, 100) == b"\x90\x3c\x64"

    def test_note_off(self):
        assert note_off(60) == b"\x80\x3c\x00"

    def test_note_off_release_velocity(self):
        assert note_off(60, 64) == b"\x80\x3c\x40"

    def test_control_change(self):
        assert control_change(7, 100) == b"\xb0\x07\x64"

    def test_program_change_has_no_second_data_byte(self):
        assert program_change(48) == b"\xc0\x30"
        assert len(program_change(0)) == 2

    def test_poly_aftertouch(self):
        assert poly_aftertouch(60, 64) == b"\xa0\x3c\x40"

    def test_channel_aftertouch_has_no_second_data_byte(self):
        assert channel_aftertouch(64) == b"\xd0\x40"
        assert len(channel_aftertouch(0)) == 2

    def test_all_notes_off_is_cc_123(self):
        assert all_notes_off() == b"\xb0\x7b\x00"

    def test_all_sound_off_is_cc_120(self):
        assert all_sound_off() == b"\xb0\x78\x00"

    def test_defaults(self):
        """Documented defaults, so a change to one is a test failure."""
        assert note_on(60) == b"\x90\x3c\x40"  # velocity 64
        assert note_off(60) == b"\x80\x3c\x00"  # velocity 0
        assert pitch_bend() == b"\xe0\x00\x40"  # centred


class TestNoteResolution:
    """A note may be a MIDI number, a name, or a Note object."""

    NOTE_TAKING = [note_on, note_off, poly_aftertouch]

    @pytest.mark.parametrize(
        ("name", "number"),
        [
            ("C-1", 0),
            ("C4", 60),
            ("C#4", 61),
            ("Db4", 61),
            ("F#3", 54),
            ("A4", 69),
            ("G9", 127),
        ],
    )
    def test_name_resolves_to_midi_number(self, name, number):
        assert note_on(name, 100) == note_on(number, 100)

    def test_middle_c_and_a440(self):
        """The two anchors of scientific pitch notation."""
        assert note_on("C4", 100)[1] == 60
        assert note_on("A4", 100)[1] == 69

    def test_name_agrees_with_music_theory(self):
        """The builders must not have their own opinion about note names."""
        for name in ("C-1", "C4", "F#3", "Bb2", "G9"):
            assert note_on(name, 100)[1] == note_name_to_midi(name)

    def test_lowercase_name(self):
        assert note_on("c4", 100) == note_on(60, 100)

    def test_note_object_resolves(self):
        assert note_on(Note("C", 4), 100) == note_on(60, 100)

    def test_all_three_forms_agree(self):
        assert note_on(60, 100) == note_on("C4", 100) == note_on(Note("C", 4), 100)

    @pytest.mark.parametrize("fn", NOTE_TAKING)
    def test_every_note_taking_builder_accepts_all_forms(self, fn):
        assert fn(60, 64) == fn("C4", 64) == fn(Note("C", 4), 64)

    def test_rejects_unparseable_name(self):
        with pytest.raises(ValueError, match="Invalid note name"):
            note_on("H4")

    def test_rejects_name_outside_midi_range(self):
        with pytest.raises(ValueError, match="out of range"):
            note_on("B9")

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError, match="Invalid note name"):
            note_on("")


class TestVelocityResolution:
    """Where the velocity comes from when the caller omits it."""

    def test_number_uses_default_velocity(self):
        assert note_on(60)[2] == DEFAULT_VELOCITY

    def test_name_uses_default_velocity(self):
        assert note_on("C4")[2] == DEFAULT_VELOCITY

    def test_note_contributes_its_own_velocity(self):
        assert note_on(Note("C", 4, velocity=80))[2] == 80

    def test_note_default_velocity_is_not_the_builder_default(self):
        """Note defaults to 100, the builder to 64; a Note must win.

        This is the documented wart of accepting Note: the effective default
        depends on the argument type.
        """
        assert Note("C", 4).velocity == 100
        assert DEFAULT_VELOCITY == 64
        assert note_on(Note("C", 4))[2] == 100
        assert note_on(60)[2] == 64

    def test_explicit_velocity_beats_note_velocity(self):
        assert note_on(Note("C", 4, velocity=80), 20)[2] == 20

    def test_explicit_zero_velocity_beats_note_velocity(self):
        """0 is falsy; it must still override rather than fall through."""
        assert note_on(Note("C", 4, velocity=80), 0)[2] == 0

    def test_note_off_ignores_note_velocity(self):
        """Note.velocity is attack; note_off carries release velocity."""
        assert note_off(Note("C", 4, velocity=80))[2] == 0

    def test_poly_aftertouch_ignores_note_velocity(self):
        """Pressure is not velocity, so it is always explicit."""
        assert poly_aftertouch(Note("C", 4, velocity=80), 64)[2] == 64


class TestChannel:
    """Channel handling, including the argument-order footgun."""

    @pytest.mark.parametrize("channel", range(16))
    def test_channel_occupies_low_nibble(self, channel):
        data = note_on(60, 100, channel=channel)
        assert data[0] == MIDIStatus.NOTE_ON | channel
        assert data[0] & 0x0F == channel
        assert data[0] & 0xF0 == MIDIStatus.NOTE_ON

    def test_channel_does_not_disturb_data_bytes(self):
        assert note_on(0x15, 0x45, channel=2) == b"\x92\x15\x45"

    @pytest.mark.parametrize(("fn", "args"), BUILDERS)
    def test_channel_is_keyword_only(self, fn, args):
        """A positional channel must fail loudly.

        capi.midi_note_on is (channel, note, velocity), so if channel were
        positional here, note_on(0, 60, 100) would build note 0 at velocity 60
        and send perfectly valid, completely wrong MIDI.
        """
        with pytest.raises(TypeError):
            fn(*args, 0)

    @pytest.mark.parametrize(("fn", "args"), BUILDERS)
    def test_channel_defaults_to_zero(self, fn, args):
        assert fn(*args) == fn(*args, channel=0)

    @pytest.mark.parametrize(("fn", "args"), BUILDERS)
    def test_rejects_out_of_range_channel(self, fn, args):
        for bad in (-1, 16, 255):
            with pytest.raises(ValueError, match="channel must be 0-15"):
                fn(*args, channel=bad)


class TestValidation:
    """Out-of-range data must raise, not silently wrap.

    MIDIEvent.to_bytes masks with & 0x7F, so a velocity of 200 becomes 72 with
    no indication. A data byte above 127 has its high bit set and reads as a
    status byte, desynchronising everything after it.
    """

    def test_rejects_velocity_above_127(self):
        with pytest.raises(ValueError, match="velocity must be 0-127"):
            note_on(60, 200)

    def test_rejects_negative_note(self):
        with pytest.raises(ValueError, match="note must be 0-127"):
            note_on(-1)

    def test_rejects_note_above_127(self):
        with pytest.raises(ValueError, match="note must be 0-127"):
            note_on(128)

    def test_rejects_controller_above_127(self):
        with pytest.raises(ValueError, match="controller must be 0-127"):
            control_change(200, 0)

    def test_rejects_program_above_127(self):
        with pytest.raises(ValueError, match="program must be 0-127"):
            program_change(128)

    def test_rejects_pressure_above_127(self):
        with pytest.raises(ValueError, match="pressure must be 0-127"):
            channel_aftertouch(128)
        with pytest.raises(ValueError, match="pressure must be 0-127"):
            poly_aftertouch(60, 128)

    def test_error_names_the_offending_argument(self):
        """The message must identify which argument is wrong."""
        with pytest.raises(ValueError, match=r"velocity must be 0-127, got 200"):
            note_on(60, 200)

    def test_rejects_non_int(self):
        with pytest.raises(TypeError, match="note must be an int"):
            note_on(60.5)
        with pytest.raises(TypeError, match="velocity must be an int"):
            note_on(60, "loud")

    def test_rejects_bool(self):
        """bool is an int subclass; True as a note number is a mistake."""
        with pytest.raises(TypeError, match="note must be an int"):
            note_on(True)

    @pytest.mark.parametrize(("fn", "args"), BUILDERS)
    def test_no_builder_emits_a_byte_above_127_in_data(self, fn, args):
        """Only the status byte may have its high bit set."""
        data = fn(*args, channel=15)
        assert data[0] & 0x80
        assert all(b <= 0x7F for b in data[1:])


class TestPitchBend:
    """Pitch bend is the only 14-bit channel voice message."""

    def test_centre_constant(self):
        assert PITCH_BEND_CENTER == 8192
        assert PITCH_BEND_MAX == 16383

    def test_centre_is_lsb_0_msb_64(self):
        assert pitch_bend(PITCH_BEND_CENTER) == b"\xe0\x00\x40"

    def test_minimum(self):
        assert pitch_bend(0) == b"\xe0\x00\x00"

    def test_maximum(self):
        assert pitch_bend(PITCH_BEND_MAX) == b"\xe0\x7f\x7f"

    @pytest.mark.parametrize("value", [0, 1, 127, 128, 8191, 8192, 8193, 16382, 16383])
    def test_lsb_msb_split_reassembles(self, value):
        """Reassembling the two 7-bit halves must give the original value."""
        data = pitch_bend(value)
        assert data[1] | (data[2] << 7) == value

    def test_rejects_above_14_bit_range(self):
        with pytest.raises(ValueError, match="value must be 0-16383"):
            pitch_bend(16384)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="value must be 0-16383"):
            pitch_bend(-1)


class TestAgreementWithCapiTuples:
    """The builders must agree with capi on values, but not on length.

    capi.midi_* returns a fixed (status, data1, data2) triple for
    music_device_midi_event(), the AudioUnit MusicDevice call, whose data2 is
    "0 if not needed". These builders return exact wire bytes, where Program
    Change and Channel Aftertouch are two bytes and a third would be read as a
    stray data byte. The two are not interchangeable, and bytes() over a capi
    triple is not a valid way to build a message.
    """

    @pytest.mark.parametrize(
        ("built", "triple"),
        [
            (note_on(60, 100, channel=3), capi.midi_note_on(3, 60, 100)),
            (note_off(60, 0, channel=3), capi.midi_note_off(3, 60, 0)),
            (control_change(7, 100, channel=3), capi.midi_control_change(3, 7, 100)),
            (pitch_bend(12000, channel=3), capi.midi_pitch_bend(3, 12000)),
        ],
    )
    def test_three_byte_messages_match_capi_exactly(self, built, triple):
        assert built == bytes(triple)

    @pytest.mark.parametrize(
        ("built", "triple"),
        [
            (program_change(48, channel=3), capi.midi_program_change(3, 48)),
        ],
    )
    def test_two_byte_messages_match_capi_on_the_bytes_that_exist(self, built, triple):
        """Status and data1 must agree; capi's padding data2 has no wire byte."""
        assert len(built) == 2
        assert built[0] == triple[0]
        assert built[1] == triple[1]
        assert triple[2] == 0  # padding, not a byte to send

    def test_capi_triple_is_not_a_wire_message(self):
        """Guards the distinction, so neither side is 'fixed' into the other.

        Sending bytes(capi.midi_program_change(...)) would append a 0x00 that a
        receiver reads as data for a running-status message.
        """
        assert len(bytes(capi.midi_program_change(3, 48))) == 3
        assert len(program_change(48, channel=3)) == 2

    @pytest.mark.parametrize(
        ("data", "status", "channel", "d1", "d2"),
        [
            (note_on(60, 100, channel=5), MIDIStatus.NOTE_ON, 5, 60, 100),
            (note_off(60, 0, channel=5), MIDIStatus.NOTE_OFF, 5, 60, 0),
            (control_change(7, 100, channel=5), MIDIStatus.CONTROL_CHANGE, 5, 7, 100),
            (poly_aftertouch(60, 64, channel=5), MIDIStatus.POLY_AFTERTOUCH, 5, 60, 64),
        ],
    )
    def test_round_trips_through_midi_event(self, data, status, channel, d1, d2):
        """MIDIEvent.from_bytes must decode what the builders encode."""
        event = MIDIEvent.from_bytes(data)
        assert event.status == status
        assert event.channel == channel
        assert event.data1 == d1
        assert event.data2 == d2
        assert event.to_bytes() == data

    def test_note_on_is_detected_as_note_on(self):
        assert MIDIEvent.from_bytes(note_on(60, 100)).is_note_on is True

    def test_zero_velocity_note_on_is_detected_as_note_off(self):
        """The running-status convention: Note On at velocity 0 means off."""
        assert MIDIEvent.from_bytes(note_on(60, 0)).is_note_off is True

    def test_all_notes_off_uses_the_named_constant(self):
        assert all_notes_off()[1] == MIDIControlChange.ALL_NOTES_OFF
        assert all_sound_off()[1] == MIDIControlChange.ALL_SOUND_OFF

    def test_status_nibbles_match_the_enum(self):
        """Guards against a builder drifting from MIDIStatus."""
        expected = {
            note_on(60, 100): MIDIStatus.NOTE_ON,
            note_off(60): MIDIStatus.NOTE_OFF,
            poly_aftertouch(60, 64): MIDIStatus.POLY_AFTERTOUCH,
            control_change(7, 100): MIDIStatus.CONTROL_CHANGE,
            program_change(48): MIDIStatus.PROGRAM_CHANGE,
            channel_aftertouch(64): MIDIStatus.CHANNEL_AFTERTOUCH,
            pitch_bend(8192): MIDIStatus.PITCH_BEND,
        }
        for data, status in expected.items():
            assert data[0] & 0xF0 == status


class TestSplitterCompatibility:
    """Built messages must survive the receive path."""

    def test_splitter_recovers_concatenated_messages(self):
        from coremusic.midi import split_midi_messages

        stream = note_on(60, 100) + note_off(60) + program_change(48)
        assert split_midi_messages(stream) == [
            note_on(60, 100),
            note_off(60),
            program_change(48),
        ]

    def test_splitter_handles_two_byte_messages(self):
        """Program Change and Channel Aftertouch are two bytes, not three."""
        from coremusic.midi import split_midi_messages

        stream = program_change(5) + channel_aftertouch(64) + note_on(60, 100)
        assert split_midi_messages(stream) == [
            program_change(5),
            channel_aftertouch(64),
            note_on(60, 100),
        ]


class TestReturnType:
    def test_returns_bytes_not_bytearray(self):
        """send_data and midi_send_data both take bytes."""
        for fn, args in BUILDERS:
            assert type(fn(*args)) is bytes
