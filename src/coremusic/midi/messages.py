#!/usr/bin/env python3
"""Builders for MIDI channel voice messages.

Every function returns the ``bytes`` accepted by :meth:`MIDIOutputPort.send_data`
and :func:`coremusic.capi.midi_send_data`, so a send reads as what it plays::

    port.send_data(destination, note_on("C4", 100))

Anywhere a note is taken it may be a MIDI number, a note name, or a
:class:`~coremusic.music.theory.Note`::

    note_on(60)            # MIDI number
    note_on("C4")          # name, scientific pitch notation
    note_on("F#3")         # sharps and flats both parse
    note_on(Note("C", 4))  # from coremusic.music.theory

Octave numbering follows scientific pitch notation, matching
:func:`~coremusic.music.theory.note_name_to_midi`: middle C is ``"C4"`` is 60,
and A440 is ``"A4"`` is 69. Ableton Live and Logic display that same note as
C3, and Cakewalk as C5, so when matching what a DAW shows on screen, prefer the
MIDI number, which is unambiguous.

Channel is keyword-only and defaults to 0. That is deliberate:
:func:`coremusic.capi.midi_note_on` takes ``(channel, note, velocity)`` and
returns a tuple, so a positional third argument here would silently build a
different message than it appears to. ``note_on(0, 60, 100)`` raises
``TypeError`` instead of quietly sending note 0 at velocity 60.

The ``capi.midi_*`` helpers are a different thing and are not interchangeable
with these. They return a fixed ``(status, data1, data2)`` triple for
:func:`coremusic.capi.music_device_midi_event`, the AudioUnit MusicDevice call,
whose ``data2`` is "0 if not needed". Program Change and Channel Aftertouch are
two bytes on the wire, so ``bytes(capi.midi_program_change(...))`` appends a
``0x00`` that a receiver reads as data for a running-status message. Use these
builders for anything sent through CoreMIDI.

Out-of-range arguments raise ``ValueError``. :meth:`MIDIEvent.to_bytes` masks
instead (``velocity & 0x7F``), which turns a velocity of 200 into 72 with no
indication; a data byte that wraps past 127 becomes a status byte and corrupts
the rest of the stream, so it is worth catching at the call.
"""

from __future__ import annotations

from ..constants.midi import MIDIControlChange, MIDIStatus
from ..music.theory import Note, note_name_to_midi

#: A note as a MIDI number, a name such as ``"C4"``, or a :class:`Note`.
#: Requires Python 3.10 at runtime, which is this package's minimum.
NoteLike = int | str | Note

__all__ = [
    "ALL_NOTES_OFF",
    "ALL_SOUND_OFF",
    "DEFAULT_VELOCITY",
    "PITCH_BEND_CENTER",
    "PITCH_BEND_MAX",
    "NoteLike",
    "all_notes_off",
    "all_sound_off",
    "channel_aftertouch",
    "control_change",
    "note_off",
    "note_on",
    "pitch_bend",
    "poly_aftertouch",
    "program_change",
]

#: Pitch bend value that means "no bend"; the 14-bit range is 0-16383.
PITCH_BEND_CENTER = 8192
#: Largest pitch bend value.
PITCH_BEND_MAX = 16383

#: Control numbers for the two panic messages, re-exported for convenience.
ALL_SOUND_OFF = MIDIControlChange.ALL_SOUND_OFF
ALL_NOTES_OFF = MIDIControlChange.ALL_NOTES_OFF


def _check(name: str, value: int, high: int) -> int:
    """Return `value`, or raise ValueError naming the argument that is wrong."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if not 0 <= value <= high:
        raise ValueError(f"{name} must be 0-{high}, got {value}")
    return value


def _status(kind: MIDIStatus, channel: int) -> int:
    """Combine a status nibble with a channel into a status byte."""
    return int(kind) | _check("channel", channel, 15)


#: Velocity used when the caller gives none and the note carries none.
DEFAULT_VELOCITY = 64


def _resolve_note(note: NoteLike) -> tuple[int, int | None]:
    """Resolve a NoteLike to (midi_number, velocity_or_None).

    The second element is the velocity the note carries, which only a
    :class:`Note` has. It is None for a number or a name, meaning the caller's
    velocity (or :data:`DEFAULT_VELOCITY`) applies.
    """
    if isinstance(note, Note):
        return _check("note", note.midi, 127), note.velocity
    if isinstance(note, str):
        # note_name_to_midi raises ValueError naming the bad input already.
        return note_name_to_midi(note), None
    return _check("note", note, 127), None


def note_on(note: NoteLike, velocity: int | None = None, *, channel: int = 0) -> bytes:
    """Build a Note On message.

    A velocity of 0 is a Note Off by convention; :func:`note_off` is clearer.

    Args:
        note: MIDI number 0-127, a name such as ``"C4"`` or ``"F#3"``, or a
            :class:`~coremusic.music.theory.Note`. Middle C is 60 / ``"C4"``.
        velocity: Attack velocity, 0-127. When omitted, a :class:`Note`
            contributes its own velocity and anything else gets
            :data:`DEFAULT_VELOCITY`. An explicit velocity always wins.
        channel: MIDI channel, 0-15.

    Returns:
        Three bytes: status, note, velocity.

    Example:
        >>> note_on("C4", 100)
        b'\\x90<d'
        >>> note_on(Note("C", 4, velocity=80)) == note_on(60, 80)
        True
    """
    number, note_velocity = _resolve_note(note)
    if velocity is None:
        velocity = DEFAULT_VELOCITY if note_velocity is None else note_velocity
    return bytes(
        [
            _status(MIDIStatus.NOTE_ON, channel),
            number,
            _check("velocity", velocity, 127),
        ]
    )


def note_off(note: NoteLike, velocity: int = 0, *, channel: int = 0) -> bytes:
    """Build a Note Off message.

    Unlike :func:`note_on`, a :class:`Note` does not contribute its velocity
    here: ``Note.velocity`` is an attack velocity, and this field is release
    velocity, which is a different quantity and defaults to 0.

    Args:
        note: MIDI number 0-127, a name such as ``"C4"``, or a
            :class:`~coremusic.music.theory.Note`.
        velocity: Release velocity, 0-127. Most receivers ignore it.
        channel: MIDI channel, 0-15.

    Returns:
        Three bytes: status, note, release velocity.
    """
    number, _ = _resolve_note(note)
    return bytes(
        [
            _status(MIDIStatus.NOTE_OFF, channel),
            number,
            _check("velocity", velocity, 127),
        ]
    )


def control_change(controller: int, value: int, *, channel: int = 0) -> bytes:
    """Build a Control Change message.

    :class:`~coremusic.constants.MIDIControlChange` names the common controller
    numbers, so ``control_change(MIDIControlChange.VOLUME, 100)`` beats
    ``control_change(7, 100)``.

    Args:
        controller: Controller number, 0-127.
        value: Controller value, 0-127.
        channel: MIDI channel, 0-15.

    Returns:
        Three bytes: status, controller, value.
    """
    return bytes(
        [
            _status(MIDIStatus.CONTROL_CHANGE, channel),
            _check("controller", controller, 127),
            _check("value", value, 127),
        ]
    )


def program_change(program: int, *, channel: int = 0) -> bytes:
    """Build a Program Change message.

    Args:
        program: Program number, 0-127.
        channel: MIDI channel, 0-15.

    Returns:
        Two bytes: status, program. Program Change carries no second data byte.
    """
    return bytes(
        [_status(MIDIStatus.PROGRAM_CHANGE, channel), _check("program", program, 127)]
    )


def pitch_bend(value: int = PITCH_BEND_CENTER, *, channel: int = 0) -> bytes:
    """Build a Pitch Bend message.

    Pitch bend is the one channel voice message with a 14-bit value, split
    across two 7-bit bytes least-significant first.

    Args:
        value: Bend amount, 0-16383. :data:`PITCH_BEND_CENTER` (8192) is no
            bend, 0 is fully down, :data:`PITCH_BEND_MAX` is fully up.
        channel: MIDI channel, 0-15.

    Returns:
        Three bytes: status, value LSB, value MSB.
    """
    _check("value", value, PITCH_BEND_MAX)
    return bytes(
        [_status(MIDIStatus.PITCH_BEND, channel), value & 0x7F, (value >> 7) & 0x7F]
    )


def poly_aftertouch(note: NoteLike, pressure: int, *, channel: int = 0) -> bytes:
    """Build a Polyphonic Aftertouch message, which pressures one held note.

    Args:
        note: MIDI number 0-127, a name such as ``"C4"``, or a
            :class:`~coremusic.music.theory.Note`.
        pressure: Pressure, 0-127. A note's own velocity does not apply here.
        channel: MIDI channel, 0-15.

    Returns:
        Three bytes: status, note, pressure.
    """
    number, _ = _resolve_note(note)
    return bytes(
        [
            _status(MIDIStatus.POLY_AFTERTOUCH, channel),
            number,
            _check("pressure", pressure, 127),
        ]
    )


def channel_aftertouch(pressure: int, *, channel: int = 0) -> bytes:
    """Build a Channel Aftertouch message, which pressures every held note.

    Args:
        pressure: Pressure, 0-127.
        channel: MIDI channel, 0-15.

    Returns:
        Two bytes: status, pressure.
    """
    return bytes(
        [
            _status(MIDIStatus.CHANNEL_AFTERTOUCH, channel),
            _check("pressure", pressure, 127),
        ]
    )


def all_notes_off(*, channel: int = 0) -> bytes:
    """Build the All Notes Off message (CC 123).

    Releases held notes as if each key were let go, so anything with a release
    tail still rings out. Use :func:`all_sound_off` to cut the sound dead.

    Args:
        channel: MIDI channel, 0-15.

    Returns:
        Three bytes: status, 123, 0.
    """
    return control_change(MIDIControlChange.ALL_NOTES_OFF, 0, channel=channel)


def all_sound_off(*, channel: int = 0) -> bytes:
    """Build the All Sound Off message (CC 120).

    Silences the channel immediately, ignoring release envelopes.

    Args:
        channel: MIDI channel, 0-15.

    Returns:
        Three bytes: status, 120, 0.
    """
    return control_change(MIDIControlChange.ALL_SOUND_OFF, 0, channel=channel)
