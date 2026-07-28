#!/usr/bin/env python3
"""Tutorial: MIDI Basics

This module demonstrates MIDI operations with coremusic.
All examples are executable doctests.

Note: Some MIDI operations require hardware. Tests that need
actual MIDI devices are skipped when no devices are available.

Run with: pytest tests/tutorials/test_midi_basics.py --doctest-modules -v
"""

from __future__ import annotations


def get_midi_device_counts():
    """Get counts of MIDI devices, sources, and destinations.

    >>> import coremusic.capi as capi
    >>> num_devices = capi.midi_get_number_of_devices()
    >>> num_sources = capi.midi_get_number_of_sources()
    >>> num_destinations = capi.midi_get_number_of_destinations()
    >>> # These should be non-negative integers
    >>> assert isinstance(num_devices, int)
    >>> assert isinstance(num_sources, int)
    >>> assert isinstance(num_destinations, int)
    >>> assert num_devices >= 0
    >>> assert num_sources >= 0
    >>> assert num_destinations >= 0
    """


def create_midi_client():
    """Create a MIDI client.

    >>> from coremusic.midi import MIDIClient
    >>> client = MIDIClient("Test Client")
    >>> assert client is not None
    >>> # Always dispose when done
    >>> client.dispose()
    """


def create_midi_output_port():
    """Create a MIDI output port.

    >>> from coremusic.midi import MIDIClient
    >>> client = MIDIClient("Test Client")
    >>> try:
    ...     output_port = client.create_output_port("Test Output")
    ...     assert output_port is not None
    ... finally:
    ...     client.dispose()
    """


def create_midi_input_port():
    """Create a MIDI input port.

    >>> from coremusic.midi import MIDIClient
    >>> client = MIDIClient("Test Client")
    >>> try:
    ...     input_port = client.create_input_port("Test Input")
    ...     assert input_port is not None
    ... finally:
    ...     client.dispose()
    """


def midi_note_on_message():
    """Construct a MIDI Note On message.

    MIDI Note On format: [0x90 + channel, note, velocity]
    - Status byte: 0x90 (Note On) + channel (0-15)
    - Note: 0-127 (60 = Middle C)
    - Velocity: 0-127

    Build one with note_on() rather than by hand; the note may be a number,
    a name, or a Note.

    >>> from coremusic.midi import note_on
    >>> # Note On: Middle C (60), velocity 100, channel 0
    >>> message = note_on("C4", 100)
    >>> assert len(message) == 3
    >>> assert message[0] == 0x90  # Note On, channel 0
    >>> assert message[1] == 60    # Middle C
    >>> assert message[2] == 100   # Velocity

    >>> # A number, a name and a Note are interchangeable
    >>> from coremusic.music.theory import Note
    >>> assert note_on(60, 100) == note_on("C4", 100) == note_on(Note("C", 4), 100)

    >>> # Channel is keyword-only, so it cannot be confused with a note
    >>> assert note_on("C4", 100, channel=3)[0] == 0x93
    """


def midi_note_off_message():
    """Construct a MIDI Note Off message.

    MIDI Note Off format: [0x80 + channel, note, velocity]
    - Status byte: 0x80 (Note Off) + channel (0-15)
    - Note: 0-127
    - Velocity: typically 0

    >>> from coremusic.midi import note_off
    >>> # Note Off: Middle C (60), channel 0
    >>> message = note_off("C4")
    >>> assert len(message) == 3
    >>> assert message[0] == 0x80  # Note Off, channel 0
    >>> assert message[1] == 60    # Middle C
    >>> assert message[2] == 0     # Velocity (release)
    """


def midi_control_change_message():
    """Construct a MIDI Control Change (CC) message.

    MIDI CC format: [0xB0 + channel, controller, value]
    - Status byte: 0xB0 (CC) + channel (0-15)
    - Controller: 0-127 (e.g., 1=mod wheel, 7=volume, 10=pan)
    - Value: 0-127

    >>> from coremusic.constants import MIDIControlChange
    >>> from coremusic.midi import control_change
    >>> # CC: Modulation wheel to 64 (50%), channel 0
    >>> mod_wheel = control_change(MIDIControlChange.MODULATION, 64)
    >>> assert len(mod_wheel) == 3
    >>> assert mod_wheel[0] == 0xB0  # CC, channel 0
    >>> assert mod_wheel[1] == 1     # Mod wheel
    >>> assert mod_wheel[2] == 64    # Value (50%)

    >>> # MIDIControlChange names the common controller numbers
    >>> assert MIDIControlChange.MODULATION == 1
    >>> assert MIDIControlChange.VOLUME == 7
    >>> assert MIDIControlChange.PAN == 10
    >>> assert MIDIControlChange.SUSTAIN_PEDAL == 64
    >>> assert MIDIControlChange.ALL_NOTES_OFF == 123

    >>> # The two panic messages have their own helpers
    >>> from coremusic.midi import all_notes_off, all_sound_off
    >>> assert all_notes_off() == control_change(123, 0)
    >>> assert all_sound_off() == control_change(120, 0)
    """


def midi_program_change_message():
    """Construct a MIDI Program Change message.

    MIDI Program Change format: [0xC0 + channel, program]
    - Status byte: 0xC0 (Program Change) + channel (0-15)
    - Program: 0-127

    Note the length: Program Change carries a single data byte, so the message
    is two bytes, not three.

    >>> from coremusic.midi import program_change
    >>> # Program Change: Select program 0 (piano), channel 0
    >>> message = program_change(0)
    >>> assert len(message) == 2
    >>> assert message[0] == 0xC0  # Program Change, channel 0
    >>> assert message[1] == 0     # Program 0
    """


def midi_pitch_bend_message():
    """Construct a MIDI Pitch Bend message.

    MIDI Pitch Bend format: [0xE0 + channel, LSB, MSB]
    - Status byte: 0xE0 (Pitch Bend) + channel (0-15)
    - Value: 14-bit (0-16383), center = 8192
    - LSB: value & 0x7F
    - MSB: (value >> 7) & 0x7F

    pitch_bend() does the LSB/MSB split for you.

    >>> from coremusic.midi import PITCH_BEND_CENTER, PITCH_BEND_MAX, pitch_bend
    >>> # Pitch Bend: Center position (no bend)
    >>> message = pitch_bend(PITCH_BEND_CENTER)
    >>> assert len(message) == 3
    >>> assert message[0] == 0xE0  # Pitch Bend, channel 0

    >>> # Reconstruct value
    >>> reconstructed = message[1] | (message[2] << 7)
    >>> assert reconstructed == PITCH_BEND_CENTER == 8192

    >>> # The extremes of the 14-bit range
    >>> assert pitch_bend(0) == b"\\xe0\\x00\\x00"
    >>> assert pitch_bend(PITCH_BEND_MAX) == b"\\xe0\\x7f\\x7f"
    """


def parse_midi_status_byte():
    """Parse MIDI status byte to get message type and channel.

    >>> status = 0x92  # Note On, channel 2
    >>> message_type = status & 0xF0
    >>> channel = status & 0x0F
    >>> assert message_type == 0x90  # Note On
    >>> assert channel == 2

    >>> # Message type constants
    >>> NOTE_OFF = 0x80
    >>> NOTE_ON = 0x90
    >>> POLY_AFTERTOUCH = 0xA0
    >>> CONTROL_CHANGE = 0xB0
    >>> PROGRAM_CHANGE = 0xC0
    >>> CHANNEL_AFTERTOUCH = 0xD0
    >>> PITCH_BEND = 0xE0
    """


def midi_note_number_to_name():
    """Convert MIDI note number to note name.

    >>> from coremusic.music import midi_to_note_name
    >>> midi_to_note_name(60)  # Middle C
    'C4'
    >>> midi_to_note_name(69)  # A440
    'A4'
    >>> midi_to_note_name(48)  # C3
    'C3'
    >>> midi_to_note_name(72)  # C5
    'C5'

    >>> # Accidentals may be spelled as flats instead of sharps
    >>> midi_to_note_name(61)
    'C#4'
    >>> midi_to_note_name(61, use_flats=True)
    'Db4'

    >>> # Round trips with note_name_to_midi
    >>> from coremusic.music import note_name_to_midi
    >>> assert all(note_name_to_midi(midi_to_note_name(n)) == n for n in range(128))
    """


def midi_name_to_note_number():
    """Convert note name to MIDI note number.

    Use note_name_to_midi rather than parsing names yourself. It validates the
    name and the resulting range, and handles enharmonic spellings.

    >>> from coremusic.music import note_name_to_midi
    >>> note_name_to_midi("C4")
    60
    >>> note_name_to_midi("A4")
    69
    >>> note_name_to_midi("C#4")
    61
    >>> note_name_to_midi("Db4")  # enharmonic with C#4
    61
    >>> note_name_to_midi("C3")
    48

    Octave numbering is scientific pitch notation: middle C is C4 is 60.
    Ableton Live and Logic display that same note as C3.

    >>> # The builders take a name directly, so this is rarely needed
    >>> from coremusic.midi import note_on
    >>> assert note_on("C4", 100) == note_on(note_name_to_midi("C4"), 100)

    >>> # Bad names and out-of-range results raise rather than pass silently
    >>> note_name_to_midi("H4")
    Traceback (most recent call last):
        ...
    ValueError: Invalid note name: H4
    """


def build_midi_melody():
    """Build a sequence of MIDI messages for a melody.

    >>> from coremusic.midi import note_off, note_on
    >>> def create_melody_messages(notes, velocities=None, channel=0):
    ...     '''Create Note On/Off messages for a melody.'''
    ...     if velocities is None:
    ...         velocities = [100] * len(notes)
    ...     messages = []
    ...     for note, vel in zip(notes, velocities, strict=True):
    ...         messages.append(('on', note_on(note, vel, channel=channel)))
    ...         messages.append(('off', note_off(note, channel=channel)))
    ...     return messages

    >>> # C major scale
    >>> scale = [60, 62, 64, 65, 67, 69, 71, 72]
    >>> messages = create_melody_messages(scale)
    >>> len(messages)
    16
    >>> messages[0][0]  # First message type
    'on'
    >>> messages[0][1][1]  # First note
    60
    """


def midi_client_lifecycle():
    """Demonstrate proper MIDI client lifecycle.

    >>> from coremusic.midi import MIDIClient
    >>> # Create client
    >>> client = MIDIClient("Lifecycle Test")
    >>> # Create ports
    >>> output = client.create_output_port("Out")
    >>> input_port = client.create_input_port("In")
    >>> # Use the client...
    >>> # Always clean up
    >>> client.dispose()
    """


# Test runner
if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
