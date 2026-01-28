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
    pass


def create_midi_client():
    """Create a MIDI client.

    >>> import coremusic as cm
    >>> client = cm.MIDIClient("Test Client")
    >>> assert client is not None
    >>> # Always dispose when done
    >>> client.dispose()
    """
    pass


def create_midi_output_port():
    """Create a MIDI output port.

    >>> import coremusic as cm
    >>> client = cm.MIDIClient("Test Client")
    >>> try:
    ...     output_port = client.create_output_port("Test Output")
    ...     assert output_port is not None
    ... finally:
    ...     client.dispose()
    """
    pass


def create_midi_input_port():
    """Create a MIDI input port.

    >>> import coremusic as cm
    >>> client = cm.MIDIClient("Test Client")
    >>> try:
    ...     input_port = client.create_input_port("Test Input")
    ...     assert input_port is not None
    ... finally:
    ...     client.dispose()
    """
    pass


def midi_note_on_message():
    """Construct a MIDI Note On message.

    MIDI Note On format: [0x90 + channel, note, velocity]
    - Status byte: 0x90 (Note On) + channel (0-15)
    - Note: 0-127 (60 = Middle C)
    - Velocity: 0-127

    >>> # Note On: Middle C (60), velocity 100, channel 0
    >>> note_on = bytes([0x90, 60, 100])
    >>> assert len(note_on) == 3
    >>> assert note_on[0] == 0x90  # Note On, channel 0
    >>> assert note_on[1] == 60    # Middle C
    >>> assert note_on[2] == 100   # Velocity
    """
    pass


def midi_note_off_message():
    """Construct a MIDI Note Off message.

    MIDI Note Off format: [0x80 + channel, note, velocity]
    - Status byte: 0x80 (Note Off) + channel (0-15)
    - Note: 0-127
    - Velocity: typically 0

    >>> # Note Off: Middle C (60), channel 0
    >>> note_off = bytes([0x80, 60, 0])
    >>> assert len(note_off) == 3
    >>> assert note_off[0] == 0x80  # Note Off, channel 0
    >>> assert note_off[1] == 60    # Middle C
    >>> assert note_off[2] == 0     # Velocity (release)
    """
    pass


def midi_control_change_message():
    """Construct a MIDI Control Change (CC) message.

    MIDI CC format: [0xB0 + channel, controller, value]
    - Status byte: 0xB0 (CC) + channel (0-15)
    - Controller: 0-127 (e.g., 1=mod wheel, 7=volume, 10=pan)
    - Value: 0-127

    >>> # CC: Modulation wheel to 64 (50%), channel 0
    >>> mod_wheel = bytes([0xB0, 1, 64])
    >>> assert len(mod_wheel) == 3
    >>> assert mod_wheel[0] == 0xB0  # CC, channel 0
    >>> assert mod_wheel[1] == 1     # Mod wheel
    >>> assert mod_wheel[2] == 64    # Value (50%)

    >>> # Common CC numbers
    >>> CC_MOD_WHEEL = 1
    >>> CC_VOLUME = 7
    >>> CC_PAN = 10
    >>> CC_SUSTAIN = 64
    >>> CC_ALL_NOTES_OFF = 123
    """
    pass


def midi_program_change_message():
    """Construct a MIDI Program Change message.

    MIDI Program Change format: [0xC0 + channel, program]
    - Status byte: 0xC0 (Program Change) + channel (0-15)
    - Program: 0-127

    >>> # Program Change: Select program 0 (piano), channel 0
    >>> program_change = bytes([0xC0, 0])
    >>> assert len(program_change) == 2
    >>> assert program_change[0] == 0xC0  # Program Change, channel 0
    >>> assert program_change[1] == 0     # Program 0
    """
    pass


def midi_pitch_bend_message():
    """Construct a MIDI Pitch Bend message.

    MIDI Pitch Bend format: [0xE0 + channel, LSB, MSB]
    - Status byte: 0xE0 (Pitch Bend) + channel (0-15)
    - Value: 14-bit (0-16383), center = 8192
    - LSB: value & 0x7F
    - MSB: (value >> 7) & 0x7F

    >>> # Pitch Bend: Center position (no bend)
    >>> center = 8192
    >>> lsb = center & 0x7F
    >>> msb = (center >> 7) & 0x7F
    >>> pitch_bend = bytes([0xE0, lsb, msb])
    >>> assert len(pitch_bend) == 3
    >>> assert pitch_bend[0] == 0xE0  # Pitch Bend, channel 0

    >>> # Reconstruct value
    >>> reconstructed = pitch_bend[1] | (pitch_bend[2] << 7)
    >>> assert reconstructed == 8192
    """
    pass


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
    pass


def midi_note_number_to_name():
    """Convert MIDI note number to note name.

    >>> NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    >>> def note_to_name(note_num):
    ...     octave = (note_num // 12) - 1
    ...     note = NOTE_NAMES[note_num % 12]
    ...     return f"{note}{octave}"

    >>> note_to_name(60)  # Middle C
    'C4'
    >>> note_to_name(69)  # A440
    'A4'
    >>> note_to_name(48)  # C3
    'C3'
    >>> note_to_name(72)  # C5
    'C5'
    """
    pass


def midi_name_to_note_number():
    """Convert note name to MIDI note number.

    >>> NOTE_MAP = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    >>> def name_to_note(name):
    ...     # Parse note name (e.g., "C#4", "Db4", "C4")
    ...     note = name[0].upper()
    ...     idx = 1
    ...     modifier = 0
    ...     if len(name) > 1 and name[1] in '#b':
    ...         modifier = 1 if name[1] == '#' else -1
    ...         idx = 2
    ...     octave = int(name[idx:])
    ...     return NOTE_MAP[note] + modifier + (octave + 1) * 12

    >>> name_to_note("C4")
    60
    >>> name_to_note("A4")
    69
    >>> name_to_note("C#4")
    61
    >>> name_to_note("C3")
    48
    """
    pass


def build_midi_melody():
    """Build a sequence of MIDI messages for a melody.

    >>> def create_melody_messages(notes, velocities=None, channel=0):
    ...     '''Create Note On/Off messages for a melody.'''
    ...     if velocities is None:
    ...         velocities = [100] * len(notes)
    ...     messages = []
    ...     for note, vel in zip(notes, velocities):
    ...         # Note On
    ...         messages.append(('on', bytes([0x90 + channel, note, vel])))
    ...         # Note Off
    ...         messages.append(('off', bytes([0x80 + channel, note, 0])))
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
    pass


def midi_client_lifecycle():
    """Demonstrate proper MIDI client lifecycle.

    >>> import coremusic as cm
    >>> # Create client
    >>> client = cm.MIDIClient("Lifecycle Test")
    >>> # Create ports
    >>> output = client.create_output_port("Out")
    >>> input_port = client.create_input_port("In")
    >>> # Use the client...
    >>> # Always clean up
    >>> client.dispose()
    """
    pass


# Test runner
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
