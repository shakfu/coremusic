"""MIDI constants."""

from enum import IntEnum

__all__ = [
    "MIDIControlChange",
    "MIDIStatus",
]


class MIDIStatus(IntEnum):
    """MIDI status bytes (high nibble)"""

    NOTE_OFF = 0x80  # 128
    NOTE_ON = 0x90  # 144
    POLY_AFTERTOUCH = 0xA0  # 160
    CONTROL_CHANGE = 0xB0  # 176
    PROGRAM_CHANGE = 0xC0  # 192
    CHANNEL_AFTERTOUCH = 0xD0  # 208
    PITCH_BEND = 0xE0  # 224
    SYSTEM = 0xF0  # 240 (system messages)


class MIDIControlChange(IntEnum):
    """Common MIDI Control Change numbers"""

    BANK_SELECT = 0
    MODULATION = 1
    BREATH_CONTROLLER = 2
    FOOT_CONTROLLER = 4
    PORTAMENTO_TIME = 5
    DATA_ENTRY_MSB = 6
    VOLUME = 7
    BALANCE = 8
    PAN = 10
    EXPRESSION = 11
    EFFECT_CONTROL_1 = 12
    EFFECT_CONTROL_2 = 13
    SUSTAIN_PEDAL = 64
    PORTAMENTO = 65
    SOSTENUTO = 66
    SOFT_PEDAL = 67
    LEGATO = 68
    HOLD_2 = 69
    SOUND_CONTROLLER_1 = 70  # Sound Variation
    SOUND_CONTROLLER_2 = 71  # Timbre/Harmonic Intensity
    SOUND_CONTROLLER_3 = 72  # Release Time
    SOUND_CONTROLLER_4 = 73  # Attack Time
    SOUND_CONTROLLER_5 = 74  # Brightness
    SOUND_CONTROLLER_6 = 75
    SOUND_CONTROLLER_7 = 76
    SOUND_CONTROLLER_8 = 77
    SOUND_CONTROLLER_9 = 78
    SOUND_CONTROLLER_10 = 79
    EFFECTS_LEVEL = 91
    TREMOLO_LEVEL = 92
    CHORUS_LEVEL = 93
    CELESTE_LEVEL = 94
    PHASER_LEVEL = 95
    ALL_SOUND_OFF = 120
    RESET_ALL_CONTROLLERS = 121
    ALL_NOTES_OFF = 123
