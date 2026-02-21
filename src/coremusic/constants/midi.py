"""MIDI constants."""

from enum import IntEnum

__all__ = [
    "MIDIStatus",
    "MIDIControlChange",
    "MIDIObjectProperty",
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


class MIDIObjectProperty(IntEnum):
    """MIDI object property IDs"""

    NAME = 1851878757  # 'name'
    MANUFACTURER = 1835101813  # 'manu'
    MODEL = 1836016748  # 'modl'
    UNIQUE_ID = 1970170212  # 'unid'
    DEVICE_ID = 1684632436  # 'devd'
    RECEIVE_CHANNELS = 1919247470  # 'rch#'
    TRANSMIT_CHANNELS = 1919894126  # 'tch#'
    MAX_SYSEX_SPEED = 1937204832  # 'sxsp'
    ADVANCE_SCHEDULE_TIME_MUSC = 1634953321  # 'adv '
    IS_EMBEDDED_ENTITY = 1701737824  # 'emb '
    IS_BROADCAST = 1651470944  # 'bdc '
    SINGLE_REALTIME_ENTITY = 1835365985  # 'srte'
    CONNECTION_UNIQUE_ID = 1668048225  # 'cuid'
    OFFLINE = 1869636966  # 'offl'
    PRIVATE = 1886548070  # 'priv'
    DRIVER_OWNER = 1685808750  # 'down'
    NAME_CONFIGURATION = 1852008291  # 'ncfg'
    IMAGE = 1768846393  # 'imag'
    DRIVER_DEVICE_EDITOR_APP = 1684104552  # 'deap'
    CAN_ROUTE = 1919051621  # 'rout'
    IS_MIXER = 1835626093  # 'mixr'
    IS_SAMPLER = 1935764595  # 'smpl'
    IS_EFFECT_UNIT = 1701209701  # 'ef x'
    MAX_RECEIVE_CHANNELS = 1919904357  # 'rxc '
    MAX_TRANSMIT_CHANNELS = 1920234597  # 'txc '
    IS_DRUM_MACHINE = 1685220205  # 'drum'
