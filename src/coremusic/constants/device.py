"""Audio object and device constants.

Every value here is the value of the named macOS SDK constant given in the
trailing comment; see ``tests/test_constants_integrity.py``.
"""

from enum import IntEnum

__all__ = [
    "AudioDeviceProperty",
    "AudioObjectProperty",
]


class AudioObjectProperty(IntEnum):
    """Audio object property IDs (AudioObjectPropertySelector)"""

    NAME = 1819173229  # kAudioObjectPropertyName ('lnam')
    MANUFACTURER = 1819107691  # kAudioObjectPropertyManufacturer ('lmak')
    MODEL_NAME = 1819111268  # kAudioObjectPropertyModelName ('lmod')
    SERIAL_NUMBER = 1936618861  # kAudioObjectPropertySerialNumber ('snum')
    FIRMWARE_VERSION = 1719105134  # kAudioObjectPropertyFirmwareVersion ('fwvn')
    ELEMENT_NAME = 1818454126  # kAudioObjectPropertyElementName ('lchn')
    ELEMENT_CATEGORY_NAME = (
        1818452846  # kAudioObjectPropertyElementCategoryName ('lccn')
    )
    ELEMENT_NUMBER_NAME = 1818455662  # kAudioObjectPropertyElementNumberName ('lcnn')


class AudioDeviceProperty(IntEnum):
    """Audio device property IDs (AudioDevicePropertyID)"""

    DEVICE_UID = 1969841184  # kAudioDevicePropertyDeviceUID ('uid ')
    MODEL_UID = 1836411236  # kAudioDevicePropertyModelUID ('muid')
    TRANSPORT_TYPE = 1953653102  # kAudioDevicePropertyTransportType ('tran')
    RELATED_DEVICES = 1634429294  # kAudioDevicePropertyRelatedDevices ('akin')
    CLOCK_DOMAIN = 1668049764  # kAudioDevicePropertyClockDomain ('clkd')
    DEVICE_IS_ALIVE = 1818850926  # kAudioDevicePropertyDeviceIsAlive ('livn')
    DEVICE_IS_RUNNING = 1735354734  # kAudioDevicePropertyDeviceIsRunning ('goin')
    DEVICE_CAN_BE_DEFAULT_DEVICE = (
        1684434036  # kAudioDevicePropertyDeviceCanBeDefaultDevice ('dflt')
    )
    DEVICE_CAN_BE_DEFAULT_SYSTEM_DEVICE = (
        1936092276  # kAudioDevicePropertyDeviceCanBeDefaultSystemDevice ('sflt')
    )
    LATENCY = 1819569763  # kAudioDevicePropertyLatency ('ltnc')
    STREAMS = 1937009955  # kAudioDevicePropertyStreams ('stm#')
    AVAILABLE_NOMINAL_SAMPLE_RATES = (
        1853059619  # kAudioDevicePropertyAvailableNominalSampleRates ('nsr#')
    )
    NOMINAL_SAMPLE_RATE = 1853059700  # kAudioDevicePropertyNominalSampleRate ('nsrt')
    STREAM_CONFIGURATION = (
        1936482681  # kAudioDevicePropertyStreamConfiguration ('slay')
    )
    VOLUME_SCALAR = 1987013741  # kAudioDevicePropertyVolumeScalar ('volm')
    MUTE = 1836414053  # kAudioDevicePropertyMute ('mute')
    IS_HIDDEN = 1751737454  # kAudioDevicePropertyIsHidden ('hidn')
    PREFERRED_CHANNELS_FOR_STEREO = (
        1684236338  # kAudioDevicePropertyPreferredChannelsForStereo ('dch2')
    )
