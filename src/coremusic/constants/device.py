"""Audio object and device constants."""

from enum import IntEnum

__all__ = [
    "AudioObjectProperty",
    "AudioDeviceProperty",
]


class AudioObjectProperty(IntEnum):
    """Audio object property IDs (AudioObjectPropertySelector)"""

    NAME = 1819173229  # 'lnam'
    MANUFACTURER = 1819107691  # 'lmak'
    ELEMENT_NAME = 1818454126  # 'lchn'
    ELEMENT_CATEGORY_NAME = 1818455908  # 'lccn'
    ELEMENT_NUMBER_NAME = 1818456174  # 'lcnn'
    DEVICE_NAME_IN_OWNER_USER_INTERFACE = 1685288559  # 'duin'
    CLASS_NAME = 1818437741  # 'lccl'


class AudioDeviceProperty(IntEnum):
    """Audio device property IDs (AudioDevicePropertyID)"""

    DEVICE_UID = 1969841184  # 'uid '
    MODEL_UID = 1836411236  # 'muid'
    TRANSPORT_TYPE = 1953653102  # 'tran'
    RELATED_DEVICES = 1634430576  # 'akin'
    CLOCK_DOMAIN = 1668049764  # 'clkd'
    DEVICE_IS_ALIVE = 1818850926  # 'livn'
    DEVICE_IS_RUNNING = 1735354734  # 'goin'
    DEVICE_CAN_BE_DEFAULT_DEVICE = 1684434036  # 'dflt'
    DEVICE_CAN_BE_DEFAULT_SYSTEM_DEVICE = 1935964528  # 'sflt'
    LATENCY = 1819569763  # 'ltnc'
    STREAMS = 1937006960  # 'stm#'
    AVAILABLE_NOMINAL_SAMPLE_RATES = 1853059619  # 'nsr#'
    NOMINAL_SAMPLE_RATE = 1853059826  # 'nsrt'
