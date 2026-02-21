"""Audio queue constants."""

from enum import IntEnum

__all__ = [
    "AudioQueueProperty",
    "AudioQueueParameter",
]


class AudioQueueProperty(IntEnum):
    """Audio queue property IDs (AudioQueuePropertyID)"""

    IS_RUNNING = 1634824814  # 'aqrn'
    DEVICE_SAMPLE_RATE = 1634825074  # 'aqsr'
    DEVICE_NUMBER_CHANNELS = 1634820963  # 'aqdc'
    CURRENT_DEVICE = 1634820964  # 'aqcd'
    MAGIC_COOKIE = 1634823523  # 'aqmc'
    MAXIMUM_OUTPUT_PACKET_SIZE = 1668445292  # 'xops'
    STREAM_DESCRIPTION = 1634821748  # 'aqft'
    CHANNEL_LAYOUT = 1634820972  # 'aqcl'
    ENABLE_LEVEL_METERING = 1634823021  # 'aqme'
    CURRENT_LEVEL_METER = 1634823026  # 'aqmv'
    CURRENT_LEVEL_METER_DB = 1634823010  # 'aqmd'
    DECODE_BUFFER_SIZE_FRAMES = 1684234854  # 'dcbf'
    CONVERTER_ERROR = 1902342501  # 'qcve'
    ENABLE_TIME_PITCH = 1902081376  # 'q_tp'
    TIME_PITCH_ALGORITHM = 1903784544  # 'qtpa'
    TIME_PITCH_BYPASS = 1903784290  # 'qtpb'


class AudioQueueParameter(IntEnum):
    """Audio queue parameter IDs (AudioQueueParameterID)"""

    VOLUME = 1  # kAudioQueueParam_Volume
    PLAYBACK_RATE = 2  # kAudioQueueParam_PlayRate
    PITCH = 3  # kAudioQueueParam_Pitch
    VolumeRampTime = 4  # kAudioQueueParam_VolumeRampTime
    PAN = 13  # kAudioQueueParam_Pan
