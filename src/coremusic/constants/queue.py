"""Audio queue constants.

Every value here is the value of the named macOS SDK constant given in the
trailing comment; see ``tests/test_constants_integrity.py``.
"""

from enum import IntEnum

__all__ = [
    "AudioQueueParameter",
    "AudioQueueProperty",
]


class AudioQueueProperty(IntEnum):
    """Audio queue property IDs (AudioQueuePropertyID)"""

    IS_RUNNING = 1634824814  # kAudioQueueProperty_IsRunning ('aqrn')
    DEVICE_SAMPLE_RATE = 1634825074  # kAudioQueueDeviceProperty_SampleRate ('aqsr')
    DEVICE_NUMBER_CHANNELS = (
        1634821219  # kAudioQueueDeviceProperty_NumberChannels ('aqdc')
    )
    CURRENT_DEVICE = 1634820964  # kAudioQueueProperty_CurrentDevice ('aqcd')
    MAGIC_COOKIE = 1634823523  # kAudioQueueProperty_MagicCookie ('aqmc')
    MAXIMUM_OUTPUT_PACKET_SIZE = (
        2020569203  # kAudioQueueProperty_MaximumOutputPacketSize ('xops')
    )
    STREAM_DESCRIPTION = 1634821748  # kAudioQueueProperty_StreamDescription ('aqft')
    CHANNEL_LAYOUT = 1634820972  # kAudioQueueProperty_ChannelLayout ('aqcl')
    ENABLE_LEVEL_METERING = (
        1634823525  # kAudioQueueProperty_EnableLevelMetering ('aqme')
    )
    CURRENT_LEVEL_METER = 1634823542  # kAudioQueueProperty_CurrentLevelMeter ('aqmv')
    CURRENT_LEVEL_METER_DB = (
        1634823524  # kAudioQueueProperty_CurrentLevelMeterDB ('aqmd')
    )
    DECODE_BUFFER_SIZE_FRAMES = (
        1684234854  # kAudioQueueProperty_DecodeBufferSizeFrames ('dcbf')
    )
    CONVERTER_ERROR = 1902343781  # kAudioQueueProperty_ConverterError ('qcve')
    ENABLE_TIME_PITCH = 1902081136  # kAudioQueueProperty_EnableTimePitch ('q_tp')
    TIME_PITCH_ALGORITHM = 1903456353  # kAudioQueueProperty_TimePitchAlgorithm ('qtpa')
    TIME_PITCH_BYPASS = 1903456354  # kAudioQueueProperty_TimePitchBypass ('qtpb')


class AudioQueueParameter(IntEnum):
    """Audio queue parameter IDs (AudioQueueParameterID)"""

    VOLUME = 1  # kAudioQueueParam_Volume
    PLAYBACK_RATE = 2  # kAudioQueueParam_PlayRate
    PITCH = 3  # kAudioQueueParam_Pitch
    VOLUME_RAMP_TIME = 4  # kAudioQueueParam_VolumeRampTime
    PAN = 13  # kAudioQueueParam_Pan

    # Deprecated alias for VOLUME_RAMP_TIME, kept for backward compatibility.
    # Every other member of this enum is UPPER_SNAKE_CASE.
    VolumeRampTime = 4  # kAudioQueueParam_VolumeRampTime
