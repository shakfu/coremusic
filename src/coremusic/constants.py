#!/usr/bin/env python3
"""CoreAudio constant enumerations for CoreMusic.

This module provides Pythonic Enum classes for CoreAudio constants, offering
better IDE support, type safety, and discoverability compared to individual
getter functions. All existing getter functions remain available for backward
compatibility.

The constants are organized by category:
- AudioFileProperty: Audio file properties
- AudioFileType: Audio file types
- AudioFormatID: Audio format identifiers
- LinearPCMFormatFlag: Linear PCM format flags
- AudioConverterProperty: Audio converter properties
- AudioConverterQuality: Converter quality settings
- AudioUnitProperty: AudioUnit properties
- AudioUnitScope: AudioUnit scopes
- AudioUnitRenderActionFlags: Render action flags
- MIDIStatus: MIDI status bytes

Usage::

    import coremusic as cm
    from coremusic.constants import AudioFileProperty, AudioFormatID

    # Use enum values
    format_property = AudioFileProperty.DATA_FORMAT
    format_id = AudioFormatID.LINEAR_PCM

    # Convert to integer for API calls
    property_id = int(format_property)  # or format_property.value

    # Compare with integers
    if some_value == AudioFileProperty.DATA_FORMAT:
        print("It's the data format property")

    # Backward compatible: getter functions still work
    property_id = cm.capi.get_audio_file_property_data_format()
"""

from enum import IntEnum

__all__ = [
    # Audio File
    "AudioFileProperty",
    "AudioFileType",
    "AudioFilePermission",
    # Audio Format
    "AudioFormatID",
    "LinearPCMFormatFlag",
    # Audio Converter
    "AudioConverterProperty",
    "AudioConverterQuality",
    # Extended Audio File
    "ExtendedAudioFileProperty",
    # Audio Unit
    "AudioUnitProperty",
    "AudioUnitScope",
    "AudioUnitElement",
    "AudioUnitRenderActionFlags",
    "AudioUnitParameterUnit",
    # Audio Queue
    "AudioQueueProperty",
    "AudioQueueParameter",
    # Audio Object/Device
    "AudioObjectProperty",
    "AudioDeviceProperty",
    # MIDI
    "MIDIStatus",
    "MIDIControlChange",
    "MIDIObjectProperty",
]


# ============================================================================
# Audio File Constants
# ============================================================================

class AudioFileProperty(IntEnum):
    """Audio file property IDs (kAudioFileProperty*)"""
    DATA_FORMAT = 1684434292  # 'dfmt'
    FILE_FORMAT = 1717988724  # 'ffmt'
    MAXIMUM_PACKET_SIZE = 1886616165  # 'psze'
    AUDIO_DATA_PACKET_COUNT = 1885564532  # 'pcnt'
    AUDIO_DATA_BYTE_COUNT = 1650683508  # 'bcnt'
    ESTIMATED_DURATION = 1701082482  # 'edur'
    BIT_RATE = 1651663220  # 'brat'
    INFO_DICTIONARY = 1768842863  # 'info'
    CHANNEL_LAYOUT = 1668112752  # 'cmap'
    FORMAT_LIST = 1718383476  # 'flst'
    PACKET_SIZE_UPPER_BOUND = 1886090594  # 'pkub'
    RESERVE_DURATION = 1920168566  # 'rsrv'
    PACKET_TABLE_INFO = 1886283375  # 'pnfo'
    MARKER_LIST = 1835756659  # 'mkls'
    REGION_LIST = 1919380595  # 'rgls'
    CHUNK_IDS = 1667787108  # 'chid'
    DATA_OFFSET = 1685022310  # 'doff'
    DATA_SIZE = 1685285242  # 'dsiz'
    DATA_IS_BIG_ENDIAN = 1684431461  # 'dfbe'


class AudioFileType(IntEnum):
    """Audio file type IDs (AudioFileTypeID)"""
    WAVE = 1463899717  # 'WAVE'
    AIFF = 1095321158  # 'AIFF'
    AIFC = 1095321155  # 'AIFC'
    NEXT = 1315264596  # 'NeXT'
    MP3 = 1297106739  # 'MPG3'
    MP2 = 1297106738  # 'MPG2'
    MP1 = 1297106737  # 'MPG1'
    AC3 = 1633889587  # 'ac-3'
    AAC_ADTS = 1633973363  # 'adts'
    MPEG4 = 1836069990  # 'm4a '
    M4A = 1836069990  # 'm4a '
    M4B = 1836069986  # 'm4b '
    CAF = 1667327590  # 'caff'
    THREEGP = 862417008  # '3gp '
    THREEGP2 = 862416946  # '3gp2'
    AMR = 1634562662  # 'amrf'


class AudioFilePermission(IntEnum):
    """Audio file permissions"""
    READ = 1  # kAudioFileReadPermission
    WRITE = 2  # kAudioFileWritePermission
    READ_WRITE = 3  # kAudioFileReadWritePermission


# ============================================================================
# Audio Format Constants
# ============================================================================

class AudioFormatID(IntEnum):
    """Audio format IDs (AudioFormatID)"""
    LINEAR_PCM = 1819304813  # 'lpcm'
    AC3 = 1633889587  # 'ac-3'
    AC3_60958 = 1667326771  # 'cac3' - 60958 AC3 variant
    APPLE_IMA4 = 1768775988  # 'ima4'
    MPEG4_AAC = 1633772320  # 'aac '
    MPEG4_CELP = 1667591280  # 'celp'
    MPEG4_HVXC = 1752594531  # 'hvxc'
    MPEG4_TWINVQ = 1953986161  # 'twvq'
    MACE3 = 1296122675  # 'MAC3'
    MACE6 = 1296122678  # 'MAC6'
    ULAW = 1970037111  # 'ulaw'
    ALAW = 1634492791  # 'alaw'
    QDESIGN_MUSIC = 1363430723  # 'QDMC'
    QDESIGN2 = 1363430706  # 'QDM2'
    QUALCOMM = 1365470320  # 'Qclp'
    MPEG_LAYER_1 = 778924081  # '.mp1'
    MPEG_LAYER_2 = 778924082  # '.mp2'
    MPEG_LAYER_3 = 778924083  # '.mp3'
    TIME_CODE = 1953066341  # 'time'
    MIDI_STREAM = 1835623529  # 'midi'
    PARAMETER_VALUE_STREAM = 1634760307  # 'apvs'
    APPLE_LOSSLESS = 1634492771  # 'alac'
    MPEG4_AAC_HE = 1633773856  # 'aach'
    MPEG4_AAC_LD = 1633771875  # 'aacl'
    MPEG4_AAC_ELD = 1633771877  # 'aace'
    MPEG4_AAC_ELD_SBR = 1633772130  # 'aacf'
    MPEG4_AAC_ELD_V2 = 1633772131  # 'aacg'
    MPEG4_AAC_HE_V2 = 1633773672  # 'aacp'
    MPEG4_AAC_SPATIAL = 1633775979  # 'aacs'
    AMR = 1935764850  # 'samr'
    AMR_WB = 1935767394  # 'sawb'
    AUDIBLE = 1096107074  # 'AUDB'
    ILBC = 1768710755  # 'ilbc'
    DVIINTEL_IMA = 1836253201  # 'ms\x00\x11'
    MICROSOFT_GSMA_ADPCM = 1836253233  # 'ms\x00\x31'
    GSM610 = 1735159122  # 'gsm '
    ADPCM_IMA_WAV = 1836253217  # 'ms\x00\x21'
    MPEG4_AAC_LD_V2 = 1633771876  # 'aacl'
    MPEG4_AAC_HE_V2_SBR = 1633773673  # 'aacp'
    OPUS = 1869641075  # 'opus'
    FLAC = 1718378851  # 'flac'


class LinearPCMFormatFlag(IntEnum):
    """Linear PCM format flags (AudioFormatFlags for kAudioFormatLinearPCM)"""
    IS_FLOAT = 1  # kAudioFormatFlagIsFloat
    IS_BIG_ENDIAN = 2  # kAudioFormatFlagIsBigEndian
    IS_SIGNED_INTEGER = 4  # kAudioFormatFlagIsSignedInteger
    IS_PACKED = 8  # kAudioFormatFlagIsPacked
    IS_ALIGNED_HIGH = 16  # kAudioFormatFlagIsAlignedHigh
    IS_NON_INTERLEAVED = 32  # kAudioFormatFlagIsNonInterleaved
    IS_NON_MIXABLE = 64  # kAudioFormatFlagIsNonMixable
    FLAGS_ALL_CLEAR = 2147483648  # kAudioFormatFlagsAreAllClear

    # Common combinations
    FLAGS_NATIVE_FLOAT_PACKED = 9  # kLinearPCMFormatFlagIsFloat | kLinearPCMFormatFlagIsPacked
    FLAGS_CANONICAL = 12  # Signed integer, packed (native format)


# ============================================================================
# Audio Converter Constants
# ============================================================================

class AudioConverterProperty(IntEnum):
    """Audio converter property IDs (AudioConverterPropertyID)"""
    MIN_INPUT_BUFFER_SIZE = 2020436322  # 'xmib'
    MIN_OUTPUT_BUFFER_SIZE = 1481655666  # 'xmob'
    MAX_INPUT_PACKET_SIZE = 1481656691  # 'xmip'
    MAX_OUTPUT_PACKET_SIZE = 1481656688  # 'xmop'
    SAMPLE_RATE_CONVERTER_QUALITY = 1936876401  # 'srcq'
    CODEC_QUALITY = 1667527029  # 'cdqu'
    CURRENT_INPUT_STREAM_DESCRIPTION = 1633906541  # 'aisd'
    CURRENT_OUTPUT_STREAM_DESCRIPTION = 1633905012  # 'aosd'
    PROPERTY_SETTINGS = 1633903476  # 'acps'
    AVAILABLE_ENCODE_BIT_RATES = 1634034290  # 'aebr'
    APPLICABLE_ENCODE_BIT_RATES = 1634169458  # 'aebr'
    AVAILABLE_ENCODE_SAMPLE_RATES = 1634038642  # 'aesr'
    APPLICABLE_ENCODE_SAMPLE_RATES = 1634169458  # 'aesr'
    AVAILABLE_ENCODE_CHANNEL_LAYOUT_TAGS = 1633906540  # 'aecl'
    BIT_RATE = 1651663220  # 'brat'
    BIT_RATE_CONTROL_MODE = 1633772909  # 'acbf'
    SOUND_QUALITY_FOR_VBR = 1986097262  # 'vbrq'


class AudioConverterQuality(IntEnum):
    """Audio converter quality settings (AudioConverterQuality)"""
    MAX = 127  # kAudioConverterQuality_Max
    HIGH = 96  # kAudioConverterQuality_High
    MEDIUM = 64  # kAudioConverterQuality_Medium
    LOW = 32  # kAudioConverterQuality_Low
    MIN = 0  # kAudioConverterQuality_Min


# ============================================================================
# Extended Audio File Constants
# ============================================================================

class ExtendedAudioFileProperty(IntEnum):
    """Extended audio file property IDs (ExtAudioFilePropertyID)"""
    FILE_DATA_FORMAT = 1717988724  # 'ffmt'
    CLIENT_DATA_FORMAT = 1667657076  # 'cfmt'
    FILE_CHANNEL_LAYOUT = 1717791855  # 'fclo'
    CLIENT_CHANNEL_LAYOUT = 1667788144  # 'cclo'
    CODEC_MANUFACTURER = 1668446576  # 'cman'
    AUDIO_FILE = 1634101612  # 'afil'
    FILE_LENGTH_FRAMES = 1718509674  # 'flgf'
    AUDIO_CONVERTER = 1633907830  # 'acnv'
    CLIENT_MAX_PACKET_SIZE = 1668048243  # 'cmps'


# ============================================================================
# AudioUnit Constants
# ============================================================================

class AudioUnitProperty(IntEnum):
    """AudioUnit property IDs (AudioUnitPropertyID)"""
    SAMPLE_RATE = 2  # kAudioUnitProperty_SampleRate
    PARAMETER_LIST = 3  # kAudioUnitProperty_ParameterList
    PARAMETER_INFO = 4  # kAudioUnitProperty_ParameterInfo
    STREAM_FORMAT = 8  # kAudioUnitProperty_StreamFormat
    ELEMENT_COUNT = 11  # kAudioUnitProperty_ElementCount
    LATENCY = 12  # kAudioUnitProperty_Latency
    MAXIMUM_FRAMES_PER_SLICE = 14  # kAudioUnitProperty_MaximumFramesPerSlice
    SET_RENDER_CALLBACK = 23  # kAudioUnitProperty_SetRenderCallback
    FACTORY_PRESETS = 24  # kAudioUnitProperty_FactoryPresets
    RENDER_QUALITY = 26  # kAudioUnitProperty_RenderQuality
    HOST_CALLBACKS = 27  # kAudioUnitProperty_HostCallbacks
    IN_PLACE_PROCESSING = 29  # kAudioUnitProperty_InPlaceProcessing
    ELEMENT_NAME = 30  # kAudioUnitProperty_ElementName
    COCOAUI = 31  # kAudioUnitProperty_CocoaUI
    CHANNEL_MAP = 33  # kAudioUnitProperty_ChannelMap
    AUDIO_CHANNEL_LAYOUT = 19  # kAudioUnitProperty_AudioChannelLayout
    TAIL_TIME = 20  # kAudioUnitProperty_TailTime
    BYPASS_EFFECT = 21  # kAudioUnitProperty_BypassEffect
    LAST_RENDER_ERROR = 22  # kAudioUnitProperty_LastRenderError
    SET_EXTERNAL_BUFFER = 15  # kAudioUnitProperty_SetExternalBuffer
    GET_UI_COMPONENT_LIST = 18  # kAudioUnitProperty_GetUIComponentList
    METER_CLIPPING = 1634755187  # 'clip'
    PRESENT_PRESET = 1886547818  # 'pset'
    OFFLINE_RENDER = 55  # kAudioUnitProperty_OfflineRender
    PARAMETER_STRING_FROM_VALUE = 33  # kAudioUnitProperty_ParameterStringFromValue
    PARAMETER_CLUMP_NAME = 34  # kAudioUnitProperty_ParameterClumpName
    CLASS_INFO = 0  # kAudioUnitProperty_ClassInfo (for presets)


class AudioUnitScope(IntEnum):
    """AudioUnit scope identifiers (AudioUnitScope)"""
    GLOBAL = 0  # kAudioUnitScope_Global
    INPUT = 1  # kAudioUnitScope_Input
    OUTPUT = 2  # kAudioUnitScope_Output
    GROUP = 3  # kAudioUnitScope_Group
    PART = 4  # kAudioUnitScope_Part
    NOTE = 5  # kAudioUnitScope_Note
    LAYER = 6  # kAudioUnitScope_Layer
    LAYER_ITEM = 7  # kAudioUnitScope_LayerItem


class AudioUnitElement(IntEnum):
    """Common AudioUnit element indices"""
    OUTPUT = 0  # Output element
    INPUT = 1  # Input element


class AudioUnitRenderActionFlags(IntEnum):
    """AudioUnit render action flags (AudioUnitRenderActionFlags)"""
    PRE_RENDER = 4  # kAudioUnitRenderAction_PreRender
    POST_RENDER = 8  # kAudioUnitRenderAction_PostRender
    OUTPUT_IS_SILENCE = 16  # kAudioUnitRenderAction_OutputIsSilence
    OFFLINE_RENDER = 32  # kAudioOfflineUnitRenderAction_Render
    OFFLINE_COMPLETE = 64  # kAudioOfflineUnitRenderAction_Complete
    OFFLINE_PREFLIGHT = 128  # kAudioOfflineUnitRenderAction_Preflight
    POST_RENDER_ERROR = 256  # kAudioUnitRenderAction_PostRenderError
    DO_NOT_CHECK_RENDER_ARGS = 512  # kAudioUnitRenderAction_DoNotCheckRenderArgs


class AudioUnitParameterUnit(IntEnum):
    """AudioUnit parameter units (AudioUnitParameterUnit)"""
    GENERIC = 0  # kAudioUnitParameterUnit_Generic
    INDEXED = 1  # kAudioUnitParameterUnit_Indexed
    BOOLEAN = 2  # kAudioUnitParameterUnit_Boolean
    PERCENT = 3  # kAudioUnitParameterUnit_Percent
    SECONDS = 4  # kAudioUnitParameterUnit_Seconds
    SAMPLE_FRAMES = 5  # kAudioUnitParameterUnit_SampleFrames
    PHASE = 6  # kAudioUnitParameterUnit_Phase
    RATE = 7  # kAudioUnitParameterUnit_Rate
    HERTZ = 8  # kAudioUnitParameterUnit_Hertz
    CENTS = 9  # kAudioUnitParameterUnit_Cents
    RELATIVE_SEMITONES = 10  # kAudioUnitParameterUnit_RelativeSemiTones
    MIDI_NOTE_NUMBER = 11  # kAudioUnitParameterUnit_MIDINoteNumber
    MIDI_CONTROLLER = 12  # kAudioUnitParameterUnit_MIDIController
    DECIBELS = 13  # kAudioUnitParameterUnit_Decibels
    LINEAR_GAIN = 14  # kAudioUnitParameterUnit_LinearGain
    DEGREES = 15  # kAudioUnitParameterUnit_Degrees
    EQUAL_POWER_CROSSFADE = 16  # kAudioUnitParameterUnit_EqualPowerCrossfade
    MIXER_FADER_CURVE1 = 17  # kAudioUnitParameterUnit_MixerFaderCurve1
    PAN = 18  # kAudioUnitParameterUnit_Pan
    METERS = 19  # kAudioUnitParameterUnit_Meters
    ABSOLUTE_CENTS = 20  # kAudioUnitParameterUnit_AbsoluteCents
    OCTAVES = 21  # kAudioUnitParameterUnit_Octaves
    BPM = 22  # kAudioUnitParameterUnit_BPM
    BEATS = 23  # kAudioUnitParameterUnit_Beats
    MILLISECONDS = 24  # kAudioUnitParameterUnit_Milliseconds
    RATIO = 25  # kAudioUnitParameterUnit_Ratio


# ============================================================================
# Audio Queue Constants
# ============================================================================

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


# ============================================================================
# Audio Object/Device Constants
# ============================================================================

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


# ============================================================================
# MIDI Constants
# ============================================================================

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
