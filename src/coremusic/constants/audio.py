"""Audio file, format, converter, and extended audio file constants."""

from enum import IntEnum

__all__ = [
    "AudioFileProperty",
    "AudioFileType",
    "AudioFilePermission",
    "AudioFormatID",
    "LinearPCMFormatFlag",
    "AudioConverterProperty",
    "AudioConverterQuality",
    "ExtendedAudioFileProperty",
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
    FLAGS_NATIVE_FLOAT_PACKED = (
        9  # kLinearPCMFormatFlagIsFloat | kLinearPCMFormatFlagIsPacked
    )
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
