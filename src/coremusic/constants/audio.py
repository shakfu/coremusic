"""Audio file, format, converter, and extended audio file constants.

Every value here is the value of the named macOS SDK constant given in the
trailing comment. ``tests/test_constants_integrity.py`` re-derives these from
the SDK headers and from the compiled ``coremusic.capi`` getters, so a value
that drifts from its C counterpart fails the test suite rather than silently
producing a wrong property ID at runtime.
"""

from enum import IntEnum

__all__ = [
    "AudioConverterProperty",
    "AudioConverterQuality",
    "AudioFilePermission",
    "AudioFileProperty",
    "AudioFileType",
    "AudioFormatID",
    "ExtendedAudioFileProperty",
    "LinearPCMFormatFlag",
]


# ============================================================================
# Audio File Constants
# ============================================================================


class AudioFileProperty(IntEnum):
    """Audio file property IDs (kAudioFileProperty*)"""

    DATA_FORMAT = 1684434292  # kAudioFilePropertyDataFormat ('dfmt')
    FILE_FORMAT = 1717988724  # kAudioFilePropertyFileFormat ('ffmt')
    MAXIMUM_PACKET_SIZE = 1886616165  # kAudioFilePropertyMaximumPacketSize ('psze')
    AUDIO_DATA_PACKET_COUNT = (
        1885564532  # kAudioFilePropertyAudioDataPacketCount ('pcnt')
    )
    AUDIO_DATA_BYTE_COUNT = 1650683508  # kAudioFilePropertyAudioDataByteCount ('bcnt')
    ESTIMATED_DURATION = 1701082482  # kAudioFilePropertyEstimatedDuration ('edur')
    BIT_RATE = 1651663220  # kAudioFilePropertyBitRate ('brat')
    INFO_DICTIONARY = 1768842863  # kAudioFilePropertyInfoDictionary ('info')
    CHANNEL_LAYOUT = 1668112752  # kAudioFilePropertyChannelLayout ('cmap')
    FORMAT_LIST = 1718383476  # kAudioFilePropertyFormatList ('flst')
    PACKET_SIZE_UPPER_BOUND = (
        1886090594  # kAudioFilePropertyPacketSizeUpperBound ('pkub')
    )
    RESERVE_DURATION = 1920168566  # kAudioFilePropertyReserveDuration ('rsrv')
    PACKET_TABLE_INFO = 1886283375  # kAudioFilePropertyPacketTableInfo ('pnfo')
    MARKER_LIST = 1835756659  # kAudioFilePropertyMarkerList ('mkls')
    REGION_LIST = 1919380595  # kAudioFilePropertyRegionList ('rgls')
    CHUNK_IDS = 1667787108  # kAudioFilePropertyChunkIDs ('chid')
    DATA_OFFSET = 1685022310  # kAudioFilePropertyDataOffset ('doff')
    IS_OPTIMIZED = 1869640813  # kAudioFilePropertyIsOptimized ('optm')
    MAGIC_COOKIE_DATA = 1835493731  # kAudioFilePropertyMagicCookieData ('mgic')


class AudioFileType(IntEnum):
    """Audio file type IDs (AudioFileTypeID)"""

    WAVE = 1463899717  # kAudioFileWAVEType ('WAVE')
    AIFF = 1095321158  # kAudioFileAIFFType ('AIFF')
    AIFC = 1095321155  # kAudioFileAIFCType ('AIFC')
    NEXT = 1315264596  # kAudioFileNextType ('NeXT')
    MP3 = 1297106739  # kAudioFileMP3Type ('MPG3')
    MP2 = 1297106738  # kAudioFileMP2Type ('MPG2')
    MP1 = 1297106737  # kAudioFileMP1Type ('MPG1')
    AC3 = 1633889587  # kAudioFileAC3Type ('ac-3')
    AAC_ADTS = 1633973363  # kAudioFileAAC_ADTSType ('adts')
    MPEG4 = 1836069990  # kAudioFileMPEG4Type ('mp4f')
    M4A = 1832149350  # kAudioFileM4AType ('m4af')
    M4B = 1832149606  # kAudioFileM4BType ('m4bf')
    CAF = 1667327590  # kAudioFileCAFType ('caff')
    FLAC = 1718378851  # kAudioFileFLACType ('flac')
    THREEGP = 862417008  # kAudioFile3GPType ('3gpp')
    THREEGP2 = 862416946  # kAudioFile3GP2Type ('3gp2')
    AMR = 1634562662  # kAudioFileAMRType ('amrf')


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

    LINEAR_PCM = 1819304813  # kAudioFormatLinearPCM ('lpcm')
    AC3 = 1633889587  # kAudioFormatAC3 ('ac-3')
    AC3_60958 = 1667326771  # kAudioFormat60958AC3 ('cac3')
    APPLE_IMA4 = 1768775988  # kAudioFormatAppleIMA4 ('ima4')
    MPEG4_AAC = 1633772320  # kAudioFormatMPEG4AAC ('aac ')
    MPEG4_CELP = 1667591280  # kAudioFormatMPEG4CELP ('celp')
    MPEG4_HVXC = 1752594531  # kAudioFormatMPEG4HVXC ('hvxc')
    MPEG4_TWINVQ = 1953986161  # kAudioFormatMPEG4TwinVQ ('twvq')
    MACE3 = 1296122675  # kAudioFormatMACE3 ('MAC3')
    MACE6 = 1296122678  # kAudioFormatMACE6 ('MAC6')
    ULAW = 1970037111  # kAudioFormatULaw ('ulaw')
    ALAW = 1634492791  # kAudioFormatALaw ('alaw')
    QDESIGN_MUSIC = 1363430723  # kAudioFormatQDesign ('QDMC')
    QDESIGN2 = 1363430706  # kAudioFormatQDesign2 ('QDM2')
    QUALCOMM = 1365470320  # kAudioFormatQUALCOMM ('Qclp')
    MPEG_LAYER_1 = 778924081  # kAudioFormatMPEGLayer1 ('.mp1')
    MPEG_LAYER_2 = 778924082  # kAudioFormatMPEGLayer2 ('.mp2')
    MPEG_LAYER_3 = 778924083  # kAudioFormatMPEGLayer3 ('.mp3')
    TIME_CODE = 1953066341  # kAudioFormatTimeCode ('time')
    MIDI_STREAM = 1835623529  # kAudioFormatMIDIStream ('midi')
    PARAMETER_VALUE_STREAM = 1634760307  # kAudioFormatParameterValueStream ('apvs')
    APPLE_LOSSLESS = 1634492771  # kAudioFormatAppleLossless ('alac')
    MPEG4_AAC_HE = 1633772392  # kAudioFormatMPEG4AAC_HE ('aach')
    MPEG4_AAC_LD = 1633772396  # kAudioFormatMPEG4AAC_LD ('aacl')
    MPEG4_AAC_ELD = 1633772389  # kAudioFormatMPEG4AAC_ELD ('aace')
    MPEG4_AAC_ELD_SBR = 1633772390  # kAudioFormatMPEG4AAC_ELD_SBR ('aacf')
    MPEG4_AAC_ELD_V2 = 1633772391  # kAudioFormatMPEG4AAC_ELD_V2 ('aacg')
    MPEG4_AAC_HE_V2 = 1633772400  # kAudioFormatMPEG4AAC_HE_V2 ('aacp')
    MPEG4_AAC_SPATIAL = 1633772403  # kAudioFormatMPEG4AAC_Spatial ('aacs')
    AMR = 1935764850  # kAudioFormatAMR ('samr')
    AMR_WB = 1935767394  # kAudioFormatAMR_WB ('sawb')
    AUDIBLE = 1096107074  # kAudioFormatAudible ('AUDB')
    ILBC = 1768710755  # kAudioFormatiLBC ('ilbc')
    DVIINTEL_IMA = 1836253201  # kAudioFormatDVIIntelIMA
    MICROSOFT_GSMA_ADPCM = 1836253233  # kAudioFormatMicrosoftGSM
    OPUS = 1869641075  # kAudioFormatOpus ('opus')
    FLAC = 1718378851  # kAudioFormatFLAC ('flac')


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
    FLAGS_NATIVE_FLOAT_PACKED = 9  # IS_FLOAT | IS_PACKED
    FLAGS_CANONICAL = 12  # IS_SIGNED_INTEGER | IS_PACKED


# ============================================================================
# Audio Converter Constants
# ============================================================================


class AudioConverterProperty(IntEnum):
    """Audio converter property IDs (AudioConverterPropertyID)"""

    MIN_INPUT_BUFFER_SIZE = (
        1835623027  # kAudioConverterPropertyMinimumInputBufferSize ('mibs')
    )
    MIN_OUTPUT_BUFFER_SIZE = (
        1836016243  # kAudioConverterPropertyMinimumOutputBufferSize ('mobs')
    )
    MAX_INPUT_PACKET_SIZE = (
        2020175987  # kAudioConverterPropertyMaximumInputPacketSize ('xips')
    )
    MAX_OUTPUT_PACKET_SIZE = (
        2020569203  # kAudioConverterPropertyMaximumOutputPacketSize ('xops')
    )
    SAMPLE_RATE_CONVERTER_QUALITY = (
        1936876401  # kAudioConverterSampleRateConverterQuality ('srcq')
    )
    CODEC_QUALITY = 1667527029  # kAudioConverterCodecQuality ('cdqu')
    CURRENT_INPUT_STREAM_DESCRIPTION = (
        1633904996  # kAudioConverterCurrentInputStreamDescription ('acid')
    )
    CURRENT_OUTPUT_STREAM_DESCRIPTION = (
        1633906532  # kAudioConverterCurrentOutputStreamDescription ('acod')
    )
    PROPERTY_SETTINGS = 1633906803  # kAudioConverterPropertySettings ('acps')
    AVAILABLE_ENCODE_BIT_RATES = (
        1986355826  # kAudioConverterAvailableEncodeBitRates ('vebr')
    )
    APPLICABLE_ENCODE_BIT_RATES = (
        1634034290  # kAudioConverterApplicableEncodeBitRates ('aebr')
    )
    AVAILABLE_ENCODE_SAMPLE_RATES = (
        1986360178  # kAudioConverterAvailableEncodeSampleRates ('vesr')
    )
    APPLICABLE_ENCODE_SAMPLE_RATES = (
        1634038642  # kAudioConverterApplicableEncodeSampleRates ('aesr')
    )
    AVAILABLE_ENCODE_CHANNEL_LAYOUT_TAGS = (
        1634034540  # kAudioConverterAvailableEncodeChannelLayoutTags ('aecl')
    )
    BIT_RATE = 1651663220  # kAudioConverterEncodeBitRate ('brat')
    BIT_RATE_CONTROL_MODE = 1633903206  # kAudioCodecPropertyBitRateControlMode ('acbf')
    SOUND_QUALITY_FOR_VBR = 1986163313  # kAudioCodecPropertySoundQualityForVBR ('vbrq')


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

    FILE_DATA_FORMAT = 1717988724  # kExtAudioFileProperty_FileDataFormat ('ffmt')
    CLIENT_DATA_FORMAT = 1667657076  # kExtAudioFileProperty_ClientDataFormat ('cfmt')
    FILE_CHANNEL_LAYOUT = 1717791855  # kExtAudioFileProperty_FileChannelLayout ('fclo')
    CLIENT_CHANNEL_LAYOUT = (
        1667460207  # kExtAudioFileProperty_ClientChannelLayout ('cclo')
    )
    CODEC_MANUFACTURER = 1668112750  # kExtAudioFileProperty_CodecManufacturer ('cman')
    AUDIO_FILE = 1634101612  # kExtAudioFileProperty_AudioFile ('afil')
    FILE_LENGTH_FRAMES = 593916525  # kExtAudioFileProperty_FileLengthFrames ('#frm')
    AUDIO_CONVERTER = 1633906294  # kExtAudioFileProperty_AudioConverter ('acnv')
    CLIENT_MAX_PACKET_SIZE = (
        1668116595  # kExtAudioFileProperty_ClientMaxPacketSize ('cmps')
    )
