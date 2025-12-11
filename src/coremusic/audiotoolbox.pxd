# audiotoolbox.pxd
# AudioToolbox framework declarations for coremusic

from .coreaudio cimport *
from .coreaudiotypes cimport *
from .corefoundation cimport *

# -----------------------------------------------------------------------------

cdef extern from "AudioToolbox/AudioQueue.h":

    ctypedef UInt32 AudioQueuePropertyID
    ctypedef UInt32 AudioQueueParameterID
    ctypedef Float32 AudioQueueParameterValue
    ctypedef struct OpaqueAudioQueue
    ctypedef OpaqueAudioQueue* AudioQueueRef
    ctypedef struct OpaqueAudioQueueTimeline
    ctypedef OpaqueAudioQueueTimeline* AudioQueueTimelineRef

    ctypedef struct AudioQueueBuffer:
        UInt32 mAudioDataBytesCapacity
        void* mAudioData
        UInt32 mAudioDataByteSize
        void* mUserData
        UInt32 mPacketDescriptionCapacity
        # AudioStreamPacketDescription* mPacketDescriptions
        UInt32 mPacketDescriptionCount

    ctypedef AudioQueueBuffer* AudioQueueBufferRef

    ctypedef void (*AudioQueueOutputCallback)(void* inUserData, AudioQueueRef inAQ, AudioQueueBufferRef inBuffer)
    ctypedef void (*AudioQueueInputCallback)(void* inUserData, AudioQueueRef inAQ, AudioQueueBufferRef inBuffer, const AudioTimeStamp* inStartTime, UInt32 inNumberPacketDescriptions, const AudioStreamPacketDescription* inPacketDescs)

    ctypedef enum:
        kAudioQueueErr_InvalidBuffer = -66687
        kAudioQueueErr_BufferEmpty = -66686
        kAudioQueueErr_DisposalPending = -66685
        kAudioQueueErr_InvalidProperty = -66684
        kAudioQueueErr_InvalidPropertySize = -66683
        kAudioQueueErr_InvalidParameter = -66682
        kAudioQueueErr_CannotStart = -66681
        kAudioQueueErr_InvalidDevice = -66680
        kAudioQueueErr_BufferInQueue = -66679
        kAudioQueueErr_InvalidRunState = -66678
        kAudioQueueErr_InvalidQueueType = -66677
        kAudioQueueErr_Permissions = -66676
        kAudioQueueErr_InvalidPropertyValue = -66675
        kAudioQueueErr_PrimeTimedOut = -66674
        kAudioQueueErr_CodecNotFound = -66673
        kAudioQueueErr_InvalidCodecAccess = -66672
        kAudioQueueErr_QueueInvalidated = -66671
        kAudioQueueErr_TooManyTaps = -66670
        kAudioQueueErr_InvalidTapContext = -66669
        kAudioQueueErr_RecordUnderrun = -66668
        kAudioQueueErr_InvalidTapType = -66667
        kAudioQueueErr_BufferEnqueuedTwice = -66666
        kAudioQueueErr_CannotStartYet = -66665
        kAudioQueueErr_EnqueueDuringReset = -66632
        kAudioQueueErr_InvalidOfflineMode = -66626

    cdef OSStatus AudioQueueNewOutput(const AudioStreamBasicDescription* inFormat, AudioQueueOutputCallback inCallbackProc, void* inUserData, CFRunLoopRef inCallbackRunLoop, CFStringRef inCallbackRunLoopMode, UInt32 inFlags, AudioQueueRef* outAQ)
    cdef OSStatus AudioQueueNewInput(const AudioStreamBasicDescription* inFormat, AudioQueueInputCallback inCallbackProc, void* inUserData, CFRunLoopRef inCallbackRunLoop, CFStringRef inCallbackRunLoopMode, UInt32 inFlags, AudioQueueRef* outAQ)
    cdef OSStatus AudioQueueDispose(AudioQueueRef inAQ, Boolean inImmediate)
    cdef OSStatus AudioQueueAllocateBuffer(AudioQueueRef inAQ, UInt32 inBufferByteSize, AudioQueueBufferRef* outBuffer)
    cdef OSStatus AudioQueueAllocateBufferWithPacketDescriptions(AudioQueueRef inAQ, UInt32 inBufferByteSize, UInt32 inNumberPacketDescriptions, AudioQueueBufferRef* outBuffer)
    cdef OSStatus AudioQueueFreeBuffer(AudioQueueRef inAQ, AudioQueueBufferRef inBuffer)
    cdef OSStatus AudioQueueEnqueueBuffer(AudioQueueRef inAQ, AudioQueueBufferRef inBuffer, UInt32 inNumPacketDescs, const void* inPacketDescs) nogil
    cdef OSStatus AudioQueueEnqueueBufferWithParameters(AudioQueueRef inAQ, AudioQueueBufferRef inBuffer, UInt32 inNumPacketDescs, const void* inPacketDescs, UInt32 inTrimFramesAtStart, UInt32 inTrimFramesAtEnd, UInt32 inNumParamValues, const AudioQueueParameterEvent* inParamValues, const AudioTimeStamp* inStartTime, AudioTimeStamp* outActualStartTime) nogil
    cdef OSStatus AudioQueueStart(AudioQueueRef inAQ, const AudioTimeStamp* inStartTime)
    cdef OSStatus AudioQueuePrime(AudioQueueRef inAQ, UInt32 inNumberOfFramesToPrepare, UInt32* outNumberOfFramesPrepared)
    cdef OSStatus AudioQueueStop(AudioQueueRef inAQ, Boolean inImmediate)
    cdef OSStatus AudioQueuePause(AudioQueueRef inAQ)
    cdef OSStatus AudioQueueFlush(AudioQueueRef inAQ)
    cdef OSStatus AudioQueueReset(AudioQueueRef inAQ)
    cdef OSStatus AudioQueueGetCurrentTime(AudioQueueRef inAQ, AudioQueueTimelineRef inTimeline, AudioTimeStamp* outTimeStamp, Boolean* outTimelineDiscontinuity)


cdef extern from "AudioToolbox/AudioFile.h":

    ctypedef UInt32 AudioFileTypeID
    ctypedef UInt32 AudioFilePropertyID
    ctypedef UInt32 AudioFileFlags
    ctypedef struct OpaqueAudioFileID
    ctypedef OpaqueAudioFileID* AudioFileID

    # AudioFileFlags constants
    ctypedef enum:
        kAudioFileFlags_EraseFile = 1
        kAudioFileFlags_DontPageAlignAudioData = 2

    # Audio file type IDs
    ctypedef enum:
        kAudioFileAIFFType = 1095321158  # 'AIFF'
        kAudioFileAIFCType = 1095321155  # 'AIFC'
        kAudioFileWAVEType = 1463899717  # 'WAVE'
        kAudioFileRF64Type = 1380333108  # 'RF64'
        kAudioFileBW64Type = 1112493108  # 'BW64'
        kAudioFileWave64Type = 1463900518  # 'W64f'
        kAudioFileSoundDesigner2Type = 1399075686  # 'Sd2f'
        kAudioFileNextType = 1315264596  # 'NeXT'
        kAudioFileMP3Type = 1297106739  # 'MPG3'
        kAudioFileMP2Type = 1297106738  # 'MPG2'
        kAudioFileMP1Type = 1297106737  # 'MPG1'
        kAudioFileAC3Type = 1633889587  # 'ac-3'
        kAudioFileAAC_ADTSType = 1633973363  # 'adts'
        kAudioFileMPEG4Type = 1836069990  # 'mp4f'
        kAudioFileM4AType = 1832149350  # 'm4af'
        kAudioFileM4BType = 1832149606  # 'm4bf'
        kAudioFileCAFType = 1667327590  # 'caff'
        kAudioFile3GPType = 862417008   # '3gpp'
        kAudioFile3GP2Type = 862416690  # '3gp2'
        kAudioFileAMRType = 1634562662  # 'amrf'
        kAudioFileFLACType = 1718378851 # 'flac'
        kAudioFileLATMInLOASType = 1819238259 # 'loas'

    ctypedef enum:
        kAudioFileReadPermission = 1
        kAudioFileWritePermission = 2
        kAudioFileReadWritePermission = 3

    ctypedef enum:
        kAudioFilePropertyFileFormat = 1717988724         # 'ffmt'
        kAudioFilePropertyDataFormat = 1684103783         # 'dfmt'
        kAudioFilePropertyIsOptimized = 1869640813        # 'optm'
        kAudioFilePropertyMagicCookieData = 1835493731    # 'mgic'
        kAudioFilePropertyAudioDataByteCount = 1650683508 # 'bcnt'
        kAudioFilePropertyAudioDataPacketCount = 1885564532 # 'pcnt'
        kAudioFilePropertyMaximumPacketSize = 1886616691  # 'psze'
        kAudioFilePropertyDataOffset = 1685022310         # 'doff'
        kAudioFilePropertyChannelLayout = 1668112500      # 'cmap'
        kAudioFilePropertyDeferSizeUpdates = 1684238953   # 'dszu'
        kAudioFilePropertyDataFormatName = 1718512996     # 'fnme'
        kAudioFilePropertyMarkerList = 1835756403         # 'mkls'
        kAudioFilePropertyRegionList = 1919380595         # 'rgls'
        kAudioFilePropertyPacketToFrame = 1886086770      # 'pkfr'
        kAudioFilePropertyFrameToPacket = 1718775151      # 'frpk'
        kAudioFilePropertyPacketToByte = 1886085753       # 'pkby'
        kAudioFilePropertyByteToPacket = 1652125803       # 'bypk'
        kAudioFilePropertyChunkIDs = 1667787108           # 'chid'
        kAudioFilePropertyInfoDictionary = 1768842863     # 'info'
        kAudioFilePropertyPacketTableInfo = 1886283375    # 'pnfo'
        kAudioFilePropertyFormatList = 1718383476         # 'flst'
        kAudioFilePropertyPacketSizeUpperBound = 1886090093 # 'pkub'
        kAudioFilePropertyReserveDuration = 1920365423    # 'rsrv'
        kAudioFilePropertyEstimatedDuration = 1701082482  # 'edur'
        kAudioFilePropertyBitRate = 1651663220            # 'brat'
        kAudioFilePropertyID3Tag = 1768174180             # 'id3 '
        kAudioFilePropertySourceBitDepth = 1935832164     # 'sbtd'
        kAudioFilePropertyAlbumArtwork = 1635015020       # 'aart'
        kAudioFilePropertyAudioTrackCount = 1635017588    # 'atct'
        kAudioFilePropertyUseAudioTrack = 1969385580      # 'uatk'

    cdef OSStatus AudioFileOpenURL(CFURLRef inFileRef, AudioFilePermissions inPermissions, AudioFileTypeID inFileTypeHint, AudioFileID* outAudioFile)
    cdef OSStatus AudioFileCreateWithURL(CFURLRef inFileRef, AudioFileTypeID inFileType, const AudioStreamBasicDescription* inFormat, AudioFileFlags inFlags, AudioFileID* outAudioFile)
    cdef OSStatus AudioFileClose(AudioFileID inAudioFile)
    cdef OSStatus AudioFileOptimize(AudioFileID inAudioFile)
    cdef OSStatus AudioFileReadBytes(AudioFileID inAudioFile, Boolean inUseCache, SInt64 inStartingByte, UInt32* ioNumBytes, void* outBuffer)
    cdef OSStatus AudioFileWriteBytes(AudioFileID inAudioFile, Boolean inUseCache, SInt64 inStartingByte, UInt32* ioNumBytes, const void* inBuffer)
    cdef OSStatus AudioFileReadPackets(AudioFileID inAudioFile, Boolean inUseCache, UInt32* ioNumBytes, void* outPacketDescriptions, SInt64 inStartingPacket, UInt32* ioNumPackets, void* outBuffer)
    cdef OSStatus AudioFileReadPacketData(AudioFileID inAudioFile, Boolean inUseCache, UInt32* ioNumBytes, void* outPacketDescriptions, SInt64 inStartingPacket, UInt32* ioNumPackets, void* outBuffer)
    cdef OSStatus AudioFileWritePackets(AudioFileID inAudioFile, Boolean inUseCache, UInt32 inNumBytes, const void* inPacketDescriptions, SInt64 inStartingPacket, UInt32* ioNumPackets, const void* inBuffer)
    cdef OSStatus AudioFileCountUserData(AudioFileID inAudioFile, UInt32 inUserDataID, UInt32* outNumberItems)
    cdef OSStatus AudioFileGetUserDataSize(AudioFileID inAudioFile, UInt32 inUserDataID, UInt32 inIndex, UInt32* outUserDataSize)
    cdef OSStatus AudioFileGetUserData(AudioFileID inAudioFile, UInt32 inUserDataID, UInt32 inIndex, UInt32* ioUserDataSize, void* outUserData)
    cdef OSStatus AudioFileSetUserData(AudioFileID inAudioFile, UInt32 inUserDataID, UInt32 inIndex, UInt32 inUserDataSize, const void* inUserData)
    cdef OSStatus AudioFileRemoveUserData(AudioFileID inAudioFile, UInt32 inUserDataID, UInt32 inIndex)
    cdef OSStatus AudioFileGetPropertyInfo(AudioFileID inAudioFile, AudioFilePropertyID inPropertyID, UInt32* outDataSize, UInt32* isWritable)
    cdef OSStatus AudioFileGetProperty(AudioFileID inAudioFile, AudioFilePropertyID inPropertyID, UInt32* ioDataSize, void* outPropertyData)
    cdef OSStatus AudioFileSetProperty(AudioFileID inAudioFile, AudioFilePropertyID inPropertyID, UInt32 inDataSize, const void* inPropertyData)

    ctypedef struct AudioQueueParameterEvent:
        AudioQueueParameterID mID
        AudioQueueParameterValue mValue


cdef extern from "AudioToolbox/AudioFileStream.h":

    # AudioFileStream types
    ctypedef UInt32 AudioFileStreamPropertyID
    ctypedef struct OpaqueAudioFileStreamID
    ctypedef OpaqueAudioFileStreamID* AudioFileStreamID

    # AudioFileStream flags
    ctypedef enum AudioFileStreamPropertyFlags:
        kAudioFileStreamPropertyFlag_PropertyIsCached = 1
        kAudioFileStreamPropertyFlag_CacheProperty = 2

    ctypedef enum AudioFileStreamParseFlags:
        kAudioFileStreamParseFlag_Discontinuity = 1

    ctypedef enum AudioFileStreamSeekFlags:
        kAudioFileStreamSeekFlag_OffsetIsEstimated = 1

    # AudioFileStream callback function types
    ctypedef void (*AudioFileStream_PropertyListenerProc)(
        void* inClientData,
        AudioFileStreamID inAudioFileStream,
        AudioFileStreamPropertyID inPropertyID,
        AudioFileStreamPropertyFlags* ioFlags)

    ctypedef void (*AudioFileStream_PacketsProc)(
        void* inClientData,
        UInt32 inNumberBytes,
        UInt32 inNumberPackets,
        const void* inInputData,
        AudioStreamPacketDescription* inPacketDescriptions)

    # AudioFileStream error codes
    ctypedef enum:
        kAudioFileStreamError_UnsupportedFileType = 1953064820  # 'typ?'
        kAudioFileStreamError_UnsupportedDataFormat = 1718449215  # 'fmt?'
        kAudioFileStreamError_UnsupportedProperty = 1886547839  # 'pty?'
        kAudioFileStreamError_BadPropertySize = 561211770  # '!siz'
        kAudioFileStreamError_NotOptimized = 1869640813  # 'optm'
        kAudioFileStreamError_InvalidPacketOffset = 1885433391  # 'pck?'
        kAudioFileStreamError_InvalidFile = 1685348415  # 'dta?'
        kAudioFileStreamError_ValueUnknown = 1970170687  # 'unk?'
        kAudioFileStreamError_DataUnavailable = 1836016741  # 'more'
        kAudioFileStreamError_IllegalOperation = 1852139888  # 'nope'
        kAudioFileStreamError_UnspecifiedError = 2003395684  # 'wht?'
        kAudioFileStreamError_DiscontinuityCantRecover = 1685348641  # 'dsc!'

    # AudioFileStream property IDs
    ctypedef enum:
        kAudioFileStreamProperty_ReadyToProducePackets = 1919247473  # 'redy'
        kAudioFileStreamProperty_FileFormat = 1717988724  # 'ffmt'
        kAudioFileStreamProperty_DataFormat = 1684103783  # 'dfmt'
        kAudioFileStreamProperty_FormatList = 1718383476  # 'flst'
        kAudioFileStreamProperty_MagicCookieData = 1835493731  # 'mgic'
        kAudioFileStreamProperty_AudioDataByteCount = 1650683508  # 'bcnt'
        kAudioFileStreamProperty_AudioDataPacketCount = 1885564532  # 'pcnt'
        kAudioFileStreamProperty_MaximumPacketSize = 1886616691  # 'psze'
        kAudioFileStreamProperty_DataOffset = 1685022310  # 'doff'
        kAudioFileStreamProperty_ChannelLayout = 1668112500  # 'cmap'
        kAudioFileStreamProperty_PacketToFrame = 1886086770  # 'pkfr'
        kAudioFileStreamProperty_FrameToPacket = 1718775151  # 'frpk'
        kAudioFileStreamProperty_RestrictsRandomAccess = 1919508592  # 'rrap'
        kAudioFileStreamProperty_PacketToRollDistance = 1886090604  # 'pkrl'
        kAudioFileStreamProperty_PreviousIndependentPacket = 1885957228  # 'pind'
        kAudioFileStreamProperty_NextIndependentPacket = 1852273252  # 'nind'
        kAudioFileStreamProperty_PacketToDependencyInfo = 1886086256  # 'pkdp'
        kAudioFileStreamProperty_PacketToByte = 1886085753  # 'pkby'
        kAudioFileStreamProperty_ByteToPacket = 1652125803  # 'bypk'
        kAudioFileStreamProperty_PacketTableInfo = 1886283375  # 'pnfo'
        kAudioFileStreamProperty_PacketSizeUpperBound = 1886090093  # 'pkub'
        kAudioFileStreamProperty_AverageBytesPerPacket = 1633969264  # 'abpp'
        kAudioFileStreamProperty_BitRate = 1651663220  # 'brat'
        kAudioFileStreamProperty_InfoDictionary = 1768842863  # 'info'

    # AudioFileStream functions
    cdef OSStatus AudioFileStreamOpen(
        void* inClientData,
        AudioFileStream_PropertyListenerProc inPropertyListenerProc,
        AudioFileStream_PacketsProc inPacketsProc,
        AudioFileTypeID inFileTypeHint,
        AudioFileStreamID* outAudioFileStream)

    cdef OSStatus AudioFileStreamParseBytes(
        AudioFileStreamID inAudioFileStream,
        UInt32 inDataByteSize,
        const void* inData,
        AudioFileStreamParseFlags inFlags)

    cdef OSStatus AudioFileStreamSeek(
        AudioFileStreamID inAudioFileStream,
        SInt64 inPacketOffset,
        SInt64* outDataByteOffset,
        AudioFileStreamSeekFlags* ioFlags)

    cdef OSStatus AudioFileStreamGetPropertyInfo(
        AudioFileStreamID inAudioFileStream,
        AudioFileStreamPropertyID inPropertyID,
        UInt32* outPropertyDataSize,
        Boolean* outWritable)

    cdef OSStatus AudioFileStreamGetProperty(
        AudioFileStreamID inAudioFileStream,
        AudioFileStreamPropertyID inPropertyID,
        UInt32* ioPropertyDataSize,
        void* outPropertyData)

    cdef OSStatus AudioFileStreamSetProperty(
        AudioFileStreamID inAudioFileStream,
        AudioFileStreamPropertyID inPropertyID,
        UInt32 inPropertyDataSize,
        const void* inPropertyData)

    cdef OSStatus AudioFileStreamClose(AudioFileStreamID inAudioFileStream)


cdef extern from "AudioToolbox/AudioComponent.h":

    ctypedef struct OpaqueAudioComponent
    ctypedef OpaqueAudioComponent* AudioComponent
    ctypedef struct OpaqueAudioComponentInstance
    ctypedef OpaqueAudioComponentInstance* AudioComponentInstance

    ctypedef struct AudioComponentDescription:
        UInt32 componentType
        UInt32 componentSubType
        UInt32 componentManufacturer
        UInt32 componentFlags
        UInt32 componentFlagsMask

    ctypedef OSStatus (*AudioComponentFactoryFunction)(AudioComponentDescription* inDescription, AudioComponent inComponent, AudioComponentInstance* outInstance)
    ctypedef void (*AudioComponentInitializer)(AudioComponentInstance inInstance)

    # AudioComponent constants
    ctypedef enum:
        kAudioComponentType_Output = 1635086197          # 'auou'
        kAudioComponentType_MusicDevice = 1635085685     # 'aumu'
        kAudioComponentType_MusicEffect = 1635085670     # 'aumf'
        kAudioComponentType_FormatConverter = 1635083875 # 'aufc'
        kAudioComponentType_Effect = 1635083896          # 'aufx'
        kAudioComponentType_Mixer = 1635085688           # 'aumx'
        kAudioComponentType_Panner = 1635086446          # 'aupn'
        kAudioComponentType_Generator = 1635084142       # 'augn'
        kAudioComponentType_OfflineEffect = 1635085676   # 'auol'
        kAudioComponentType_MIDIProcessor = 1635085673   # 'aumi'

        kAudioComponentSubType_DefaultOutput = 1684366880        # 'def '
        kAudioComponentSubType_HALOutput = 1634230896            # 'ahal'
        kAudioComponentSubType_SystemOutput = 1937339252         # 'sys '
        kAudioComponentSubType_GenericOutput = 1734700658       # 'genr'

        kAudioComponentManufacturer_Apple = 1634758764  # 'appl'

    # AudioComponent functions
    cdef AudioComponent AudioComponentFindNext(AudioComponent inComponent, const AudioComponentDescription* inDesc)
    cdef OSStatus AudioComponentCopyName(AudioComponent inComponent, CFStringRef* outName)
    cdef OSStatus AudioComponentGetDescription(AudioComponent inComponent, AudioComponentDescription* outDesc)
    cdef OSStatus AudioComponentGetVersion(AudioComponent inComponent, UInt32* outVersion)
    cdef OSStatus AudioComponentInstanceNew(AudioComponent inComponent, AudioComponentInstance* outInstance)
    cdef OSStatus AudioComponentInstanceDispose(AudioComponentInstance inInstance)
    cdef OSStatus AudioComponentInstanceCanDo(AudioComponentInstance inInstance, SInt16 inSelectorID)


cdef extern from "AudioToolbox/AudioUnitProperties.h":
    pass


cdef extern from "AudioToolbox/AUComponent.h":

    # AudioUnit is typedef'd as AudioComponentInstance
    ctypedef AudioComponentInstance AudioUnit

    # AudioUnit Types (same as AudioComponent types but with different name)
    ctypedef enum:
        kAudioUnitType_Output = 1635086197               # 'auou'
        kAudioUnitType_MusicDevice = 1635085685          # 'aumu'
        kAudioUnitType_MusicEffect = 1635085670          # 'aumf'
        kAudioUnitType_FormatConverter = 1635083875      # 'aufc'
        kAudioUnitType_Effect = 1635083896               # 'aufx'
        kAudioUnitType_Mixer = 1635085688                # 'aumx'
        kAudioUnitType_Panner = 1635086446               # 'aupn'
        kAudioUnitType_Generator = 1635084142            # 'augn'
        kAudioUnitType_OfflineEffect = 1635085676        # 'auol'
        kAudioUnitType_MIDIProcessor = 1635085673        # 'aumi'

        kAudioUnitSubType_DefaultOutput = 1684366880     # 'def '
        kAudioUnitSubType_HALOutput = 1634230896         # 'ahal'
        kAudioUnitSubType_SystemOutput = 1937339252      # 'sys '
        kAudioUnitSubType_GenericOutput = 1734700658     # 'genr'

        kAudioUnitManufacturer_Apple = 1634758764        # 'appl'

    # AudioUnit Property IDs
    ctypedef UInt32 AudioUnitPropertyID
    ctypedef UInt32 AudioUnitScope
    ctypedef UInt32 AudioUnitElement
    ctypedef UInt32 AudioUnitParameterID
    ctypedef Float32 AudioUnitParameterValue

    ctypedef enum:
        # Global scope properties
        kAudioUnitProperty_ClassInfo = 0
        kAudioUnitProperty_MakeConnection = 1
        kAudioUnitProperty_SampleRate = 2
        kAudioUnitProperty_ParameterList = 3
        kAudioUnitProperty_ParameterInfo = 4
        kAudioUnitProperty_CPULoad = 6
        kAudioUnitProperty_StreamFormat = 8
        kAudioUnitProperty_ElementCount = 11
        kAudioUnitProperty_Latency = 12
        kAudioUnitProperty_SupportedNumChannels = 13
        kAudioUnitProperty_MaximumFramesPerSlice = 14
        kAudioUnitProperty_ParameterValueStrings = 16
        kAudioUnitProperty_AudioChannelLayout = 19
        kAudioUnitProperty_TailTime = 20
        kAudioUnitProperty_BypassEffect = 21
        kAudioUnitProperty_LastRenderError = 22
        kAudioUnitProperty_SetRenderCallback = 23
        kAudioUnitProperty_FactoryPresets = 24
        kAudioUnitProperty_RenderQuality = 26
        kAudioUnitProperty_HostCallbacks = 27
        kAudioUnitProperty_InPlaceProcessing = 29
        kAudioUnitProperty_ElementName = 30
        kAudioUnitProperty_SupportedChannelLayoutTags = 32
        kAudioUnitProperty_PresentPreset = 36
        kAudioUnitProperty_OfflineRender = 37
        kAudioUnitProperty_ParameterValueName = 38
        kAudioUnitProperty_ParameterStringFromValue = 39
        kAudioUnitProperty_ParameterValueFromString = 40
        kAudioUnitProperty_NickName = 54

    # AudioUnit Parameter Units
    ctypedef UInt32 AudioUnitParameterUnit
    ctypedef enum:
        kAudioUnitParameterUnit_Generic = 0
        kAudioUnitParameterUnit_Indexed = 1
        kAudioUnitParameterUnit_Boolean = 2
        kAudioUnitParameterUnit_Percent = 3
        kAudioUnitParameterUnit_Seconds = 4
        kAudioUnitParameterUnit_SampleFrames = 5
        kAudioUnitParameterUnit_Phase = 6
        kAudioUnitParameterUnit_Rate = 7
        kAudioUnitParameterUnit_Hertz = 8
        kAudioUnitParameterUnit_Cents = 9
        kAudioUnitParameterUnit_RelativeSemiTones = 10
        kAudioUnitParameterUnit_MIDINoteNumber = 11
        kAudioUnitParameterUnit_MIDIController = 12
        kAudioUnitParameterUnit_Decibels = 13
        kAudioUnitParameterUnit_LinearGain = 14
        kAudioUnitParameterUnit_Degrees = 15
        kAudioUnitParameterUnit_EqualPowerCrossfade = 16
        kAudioUnitParameterUnit_MixerFaderCurve1 = 17
        kAudioUnitParameterUnit_Pan = 18
        kAudioUnitParameterUnit_Meters = 19
        kAudioUnitParameterUnit_AbsoluteCents = 20
        kAudioUnitParameterUnit_Octaves = 21
        kAudioUnitParameterUnit_BPM = 22
        kAudioUnitParameterUnit_Beats = 23
        kAudioUnitParameterUnit_Milliseconds = 24
        kAudioUnitParameterUnit_Ratio = 25
        kAudioUnitParameterUnit_CustomUnit = 26

    # AudioUnit Parameter Flags
    ctypedef UInt32 AudioUnitParameterFlags
    ctypedef enum:
        kAudioUnitParameterFlag_CFNameRelease = (1 << 4)
        kAudioUnitParameterFlag_HasClump = (1 << 20)
        kAudioUnitParameterFlag_HasName = (1 << 21)
        kAudioUnitParameterFlag_DisplayLogarithmic = (1 << 22)
        kAudioUnitParameterFlag_DisplaySquareRoot = (1 << 23)
        kAudioUnitParameterFlag_DisplaySquared = (1 << 24)
        kAudioUnitParameterFlag_DisplayCubed = (1 << 25)
        kAudioUnitParameterFlag_DisplayCubeRoot = (1 << 26)
        kAudioUnitParameterFlag_DisplayExponential = (1 << 27)
        kAudioUnitParameterFlag_HasCFNameString = (1 << 28)
        kAudioUnitParameterFlag_IsGlobalMeta = (1 << 29)
        kAudioUnitParameterFlag_IsElementMeta = (1 << 30)
        kAudioUnitParameterFlag_IsReadable = (1 << 31)
        kAudioUnitParameterFlag_IsWritable = (1 << 31) >> 1

    # AudioUnit Parameter Info structure
    ctypedef struct AudioUnitParameterInfo:
        char name[52]
        CFStringRef unitName
        UInt32 clumpID
        CFStringRef cfNameString
        AudioUnitParameterUnit unit
        AudioUnitParameterValue minValue
        AudioUnitParameterValue maxValue
        AudioUnitParameterValue defaultValue
        UInt32 flags

    # AudioUnit Preset (AUPreset) structure
    ctypedef struct AUPreset:
        SInt32 presetNumber
        CFStringRef presetName

    # AudioUnit Preset dictionary keys (for kAudioUnitProperty_FactoryPresets)
    # These are #define macros, not CFStringRef constants
    # #define kAUPresetNumberKey "preset-number"
    # #define kAUPresetNameKey "name"

    # AudioUnit Parameter Value translation structure
    ctypedef struct AudioUnitParameterValueTranslation:
        AudioUnit inUnit
        AudioUnitParameterID inParameterID
        AudioUnitScope inScope
        AudioUnitElement inElement
        CFStringRef inString
        AudioUnitParameterValue outValue

    # AudioUnit Scopes
    ctypedef enum:
        kAudioUnitScope_Global = 0
        kAudioUnitScope_Input = 1
        kAudioUnitScope_Output = 2
        kAudioUnitScope_Group = 3
        kAudioUnitScope_Part = 4
        kAudioUnitScope_Note = 5
        kAudioUnitScope_Layer = 6
        kAudioUnitScope_LayerItem = 7

    # Render callback function type
    ctypedef UInt32 AudioUnitRenderActionFlags
    ctypedef OSStatus (*AURenderCallback)(void* inRefCon,
                                           AudioUnitRenderActionFlags* ioActionFlags,
                                           const AudioTimeStamp* inTimeStamp,
                                           UInt32 inBusNumber,
                                           UInt32 inNumberFrames,
                                           AudioBufferList* ioData)

    ctypedef struct AURenderCallbackStruct:
        AURenderCallback inputProc
        void* inputProcRefCon

    # AudioUnit functions
    cdef OSStatus AudioUnitInitialize(AudioUnit inUnit)
    cdef OSStatus AudioUnitUninitialize(AudioUnit inUnit)
    cdef OSStatus AudioUnitGetPropertyInfo(AudioUnit inUnit, AudioUnitPropertyID inID, AudioUnitScope inScope, AudioUnitElement inElement, UInt32* outDataSize, Boolean* outWritable)
    cdef OSStatus AudioUnitGetProperty(AudioUnit inUnit, AudioUnitPropertyID inID, AudioUnitScope inScope, AudioUnitElement inElement, void* outData, UInt32* ioDataSize)
    cdef OSStatus AudioUnitSetProperty(AudioUnit inUnit, AudioUnitPropertyID inID, AudioUnitScope inScope, AudioUnitElement inElement, const void* inData, UInt32 inDataSize)
    cdef OSStatus AudioUnitAddPropertyListener(AudioUnit inUnit, AudioUnitPropertyID inID, void* inProc, void* inProcUserData)
    cdef OSStatus AudioUnitRemovePropertyListener(AudioUnit inUnit, AudioUnitPropertyID inID, void* inProc)
    cdef OSStatus AudioUnitRemovePropertyListenerWithUserData(AudioUnit inUnit, AudioUnitPropertyID inID, void* inProc, void* inProcUserData)
    cdef OSStatus AudioUnitAddRenderNotify(AudioUnit inUnit, AURenderCallback inProc, void* inProcUserData)
    cdef OSStatus AudioUnitRemoveRenderNotify(AudioUnit inUnit, AURenderCallback inProc, void* inProcUserData)
    cdef OSStatus AudioUnitGetParameter(AudioUnit inUnit, AudioUnitParameterID inID, AudioUnitScope inScope, AudioUnitElement inElement, AudioUnitParameterValue* outValue)
    cdef OSStatus AudioUnitSetParameter(AudioUnit inUnit, AudioUnitParameterID inID, AudioUnitScope inScope, AudioUnitElement inElement, AudioUnitParameterValue inValue, UInt32 inBufferOffsetInFrames)
    cdef OSStatus AudioUnitScheduleParameters(AudioUnit inUnit, const void* inParameterEvent, UInt32 inNumParamEvents)
    cdef OSStatus AudioUnitRender(AudioUnit inUnit, AudioUnitRenderActionFlags* ioActionFlags, const AudioTimeStamp* inTimeStamp, UInt32 inOutputBusNumber, UInt32 inNumberFrames, AudioBufferList* ioData)
    cdef OSStatus AudioUnitProcess(AudioUnit inUnit, AudioUnitRenderActionFlags* ioActionFlags, const AudioTimeStamp* inTimeStamp, UInt32 inNumberFrames, AudioBufferList* ioData)
    cdef OSStatus AudioUnitProcessMultiple(AudioUnit inUnit, AudioUnitRenderActionFlags* ioActionFlags, const AudioTimeStamp* inTimeStamp, UInt32 inNumberFrames, UInt32 inNumberInputBufferLists, const AudioBufferList** inInputBufferLists, UInt32 inNumberOutputBufferLists, AudioBufferList** ioOutputBufferLists)
    cdef OSStatus AudioUnitReset(AudioUnit inUnit, AudioUnitScope inScope, AudioUnitElement inElement)


cdef extern from "AudioToolbox/AudioOutputUnit.h":

    cdef OSStatus AudioOutputUnitStart(AudioUnit ci)
    cdef OSStatus AudioOutputUnitStop(AudioUnit ci)


cdef extern from "AudioToolbox/AudioConverter.h":

    # AudioConverter types
    ctypedef struct OpaqueAudioConverter
    ctypedef OpaqueAudioConverter* AudioConverterRef
    ctypedef UInt32 AudioConverterPropertyID

    ctypedef enum AudioConverterOptions:
        kAudioConverterOption_Unbuffered = 65536

    # AudioConverter property IDs
    ctypedef enum:
        kAudioConverterPropertyMinimumInputBufferSize = 1835623027  # 'mibs'
        kAudioConverterPropertyMinimumOutputBufferSize = 1836016243  # 'mobs'
        kAudioConverterPropertyMaximumInputPacketSize = 2020175987  # 'xips'
        kAudioConverterPropertyMaximumOutputPacketSize = 2020569203  # 'xops'
        kAudioConverterPropertyCalculateInputBufferSize = 1667850867  # 'cibs'
        kAudioConverterPropertyCalculateOutputBufferSize = 1668244083  # 'cobs'
        kAudioConverterPropertyInputCodecParameters = 1768121456  # 'icdp'
        kAudioConverterPropertyOutputCodecParameters = 1868784752  # 'ocdp'
        kAudioConverterSampleRateConverterComplexity = 1936876385  # 'srca'
        kAudioConverterSampleRateConverterQuality = 1936876401  # 'srcq'
        kAudioConverterSampleRateConverterInitialPhase = 1936876400  # 'srcp'
        kAudioConverterCodecQuality = 1667527029  # 'cdqu'
        kAudioConverterPrimeMethod = 1886547309  # 'prmm'
        kAudioConverterPrimeInfo = 1886546285  # 'prim'
        kAudioConverterChannelMap = 1667788144  # 'chmp'
        kAudioConverterDecompressionMagicCookie = 1684891491  # 'dmgc'
        kAudioConverterCompressionMagicCookie = 1668114275  # 'cmgc'
        kAudioConverterEncodeBitRate = 1651663220  # 'brat'
        kAudioConverterEncodeAdjustableSampleRate = 1634366322  # 'ajsr'
        kAudioConverterInputChannelLayout = 1768123424  # 'icl '
        kAudioConverterOutputChannelLayout = 1868786720  # 'ocl '
        kAudioConverterApplicableEncodeBitRates = 1634034290  # 'aebr'
        kAudioConverterAvailableEncodeBitRates = 1986355826  # 'vebr'
        kAudioConverterApplicableEncodeSampleRates = 1634038642  # 'aesr'
        kAudioConverterAvailableEncodeSampleRates = 1986360178  # 'vesr'
        kAudioConverterAvailableEncodeChannelLayoutTags = 1634034540  # 'aecl'
        kAudioConverterCurrentOutputStreamDescription = 1633906532  # 'acod'
        kAudioConverterCurrentInputStreamDescription = 1633904996  # 'acid'
        kAudioConverterPropertySettings = 1633906803  # 'acps'
        kAudioConverterPropertyBitDepthHint = 1633903204  # 'acbd'
        kAudioConverterPropertyFormatList = 1718383476  # 'flst'

    # macOS-only properties
    ctypedef enum:
        kAudioConverterPropertyDithering = 1684632680  # 'dith'
        kAudioConverterPropertyDitherBitDepth = 1684171124  # 'dbit'

    # Quality constants
    ctypedef enum:
        kAudioConverterQuality_Max = 127
        kAudioConverterQuality_High = 96
        kAudioConverterQuality_Medium = 64
        kAudioConverterQuality_Low = 32
        kAudioConverterQuality_Min = 0

    # Sample rate converter complexity
    ctypedef enum:
        kAudioConverterSampleRateConverterComplexity_Linear = 1818846821  # 'line'
        kAudioConverterSampleRateConverterComplexity_Normal = 1852797541  # 'norm'
        kAudioConverterSampleRateConverterComplexity_Mastering = 1651471971  # 'bats'
        kAudioConverterSampleRateConverterComplexity_MinimumPhase = 1835622000  # 'minp'

    # Prime method constants
    ctypedef enum:
        kConverterPrimeMethod_Pre = 0
        kConverterPrimeMethod_Normal = 1
        kConverterPrimeMethod_None = 2

    # Dithering algorithms
    ctypedef enum:
        kDitherAlgorithm_TPDF = 1
        kDitherAlgorithm_NoiseShaping = 2

    # AudioConverterPrimeInfo structure
    ctypedef struct AudioConverterPrimeInfo:
        UInt32 leadingFrames
        UInt32 trailingFrames

    # Error codes
    ctypedef enum:
        kAudioConverterErr_FormatNotSupported = 1718447200  # 'fmt?'
        kAudioConverterErr_OperationNotSupported = 1869638207  # 'op??'
        kAudioConverterErr_PropertyNotSupported = 1886547824  # 'prop'
        kAudioConverterErr_InvalidInputSize = 1768845682  # 'insz'
        kAudioConverterErr_InvalidOutputSize = 1869771634  # 'otsz'
        kAudioConverterErr_UnspecifiedError = 2003395684  # 'what'
        kAudioConverterErr_BadPropertySizeError = 1937010802  # '!siz'
        kAudioConverterErr_RequiresPacketDescriptionsError = 1937010788  # '!pkd'
        kAudioConverterErr_InputSampleRateOutOfRange = 1768845682  # '!isr'
        kAudioConverterErr_OutputSampleRateOutOfRange = 1869771634  # '!osr'

    # iOS-only error codes
    ctypedef enum:
        kAudioConverterErr_HardwareInUse = 1752392805  # 'hwiu'
        kAudioConverterErr_NoHardwarePermission = 1886547821  # 'perm'

    # Callback function types
    ctypedef OSStatus (*AudioConverterComplexInputDataProc)(AudioConverterRef inAudioConverter,
                                                           UInt32* ioNumberDataPackets,
                                                           AudioBufferList* ioData,
                                                           AudioStreamPacketDescription** outDataPacketDescription,
                                                           void* inUserData)

    ctypedef OSStatus (*AudioConverterInputDataProc)(AudioConverterRef inAudioConverter,
                                                    UInt32* ioDataSize,
                                                    void** outData,
                                                    void* inUserData)

    # AudioConverter functions
    cdef OSStatus AudioConverterNew(const AudioStreamBasicDescription* inSourceFormat,
                                   const AudioStreamBasicDescription* inDestinationFormat,
                                   AudioConverterRef* outAudioConverter)

    cdef OSStatus AudioConverterNewSpecific(const AudioStreamBasicDescription* inSourceFormat,
                                           const AudioStreamBasicDescription* inDestinationFormat,
                                           UInt32 inNumberClassDescriptions,
                                           const AudioClassDescription* inClassDescriptions,
                                           AudioConverterRef* outAudioConverter)

    cdef OSStatus AudioConverterDispose(AudioConverterRef inAudioConverter)
    cdef OSStatus AudioConverterReset(AudioConverterRef inAudioConverter)

    cdef OSStatus AudioConverterGetPropertyInfo(AudioConverterRef inAudioConverter,
                                               AudioConverterPropertyID inPropertyID,
                                               UInt32* outSize,
                                               Boolean* outWritable)

    cdef OSStatus AudioConverterGetProperty(AudioConverterRef inAudioConverter,
                                           AudioConverterPropertyID inPropertyID,
                                           UInt32* ioPropertyDataSize,
                                           void* outPropertyData)

    cdef OSStatus AudioConverterSetProperty(AudioConverterRef inAudioConverter,
                                           AudioConverterPropertyID inPropertyID,
                                           UInt32 inPropertyDataSize,
                                           const void* inPropertyData)

    cdef OSStatus AudioConverterConvertBuffer(AudioConverterRef inAudioConverter,
                                             UInt32 inInputDataSize,
                                             const void* inInputData,
                                             UInt32* ioOutputDataSize,
                                             void* outOutputData)

    cdef OSStatus AudioConverterFillComplexBuffer(AudioConverterRef inAudioConverter,
                                                 AudioConverterComplexInputDataProc inInputDataProc,
                                                 void* inInputDataProcUserData,
                                                 UInt32* ioOutputDataPacketSize,
                                                 AudioBufferList* outOutputData,
                                                 AudioStreamPacketDescription* outPacketDescription)

    cdef OSStatus AudioConverterConvertComplexBuffer(AudioConverterRef inAudioConverter,
                                                    UInt32 inNumberPCMFrames,
                                                    const AudioBufferList* inInputData,
                                                    AudioBufferList* outOutputData)

    # DEPRECATED: Use AudioConverterFillComplexBuffer instead (macOS 10.2+)
    # This function only works with non-interleaved audio and has been superseded
    # by AudioConverterFillComplexBuffer which handles all audio configurations.
    cdef OSStatus AudioConverterFillBuffer(AudioConverterRef inAudioConverter,
                                          AudioConverterInputDataProc inInputDataProc,
                                          void* inInputDataProcUserData,
                                          UInt32* ioOutputDataSize,
                                          void* outOutputData)


cdef extern from "AudioToolbox/ExtendedAudioFile.h":

    # ExtendedAudioFile types
    ctypedef struct OpaqueExtAudioFile
    ctypedef OpaqueExtAudioFile* ExtAudioFileRef
    ctypedef UInt32 ExtAudioFilePropertyID
    ctypedef SInt32 ExtAudioFilePacketTableInfoOverride

    # Packet table info override constants
    ctypedef enum:
        kExtAudioFilePacketTableInfoOverride_UseFileValue = -1
        kExtAudioFilePacketTableInfoOverride_UseFileValueIfValid = -2

    # ExtendedAudioFile property IDs
    ctypedef enum:
        kExtAudioFileProperty_FileDataFormat = 1717988724  # 'ffmt'
        kExtAudioFileProperty_FileChannelLayout = 1717791855  # 'fclo'
        kExtAudioFileProperty_ClientDataFormat = 1667657076  # 'cfmt'
        kExtAudioFileProperty_ClientChannelLayout = 1667460207  # 'cclo'
        kExtAudioFileProperty_CodecManufacturer = 1668112750  # 'cman'
        kExtAudioFileProperty_AudioConverter = 1633906294  # 'acnv'
        kExtAudioFileProperty_AudioFile = 1634101612  # 'afil'
        kExtAudioFileProperty_FileMaxPacketSize = 1718448243  # 'fmps'
        kExtAudioFileProperty_ClientMaxPacketSize = 1668116595  # 'cmps'
        kExtAudioFileProperty_FileLengthFrames = 593916525  # '#frm'
        kExtAudioFileProperty_ConverterConfig = 1633903462  # 'accf'
        kExtAudioFileProperty_IOBufferSizeBytes = 1768907379  # 'iobs'
        kExtAudioFileProperty_IOBuffer = 1768907366  # 'iobf'
        kExtAudioFileProperty_PacketTable = 2020635753  # 'xpti'

    # Error codes
    ctypedef enum:
        kExtAudioFileError_InvalidProperty = -66561
        kExtAudioFileError_InvalidPropertySize = -66562
        kExtAudioFileError_NonPCMClientFormat = -66563
        kExtAudioFileError_InvalidChannelMap = -66564
        kExtAudioFileError_InvalidOperationOrder = -66565
        kExtAudioFileError_InvalidDataFormat = -66566
        kExtAudioFileError_MaxPacketSizeUnknown = -66567
        kExtAudioFileError_InvalidSeek = -66568
        kExtAudioFileError_AsyncWriteTooLarge = -66569
        kExtAudioFileError_AsyncWriteBufferOverflow = -66570

    # iOS-only error codes
    ctypedef enum:
        kExtAudioFileError_CodecUnavailableInputConsumed = -66559
        kExtAudioFileError_CodecUnavailableInputNotConsumed = -66560

    # ExtendedAudioFile functions
    cdef OSStatus ExtAudioFileOpenURL(CFURLRef inURL,
                                     ExtAudioFileRef* outExtAudioFile)

    cdef OSStatus ExtAudioFileWrapAudioFileID(AudioFileID inFileID,
                                             Boolean inForWriting,
                                             ExtAudioFileRef* outExtAudioFile)

    cdef OSStatus ExtAudioFileCreateWithURL(CFURLRef inURL,
                                           AudioFileTypeID inFileType,
                                           const AudioStreamBasicDescription* inStreamDesc,
                                           const AudioChannelLayout* inChannelLayout,
                                           UInt32 inFlags,
                                           ExtAudioFileRef* outExtAudioFile)

    # I/O functions
    cdef OSStatus ExtAudioFileRead(ExtAudioFileRef inExtAudioFile,
                                  UInt32* ioNumberFrames,
                                  AudioBufferList* ioData)

    cdef OSStatus ExtAudioFileWrite(ExtAudioFileRef inExtAudioFile,
                                   UInt32 inNumberFrames,
                                   const AudioBufferList* ioData)

    cdef OSStatus ExtAudioFileWriteAsync(ExtAudioFileRef inExtAudioFile,
                                        UInt32 inNumberFrames,
                                        const AudioBufferList* ioData)

    cdef OSStatus ExtAudioFileSeek(ExtAudioFileRef inExtAudioFile,
                                  SInt64 inFrameOffset)

    cdef OSStatus ExtAudioFileTell(ExtAudioFileRef inExtAudioFile,
                                  SInt64* outFrameOffset)

    # Property functions
    cdef OSStatus ExtAudioFileGetPropertyInfo(ExtAudioFileRef inExtAudioFile,
                                             ExtAudioFilePropertyID inPropertyID,
                                             UInt32* outSize,
                                             Boolean* outWritable)

    cdef OSStatus ExtAudioFileGetProperty(ExtAudioFileRef inExtAudioFile,
                                         ExtAudioFilePropertyID inPropertyID,
                                         UInt32* ioPropertyDataSize,
                                         void* outPropertyData)

    cdef OSStatus ExtAudioFileSetProperty(ExtAudioFileRef inExtAudioFile,
                                         ExtAudioFilePropertyID inPropertyID,
                                         UInt32 inPropertyDataSize,
                                         const void* inPropertyData)

    cdef OSStatus ExtAudioFileDispose(ExtAudioFileRef inExtAudioFile)


cdef extern from "AudioToolbox/AudioFormat.h":

    # AudioFormat types
    ctypedef UInt32 AudioFormatPropertyID
    ctypedef UInt32 AudioPanningMode
    ctypedef UInt32 AudioBalanceFadeType

    # Panning mode constants
    ctypedef enum:
        kPanningMode_SoundField = 3
        kPanningMode_VectorBasedPanning = 4

    # Balance fade type constants
    ctypedef enum:
        kAudioBalanceFadeType_MaxUnityGain = 0
        kAudioBalanceFadeType_EqualPower = 1

    # AudioPanningInfo structure
    ctypedef struct AudioPanningInfo:
        AudioPanningMode mPanningMode
        UInt32 mCoordinateFlags
        Float32 mCoordinates[3]
        Float32 mGainScale
        const AudioChannelLayout* mOutputChannelMap

    # AudioBalanceFade structure
    ctypedef struct AudioBalanceFade:
        Float32 mLeftRightBalance
        Float32 mBackFrontFade
        AudioBalanceFadeType mType
        const AudioChannelLayout* mChannelLayout

    # AudioFormatInfo structure
    ctypedef struct AudioFormatInfo:
        AudioStreamBasicDescription mASBD
        const void* mMagicCookie
        UInt32 mMagicCookieSize

    # ExtendedAudioFormatInfo structure
    ctypedef struct ExtendedAudioFormatInfo:
        AudioStreamBasicDescription mASBD
        const void* mMagicCookie
        UInt32 mMagicCookieSize
        AudioClassDescription mClassDescription

    # AudioFormat property IDs
    ctypedef enum:
        kAudioFormatProperty_FormatInfo = 1718449257  # 'fmti'
        kAudioFormatProperty_FormatName = 1718509933  # 'fnam'
        kAudioFormatProperty_EncodeFormatIDs = 1633906534  # 'acof'
        kAudioFormatProperty_DecodeFormatIDs = 1633904998  # 'acif'
        kAudioFormatProperty_FormatList = 1718383476  # 'flst'
        kAudioFormatProperty_ASBDFromESDS = 1702064996  # 'essd'
        kAudioFormatProperty_ChannelLayoutFromESDS = 1702060908  # 'escl'
        kAudioFormatProperty_OutputFormatList = 1868983411  # 'ofls'
        kAudioFormatProperty_FirstPlayableFormatFromList = 1718642284  # 'fpfl'
        kAudioFormatProperty_FormatIsVBR = 1719034482  # 'fvbr'
        kAudioFormatProperty_FormatIsExternallyFramed = 1717925990  # 'fexf'
        kAudioFormatProperty_FormatEmploysDependentPackets = 1717855600  # 'fdep'
        kAudioFormatProperty_FormatIsEncrypted = 1668446576  # 'cryp'
        kAudioFormatProperty_Encoders = 1635149166  # 'aven'
        kAudioFormatProperty_Decoders = 1635148901  # 'avde'
        kAudioFormatProperty_AvailableEncodeBitRates = 1634034290  # 'aebr'
        kAudioFormatProperty_AvailableEncodeSampleRates = 1634038642  # 'aesr'
        kAudioFormatProperty_AvailableEncodeChannelLayoutTags = 1634034540  # 'aecl'
        kAudioFormatProperty_AvailableEncodeNumberChannels = 1635151459  # 'avnc'
        kAudioFormatProperty_AvailableDecodeNumberChannels = 1633971811  # 'adnc'
        kAudioFormatProperty_ASBDFromMPEGPacket = 1633971568  # 'admp'
        kAudioFormatProperty_BitmapForLayoutTag = 1651340391  # 'bmtg'
        kAudioFormatProperty_MatrixMixMap = 1835884912  # 'mmap'
        kAudioFormatProperty_ChannelMap = 1667788144  # 'chmp'
        kAudioFormatProperty_NumberOfChannelsForLayout = 1852008557  # 'nchm'
        kAudioFormatProperty_AreChannelLayoutsEquivalent = 1667786097  # 'cheq'
        kAudioFormatProperty_ChannelLayoutHash = 1667786849  # 'chha'
        kAudioFormatProperty_ValidateChannelLayout = 1986093932  # 'vacl'
        kAudioFormatProperty_ChannelLayoutForTag = 1668116588  # 'cmpl'
        kAudioFormatProperty_TagForChannelLayout = 1668116596  # 'cmpt'
        kAudioFormatProperty_ChannelLayoutName = 1819242093  # 'lonm'
        kAudioFormatProperty_ChannelLayoutSimpleName = 1819504237  # 'lsnm'
        kAudioFormatProperty_ChannelLayoutForBitmap = 1668116578  # 'cmpb'
        kAudioFormatProperty_ChannelName = 1668178285  # 'cnam'
        kAudioFormatProperty_ChannelShortName = 1668509293  # 'csnm'
        kAudioFormatProperty_TagsForNumberOfChannels = 1952540515  # 'tagc'
        kAudioFormatProperty_PanningMatrix = 1885433453  # 'panm'
        kAudioFormatProperty_BalanceFade = 1650551910  # 'balf'
        kAudioFormatProperty_ID3TagSize = 1768174451  # 'id3s'
        kAudioFormatProperty_ID3TagToDictionary = 1768174436  # 'id3d'

    # iOS-only properties
    ctypedef enum:
        kAudioFormatProperty_HardwareCodecCapabilities = 1752654691  # 'hwcc'

    # iOS-only codec types
    ctypedef enum:
        kAudioDecoderComponentType = 1633969507  # 'adec'
        kAudioEncoderComponentType = 1634037347  # 'aenc'

    # iOS-only codec manufacturers
    ctypedef enum:
        kAppleSoftwareAudioCodecManufacturer = 1634758764  # 'appl'
        kAppleHardwareAudioCodecManufacturer = 1634756727  # 'aphw'

    # Error codes
    ctypedef enum:
        kAudioFormatUnspecifiedError = 2003395684  # 'what'
        kAudioFormatUnsupportedPropertyError = 1886547824  # 'prop'
        kAudioFormatBadPropertySizeError = 561211770  # '!siz'
        kAudioFormatBadSpecifierSizeError = 561213539  # '!spc'
        kAudioFormatUnsupportedDataFormatError = 1718449215  # 'fmt?'
        kAudioFormatUnknownFormatError = 560360820  # '!fmt'

    # AudioFormat functions
    cdef OSStatus AudioFormatGetPropertyInfo(AudioFormatPropertyID inPropertyID,
                                            UInt32 inSpecifierSize,
                                            const void* inSpecifier,
                                            UInt32* outPropertyDataSize)

    cdef OSStatus AudioFormatGetProperty(AudioFormatPropertyID inPropertyID,
                                        UInt32 inSpecifierSize,
                                        const void* inSpecifier,
                                        UInt32* ioPropertyDataSize,
                                        void* outPropertyData)


cdef extern from "AudioToolbox/AudioServices.h":

    # AudioServices types
    ctypedef UInt32 SystemSoundID
    ctypedef UInt32 AudioServicesPropertyID

    # AudioServices callback function type
    ctypedef void (*AudioServicesSystemSoundCompletionProc)(
        SystemSoundID ssID,
        void* clientData)

    # AudioServices error codes
    ctypedef enum:
        kAudioServicesNoError = 0
        kAudioServicesUnsupportedPropertyError = 1886547839  # 'pty?'
        kAudioServicesBadPropertySizeError = 561211770  # '!siz'
        kAudioServicesBadSpecifierSizeError = 561213539  # '!spc'
        kAudioServicesSystemSoundUnspecifiedError = -1500
        kAudioServicesSystemSoundClientTimedOutError = -1501
        kAudioServicesSystemSoundExceededMaximumDurationError = -1502

    # AudioServices system sound constants
    ctypedef enum:
        kSystemSoundID_UserPreferredAlert = 0x00001000
        kSystemSoundID_FlashScreen = 0x00000FFE
        kSystemSoundID_Vibrate = 0x00000FFF
        kUserPreferredAlert = 0x00001000  # alias for kSystemSoundID_UserPreferredAlert

    # AudioServices property IDs
    ctypedef enum:
        kAudioServicesPropertyIsUISound = 1769239657  # 'isui'
        kAudioServicesPropertyCompletePlaybackIfAppDies = 1768842857  # 'ifdi'

    # AudioServices functions
    cdef OSStatus AudioServicesCreateSystemSoundID(
        CFURLRef inFileURL,
        SystemSoundID* outSystemSoundID)

    cdef OSStatus AudioServicesDisposeSystemSoundID(SystemSoundID inSystemSoundID)

    # Note: Block-based API - actual signature uses dispatch_block_t (void (^)(void))
    # Simplified to void* for Cython compatibility since blocks cannot be directly
    # represented in Cython. Use AudioServicesAddSystemSoundCompletion for callback support.
    cdef void AudioServicesPlayAlertSoundWithCompletion(
        SystemSoundID inSystemSoundID,
        void* inCompletionBlock)

    # Note: Block-based API - actual signature uses dispatch_block_t (void (^)(void))
    # Simplified to void* for Cython compatibility since blocks cannot be directly
    # represented in Cython. Use AudioServicesAddSystemSoundCompletion for callback support.
    cdef void AudioServicesPlaySystemSoundWithCompletion(
        SystemSoundID inSystemSoundID,
        void* inCompletionBlock)

    cdef OSStatus AudioServicesGetPropertyInfo(
        AudioServicesPropertyID inPropertyID,
        UInt32 inSpecifierSize,
        const void* inSpecifier,
        UInt32* outPropertyDataSize,
        Boolean* outWritable)

    cdef OSStatus AudioServicesGetProperty(
        AudioServicesPropertyID inPropertyID,
        UInt32 inSpecifierSize,
        const void* inSpecifier,
        UInt32* ioPropertyDataSize,
        void* outPropertyData)

    cdef OSStatus AudioServicesSetProperty(
        AudioServicesPropertyID inPropertyID,
        UInt32 inSpecifierSize,
        const void* inSpecifier,
        UInt32 inPropertyDataSize,
        const void* inPropertyData)

    # DEPRECATED: Use AudioServicesPlayAlertSoundWithCompletion instead (macOS 10.11+)
    # These functions are deprecated but still widely used for simple sound playback.
    # They work synchronously and don't provide completion notification.
    cdef void AudioServicesPlayAlertSound(SystemSoundID inSystemSoundID)

    # DEPRECATED: Use AudioServicesPlaySystemSoundWithCompletion instead (macOS 10.11+)
    cdef void AudioServicesPlaySystemSound(SystemSoundID inSystemSoundID)

    cdef OSStatus AudioServicesAddSystemSoundCompletion(
        SystemSoundID inSystemSoundID,
        CFRunLoopRef inRunLoop,
        CFStringRef inRunLoopMode,
        AudioServicesSystemSoundCompletionProc inCompletionRoutine,
        void* inClientData)

    cdef void AudioServicesRemoveSystemSoundCompletion(SystemSoundID inSystemSoundID)


# MusicDevice API declarations
cdef extern from "AudioToolbox/MusicDevice.h":

    # Type definitions
    ctypedef UInt32 MusicDeviceInstrumentID
    ctypedef UInt32 MusicDeviceGroupID
    ctypedef UInt32 NoteInstanceID
    ctypedef AudioComponentInstance MusicDeviceComponent

    # Forward declaration for MIDIEventList
    ctypedef struct MIDIEventList:
        pass

    # Standard note parameters structure
    ctypedef struct MusicDeviceStdNoteParams:
        UInt32 argCount
        Float32 mPitch
        Float32 mVelocity

    # Note parameter control value
    ctypedef struct NoteParamsControlValue:
        AudioUnitParameterID mID
        AudioUnitParameterValue mValue

    # Music device note parameters (variable length)
    ctypedef struct MusicDeviceNoteParams:
        UInt32 argCount
        Float32 mPitch
        Float32 mVelocity
        NoteParamsControlValue mControls[1]

    # Constants
    ctypedef enum:
        kMusicNoteEvent_UseGroupInstrument = 0xFFFFFFFF
        kMusicNoteEvent_Unused = 0xFFFFFFFF

    # Selector constants
    ctypedef enum:
        kMusicDeviceRange = 0x0100
        kMusicDeviceMIDIEventSelect = 0x0101
        kMusicDeviceSysExSelect = 0x0102
        kMusicDevicePrepareInstrumentSelect = 0x0103
        kMusicDeviceReleaseInstrumentSelect = 0x0104
        kMusicDeviceStartNoteSelect = 0x0105
        kMusicDeviceStopNoteSelect = 0x0106
        kMusicDeviceMIDIEventListSelect = 0x0107

    # Core functions
    cdef OSStatus MusicDeviceMIDIEvent(
        MusicDeviceComponent inUnit,
        UInt32 inStatus,
        UInt32 inData1,
        UInt32 inData2,
        UInt32 inOffsetSampleFrame)

    cdef OSStatus MusicDeviceSysEx(
        MusicDeviceComponent inUnit,
        const UInt8* inData,
        UInt32 inLength)

    cdef OSStatus MusicDeviceMIDIEventList(
        MusicDeviceComponent inUnit,
        UInt32 inOffsetSampleFrame,
        const MIDIEventList* evtList)

    cdef OSStatus MusicDeviceStartNote(
        MusicDeviceComponent inUnit,
        MusicDeviceInstrumentID inInstrument,
        MusicDeviceGroupID inGroupID,
        NoteInstanceID* outNoteInstanceID,
        UInt32 inOffsetSampleFrame,
        const MusicDeviceNoteParams* inParams)

    cdef OSStatus MusicDeviceStopNote(
        MusicDeviceComponent inUnit,
        MusicDeviceGroupID inGroupID,
        NoteInstanceID inNoteInstanceID,
        UInt32 inOffsetSampleFrame)

    # Function pointer types for fast dispatch
    ctypedef OSStatus (*MusicDeviceMIDIEventProc)(
        void* self,
        UInt32 inStatus,
        UInt32 inData1,
        UInt32 inData2,
        UInt32 inOffsetSampleFrame)

    ctypedef OSStatus (*MusicDeviceSysExProc)(
        void* self,
        const UInt8* inData,
        UInt32 inLength)

    ctypedef OSStatus (*MusicDeviceStartNoteProc)(
        void* self,
        MusicDeviceInstrumentID inInstrument,
        MusicDeviceGroupID inGroupID,
        NoteInstanceID* outNoteInstanceID,
        UInt32 inOffsetSampleFrame,
        const MusicDeviceNoteParams* inParams)

    ctypedef OSStatus (*MusicDeviceStopNoteProc)(
        void* self,
        MusicDeviceGroupID inGroupID,
        NoteInstanceID inNoteInstanceID,
        UInt32 inOffsetSampleFrame)


# MusicPlayer API declarations
cdef extern from "AudioToolbox/MusicPlayer.h":

    # Opaque types
    ctypedef struct OpaqueMusicPlayer:
        pass
    ctypedef OpaqueMusicPlayer* MusicPlayer

    ctypedef struct OpaqueMusicSequence:
        pass
    ctypedef OpaqueMusicSequence* MusicSequence

    ctypedef struct OpaqueMusicTrack:
        pass
    ctypedef OpaqueMusicTrack* MusicTrack

    ctypedef struct OpaqueMusicEventIterator:
        pass
    ctypedef OpaqueMusicEventIterator* MusicEventIterator

    # Basic types
    ctypedef Float64 MusicTimeStamp
    ctypedef UInt32 MusicEventType

    # Forward declarations for missing types
    ctypedef void* AUGraph
    ctypedef UInt32 AUNode

    # Event types constants
    ctypedef enum:
        kMusicEventType_NULL = 0
        kMusicEventType_ExtendedNote = 1
        kMusicEventType_ExtendedTempo = 3
        kMusicEventType_User = 4
        kMusicEventType_Meta = 5
        kMusicEventType_MIDINoteMessage = 6
        kMusicEventType_MIDIChannelMessage = 7
        kMusicEventType_MIDIRawData = 8
        kMusicEventType_Parameter = 9
        kMusicEventType_AUPreset = 10

    # Sequence types
    ctypedef enum:
        kMusicSequenceType_Beats = 1651077476  # 'beat'
        kMusicSequenceType_Seconds = 1936941667  # 'secs'
        kMusicSequenceType_Samples = 1935832176  # 'samp'

    # File types
    ctypedef enum:
        kMusicSequenceFile_AnyType = 0
        kMusicSequenceFile_MIDIType = 1835623529  # 'midi'
        kMusicSequenceFile_iMelodyType = 1768776044  # 'imel'

    # Load and file flags
    ctypedef enum:
        kMusicSequenceLoadSMF_PreserveTracks = 0
        kMusicSequenceLoadSMF_ChannelsToTracks = 1
        kMusicSequenceFileFlags_Default = 0
        kMusicSequenceFileFlags_EraseFile = 1

    # Track property constants
    ctypedef enum:
        kSequenceTrackProperty_LoopInfo = 0
        kSequenceTrackProperty_OffsetTime = 1
        kSequenceTrackProperty_MuteStatus = 2
        kSequenceTrackProperty_SoloStatus = 3
        kSequenceTrackProperty_AutomatedParameters = 4
        kSequenceTrackProperty_TrackLength = 5
        kSequenceTrackProperty_TimeResolution = 6

    # Error constants
    ctypedef enum:
        kAudioToolboxErr_InvalidSequenceType = -10846
        kAudioToolboxErr_TrackIndexError = -10859
        kAudioToolboxErr_TrackNotFound = -10858
        kAudioToolboxErr_EndOfTrack = -10857
        kAudioToolboxErr_StartOfTrack = -10856
        kAudioToolboxErr_IllegalTrackDestination = -10855
        kAudioToolboxErr_NoSequence = -10854
        kAudioToolboxErr_InvalidEventType = -10853
        kAudioToolboxErr_InvalidPlayerState = -10852
        kAudioToolboxErr_CannotDoInCurrentContext = -10863
        kAudioToolboxError_NoTrackDestination = -66720

    # Event structures
    ctypedef struct MIDINoteMessage:
        UInt8 channel
        UInt8 note
        UInt8 velocity
        UInt8 releaseVelocity
        Float32 duration

    ctypedef struct MIDIChannelMessage:
        UInt8 status
        UInt8 data1
        UInt8 data2
        UInt8 reserved

    ctypedef struct MIDIRawData:
        UInt32 length
        UInt8 data[1]

    ctypedef struct MIDIMetaEvent:
        UInt8 metaEventType
        UInt8 unused1
        UInt8 unused2
        UInt8 unused3
        UInt32 dataLength
        UInt8 data[1]

    ctypedef struct MusicEventUserData:
        UInt32 length
        UInt8 data[1]

    ctypedef struct ExtendedNoteOnEvent:
        MusicDeviceInstrumentID instrumentID
        MusicDeviceGroupID groupID
        Float32 duration
        MusicDeviceNoteParams extendedParams

    ctypedef struct ParameterEvent:
        AudioUnitParameterID parameterID
        AudioUnitScope scope
        AudioUnitElement element
        AudioUnitParameterValue value

    ctypedef struct ExtendedTempoEvent:
        Float64 bpm

    ctypedef struct AUPresetEvent:
        AudioUnitScope scope
        AudioUnitElement element
        void* preset  # CFPropertyListRef

    ctypedef struct MusicTrackLoopInfo:
        MusicTimeStamp loopDuration
        SInt32 numberOfLoops

    ctypedef struct CABarBeatTime:
        SInt32 bar
        UInt16 beat
        UInt16 subbeat
        UInt16 subbeatDivisor
        UInt16 reserved

    # Callback types
    ctypedef void (*MusicSequenceUserCallback)(
        void* inClientData,
        MusicSequence inSequence,
        MusicTrack inTrack,
        MusicTimeStamp inEventTime,
        const MusicEventUserData* inEventData,
        MusicTimeStamp inStartSliceBeat,
        MusicTimeStamp inEndSliceBeat)

    # MusicPlayer functions
    cdef OSStatus NewMusicPlayer(MusicPlayer* outPlayer)
    cdef OSStatus DisposeMusicPlayer(MusicPlayer inPlayer)
    cdef OSStatus MusicPlayerSetSequence(MusicPlayer inPlayer, MusicSequence inSequence)
    cdef OSStatus MusicPlayerGetSequence(MusicPlayer inPlayer, MusicSequence* outSequence)
    cdef OSStatus MusicPlayerSetTime(MusicPlayer inPlayer, MusicTimeStamp inTime)
    cdef OSStatus MusicPlayerGetTime(MusicPlayer inPlayer, MusicTimeStamp* outTime)
    cdef OSStatus MusicPlayerGetHostTimeForBeats(MusicPlayer inPlayer, MusicTimeStamp inBeats, UInt64* outHostTime)
    cdef OSStatus MusicPlayerGetBeatsForHostTime(MusicPlayer inPlayer, UInt64 inHostTime, MusicTimeStamp* outBeats)
    cdef OSStatus MusicPlayerPreroll(MusicPlayer inPlayer)
    cdef OSStatus MusicPlayerStart(MusicPlayer inPlayer)
    cdef OSStatus MusicPlayerStop(MusicPlayer inPlayer)
    cdef OSStatus MusicPlayerIsPlaying(MusicPlayer inPlayer, Boolean* outIsPlaying)
    cdef OSStatus MusicPlayerSetPlayRateScalar(MusicPlayer inPlayer, Float64 inScaleRate)
    cdef OSStatus MusicPlayerGetPlayRateScalar(MusicPlayer inPlayer, Float64* outScaleRate)

    # MusicSequence functions
    cdef OSStatus NewMusicSequence(MusicSequence* outSequence)
    cdef OSStatus DisposeMusicSequence(MusicSequence inSequence)
    cdef OSStatus MusicSequenceNewTrack(MusicSequence inSequence, MusicTrack* outTrack)
    cdef OSStatus MusicSequenceDisposeTrack(MusicSequence inSequence, MusicTrack inTrack)
    cdef OSStatus MusicSequenceGetTrackCount(MusicSequence inSequence, UInt32* outNumberOfTracks)
    cdef OSStatus MusicSequenceGetIndTrack(MusicSequence inSequence, UInt32 inTrackIndex, MusicTrack* outTrack)
    cdef OSStatus MusicSequenceGetTrackIndex(MusicSequence inSequence, MusicTrack inTrack, UInt32* outTrackIndex)
    cdef OSStatus MusicSequenceGetTempoTrack(MusicSequence inSequence, MusicTrack* outTrack)
    cdef OSStatus MusicSequenceSetAUGraph(MusicSequence inSequence, AUGraph inGraph)
    cdef OSStatus MusicSequenceGetAUGraph(MusicSequence inSequence, AUGraph* outGraph)
    cdef OSStatus MusicSequenceSetSequenceType(MusicSequence inSequence, UInt32 inType)
    cdef OSStatus MusicSequenceGetSequenceType(MusicSequence inSequence, UInt32* outType)
    cdef OSStatus MusicSequenceFileLoad(MusicSequence inSequence, CFURLRef inFileRef, UInt32 inFileTypeHint, UInt32 inFlags)
    # cdef OSStatus MusicSequenceFileLoadData(MusicSequence inSequence, CFDataRef inData, UInt32 inFileTypeHint, UInt32 inFlags)
    # cdef OSStatus MusicSequenceFileCreate(MusicSequence inSequence, CFURLRef inFileRef, UInt32 inFileType, UInt32 inFlags, SInt16 inResolution)
    # cdef OSStatus MusicSequenceFileCreateData(MusicSequence inSequence, UInt32 inFileType, UInt32 inFlags, SInt16 inResolution, CFDataRef* outData)
    cdef OSStatus MusicSequenceReverse(MusicSequence inSequence)
    cdef OSStatus MusicSequenceGetSecondsForBeats(MusicSequence inSequence, MusicTimeStamp inBeats, Float64* outSeconds)
    cdef OSStatus MusicSequenceGetBeatsForSeconds(MusicSequence inSequence, Float64 inSeconds, MusicTimeStamp* outBeats)
    cdef OSStatus MusicSequenceSetUserCallback(MusicSequence inSequence, MusicSequenceUserCallback inCallback, void* inClientData)
    cdef OSStatus MusicSequenceBeatsToBarBeatTime(MusicSequence inSequence, MusicTimeStamp inBeats, UInt32 inSubbeatDivisor, CABarBeatTime* outBarBeatTime)
    cdef OSStatus MusicSequenceBarBeatTimeToBeats(MusicSequence inSequence, const CABarBeatTime* inBarBeatTime, MusicTimeStamp* outBeats)
    # cdef CFDictionaryRef MusicSequenceGetInfoDictionary(MusicSequence inSequence)

    # MusicTrack functions
    cdef OSStatus MusicTrackGetSequence(MusicTrack inTrack, MusicSequence* outSequence)
    cdef OSStatus MusicTrackSetDestNode(MusicTrack inTrack, AUNode inNode)
    cdef OSStatus MusicTrackGetDestNode(MusicTrack inTrack, AUNode* outNode)
    cdef OSStatus MusicTrackSetProperty(MusicTrack inTrack, UInt32 inPropertyID, void* inData, UInt32 inLength)
    cdef OSStatus MusicTrackGetProperty(MusicTrack inTrack, UInt32 inPropertyID, void* outData, UInt32* ioLength)
    cdef OSStatus MusicTrackMoveEvents(MusicTrack inTrack, MusicTimeStamp inStartTime, MusicTimeStamp inEndTime, MusicTimeStamp inMoveTime)
    cdef OSStatus MusicTrackClear(MusicTrack inTrack, MusicTimeStamp inStartTime, MusicTimeStamp inEndTime)
    cdef OSStatus MusicTrackCut(MusicTrack inTrack, MusicTimeStamp inStartTime, MusicTimeStamp inEndTime)
    cdef OSStatus MusicTrackCopyInsert(MusicTrack inSourceTrack, MusicTimeStamp inSourceStartTime, MusicTimeStamp inSourceEndTime, MusicTrack inDestTrack, MusicTimeStamp inDestInsertTime)
    cdef OSStatus MusicTrackMerge(MusicTrack inSourceTrack, MusicTimeStamp inSourceStartTime, MusicTimeStamp inSourceEndTime, MusicTrack inDestTrack, MusicTimeStamp inDestInsertTime)

    # MusicTrack event creation functions
    cdef OSStatus MusicTrackNewMIDINoteEvent(MusicTrack inTrack, MusicTimeStamp inTimeStamp, const MIDINoteMessage* inMessage)
    cdef OSStatus MusicTrackNewMIDIChannelEvent(MusicTrack inTrack, MusicTimeStamp inTimeStamp, const MIDIChannelMessage* inMessage)
    cdef OSStatus MusicTrackNewMIDIRawDataEvent(MusicTrack inTrack, MusicTimeStamp inTimeStamp, const MIDIRawData* inRawData)
    cdef OSStatus MusicTrackNewExtendedNoteEvent(MusicTrack inTrack, MusicTimeStamp inTimeStamp, const ExtendedNoteOnEvent* inInfo)
    cdef OSStatus MusicTrackNewParameterEvent(MusicTrack inTrack, MusicTimeStamp inTimeStamp, const ParameterEvent* inInfo)
    cdef OSStatus MusicTrackNewExtendedTempoEvent(MusicTrack inTrack, MusicTimeStamp inTimeStamp, Float64 inBPM)
    cdef OSStatus MusicTrackNewMetaEvent(MusicTrack inTrack, MusicTimeStamp inTimeStamp, const MIDIMetaEvent* inMetaEvent)
    cdef OSStatus MusicTrackNewUserEvent(MusicTrack inTrack, MusicTimeStamp inTimeStamp, const MusicEventUserData* inUserData)
    cdef OSStatus MusicTrackNewAUPresetEvent(MusicTrack inTrack, MusicTimeStamp inTimeStamp, const AUPresetEvent* inPresetEvent)

    # MusicEventIterator functions
    cdef OSStatus NewMusicEventIterator(MusicTrack inTrack, MusicEventIterator* outIterator)
    cdef OSStatus DisposeMusicEventIterator(MusicEventIterator inIterator)
    cdef OSStatus MusicEventIteratorSeek(MusicEventIterator inIterator, MusicTimeStamp inTimeStamp)
    cdef OSStatus MusicEventIteratorNextEvent(MusicEventIterator inIterator)
    cdef OSStatus MusicEventIteratorPreviousEvent(MusicEventIterator inIterator)
    cdef OSStatus MusicEventIteratorGetEventInfo(MusicEventIterator inIterator, MusicTimeStamp* outTimeStamp, MusicEventType* outEventType, const void** outEventData, UInt32* outEventDataSize)
    cdef OSStatus MusicEventIteratorSetEventInfo(MusicEventIterator inIterator, MusicEventType inEventType, const void* inEventData)
    cdef OSStatus MusicEventIteratorSetEventTime(MusicEventIterator inIterator, MusicTimeStamp inTimeStamp)
    cdef OSStatus MusicEventIteratorDeleteEvent(MusicEventIterator inIterator)
    cdef OSStatus MusicEventIteratorHasPreviousEvent(MusicEventIterator inIterator, Boolean* outHasPrevEvent)
    cdef OSStatus MusicEventIteratorHasNextEvent(MusicEventIterator inIterator, Boolean* outHasNextEvent)
    cdef OSStatus MusicEventIteratorHasCurrentEvent(MusicEventIterator inIterator, Boolean* outHasCurEvent)


# -----------------------------------------------------------------------------
# AUGraph API
# -----------------------------------------------------------------------------

cdef extern from "AudioToolbox/AUGraph.h":

    # Type definitions
    ctypedef struct OpaqueAUGraph
    ctypedef OpaqueAUGraph* AUGraph
    ctypedef SInt32 AUNode

    # Error codes
    ctypedef enum:
        kAUGraphErr_NodeNotFound = -10860
        kAUGraphErr_InvalidConnection = -10861
        kAUGraphErr_OutputNodeErr = -10862
        kAUGraphErr_CannotDoInCurrentContext = -10863
        kAUGraphErr_InvalidAudioUnit = -10864

    # Node interaction types
    ctypedef enum:
        kAUNodeInteraction_Connection = 1
        kAUNodeInteraction_InputCallback = 2

    # Structures
    ctypedef struct AudioUnitNodeConnection:
        AUNode sourceNode
        UInt32 sourceOutputNumber
        AUNode destNode
        UInt32 destInputNumber

    ctypedef AudioUnitNodeConnection AUNodeConnection

    ctypedef struct AUNodeRenderCallback:
        AUNode destNode
        UInt32 destInputNumber
        AURenderCallbackStruct cback

    ctypedef struct AUNodeInteraction:
        UInt32 nodeInteractionType
        # Union not fully declared - we'll handle at Python level

    # Graph lifecycle
    cdef OSStatus NewAUGraph(AUGraph* outGraph)
    cdef OSStatus DisposeAUGraph(AUGraph inGraph)
    cdef OSStatus AUGraphOpen(AUGraph inGraph)
    cdef OSStatus AUGraphClose(AUGraph inGraph)
    cdef OSStatus AUGraphInitialize(AUGraph inGraph)
    cdef OSStatus AUGraphUninitialize(AUGraph inGraph)

    # Graph control
    cdef OSStatus AUGraphStart(AUGraph inGraph)
    cdef OSStatus AUGraphStop(AUGraph inGraph)
    cdef OSStatus AUGraphIsOpen(AUGraph inGraph, Boolean* outIsOpen)
    cdef OSStatus AUGraphIsInitialized(AUGraph inGraph, Boolean* outIsInitialized)
    cdef OSStatus AUGraphIsRunning(AUGraph inGraph, Boolean* outIsRunning)

    # Node management
    cdef OSStatus AUGraphAddNode(AUGraph inGraph, const AudioComponentDescription* inDescription, AUNode* outNode)
    cdef OSStatus AUGraphRemoveNode(AUGraph inGraph, AUNode inNode)
    cdef OSStatus AUGraphGetNodeCount(AUGraph inGraph, UInt32* outNumberOfNodes)
    cdef OSStatus AUGraphGetIndNode(AUGraph inGraph, UInt32 inIndex, AUNode* outNode)
    cdef OSStatus AUGraphNodeInfo(AUGraph inGraph, AUNode inNode, AudioComponentDescription* outDescription, AudioUnit* outAudioUnit)

    # Connections
    cdef OSStatus AUGraphConnectNodeInput(AUGraph inGraph, AUNode inSourceNode, UInt32 inSourceOutputNumber, AUNode inDestNode, UInt32 inDestInputNumber)
    cdef OSStatus AUGraphDisconnectNodeInput(AUGraph inGraph, AUNode inDestNode, UInt32 inDestInputNumber)
    cdef OSStatus AUGraphClearConnections(AUGraph inGraph)

    # Graph updates
    cdef OSStatus AUGraphUpdate(AUGraph inGraph, Boolean* outIsUpdated)

    # Utilities
    cdef OSStatus AUGraphGetCPULoad(AUGraph inGraph, Float32* outAverageCPULoad)
    cdef OSStatus AUGraphGetMaxCPULoad(AUGraph inGraph, Float32* outMaxLoad)


# ============================================================================
# CoreAudioClock - Audio/MIDI Synchronization and Timing Services
# ============================================================================

cdef extern from "AudioToolbox/CoreAudioClock.h":
    # Opaque clock reference
    ctypedef struct OpaqueCAClock:
        pass
    ctypedef OpaqueCAClock* CAClockRef

    # Time and tempo types
    ctypedef Float64 CAClockBeats       # MIDI quarter notes
    ctypedef Float64 CAClockTempo       # beats per minute
    ctypedef Float64 CAClockSamples     # audio samples
    ctypedef Float64 CAClockSeconds     # seconds

    # Property IDs
    ctypedef enum CAClockPropertyID:
        kCAClockProperty_InternalTimebase       = 0x696E7462  # 'intb'
        kCAClockProperty_TimebaseSource         = 0x69746273  # 'itbs'
        kCAClockProperty_SyncMode               = 0x73796E6D  # 'synm'
        kCAClockProperty_SyncSource             = 0x73796E73  # 'syns'
        kCAClockProperty_SMPTEFormat            = 0x736D7066  # 'smpf'
        kCAClockProperty_SMPTEOffset            = 0x736D706F  # 'smpo'
        kCAClockProperty_MIDIClockDestinations  = 0x6D626364  # 'mbcd'
        kCAClockProperty_MTCDestinations        = 0x6D746364  # 'mtcd'
        kCAClockProperty_MTCFreewheelTime       = 0x6D746677  # 'mtfw'
        kCAClockProperty_TempoMap               = 0x746D706F  # 'tmpo'
        kCAClockProperty_MeterTrack             = 0x6D657472  # 'metr'
        kCAClockProperty_Name                   = 0x6E616D65  # 'name'
        kCAClockProperty_SendMIDISPP            = 0x6D737070  # 'mspp'

    # Timebase types
    ctypedef enum CAClockTimebase:
        kCAClockTimebase_HostTime           = 0x686F7374  # 'host'
        kCAClockTimebase_AudioDevice        = 0x61756469  # 'audi'
        kCAClockTimebase_AudioOutputUnit    = 0x61756F75  # 'auou'

    # Sync modes
    ctypedef enum CAClockSyncMode:
        kCAClockSyncMode_Internal               = 0x696E7472  # 'intr'
        kCAClockSyncMode_MIDIClockTransport     = 0x6D636C6B  # 'mclk'
        kCAClockSyncMode_MTCTransport           = 0x6D6D7463  # 'mmtc'

    # SMPTE format (already defined in coreaudio.pxd as SMPTETimeType)
    ctypedef SMPTETimeType CAClockSMPTEFormat

    # Clock messages
    ctypedef enum CAClockMessage:
        kCAClockMessage_StartTimeSet        = 0x7374696D  # 'stim'
        kCAClockMessage_Started             = 0x73747274  # 'strt'
        kCAClockMessage_Stopped             = 0x73746F70  # 'stop'
        kCAClockMessage_Armed               = 0x61726D64  # 'armd'
        kCAClockMessage_Disarmed            = 0x6461726D  # 'darm'
        kCAClockMessage_PropertyChanged     = 0x70636867  # 'pchg'
        kCAClockMessage_WrongSMPTEFormat    = 0x3F736D70  # '?smp'

    # Time formats
    ctypedef enum CAClockTimeFormat:
        kCAClockTimeFormat_HostTime         = 0x686F7374  # 'host'
        kCAClockTimeFormat_Samples          = 0x73616D70  # 'samp'
        kCAClockTimeFormat_Beats            = 0x62656174  # 'beat'
        kCAClockTimeFormat_Seconds          = 0x73656373  # 'secs'
        kCAClockTimeFormat_SMPTESeconds     = 0x736D7073  # 'smps'
        kCAClockTimeFormat_SMPTETime        = 0x736D7074  # 'smpt'
        kCAClockTimeFormat_AbsoluteSeconds  = 0x61736563  # 'asec'

    # Clock time structure
    ctypedef union CAClockTimeUnion:
        UInt64          hostTime
        CAClockSamples  samples
        CAClockBeats    beats
        CAClockSeconds  seconds
        SMPTETime       smpte

    ctypedef struct CAClockTime:
        CAClockTimeFormat   format
        UInt32              reserved
        CAClockTimeUnion    time

    # Tempo map entry
    ctypedef struct CATempoMapEntry:
        CAClockBeats    beats
        CAClockTempo    tempoBPM

    # Meter track entry
    ctypedef struct CAMeterTrackEntry:
        CAClockBeats    beats
        UInt16          meterNumer
        UInt16          meterDenom

    # Listener callback
    ctypedef void (*CAClockListenerProc)(void* userData, CAClockMessage message, const void* param)

    # Clock lifecycle functions
    cdef OSStatus CAClockNew(UInt32 inReservedFlags, CAClockRef* outCAClock)
    cdef OSStatus CAClockDispose(CAClockRef inCAClock)

    # Property functions
    cdef OSStatus CAClockGetPropertyInfo(CAClockRef inCAClock, CAClockPropertyID inPropertyID, UInt32* outSize, Boolean* outWritable)
    cdef OSStatus CAClockGetProperty(CAClockRef inCAClock, CAClockPropertyID inPropertyID, UInt32* ioSize, void* outData)
    cdef OSStatus CAClockSetProperty(CAClockRef inCAClock, CAClockPropertyID inPropertyID, UInt32 inSize, const void* inData)

    # Listener functions
    cdef OSStatus CAClockAddListener(CAClockRef inCAClock, CAClockListenerProc inListenerProc, void* inUserData)
    cdef OSStatus CAClockRemoveListener(CAClockRef inCAClock, CAClockListenerProc inListenerProc, void* inUserData)

    # Time control functions
    cdef OSStatus CAClockSetCurrentTime(CAClockRef inCAClock, const CAClockTime* inTime)
    cdef OSStatus CAClockGetCurrentTime(CAClockRef inCAClock, CAClockTimeFormat inFormat, CAClockTime* outTime)
    cdef OSStatus CAClockGetStartTime(CAClockRef inCAClock, CAClockTimeFormat inFormat, CAClockTime* outTime)

    # Playback control
    cdef OSStatus CAClockSetPlayRate(CAClockRef inCAClock, Float64 inRate)
    cdef OSStatus CAClockGetPlayRate(CAClockRef inCAClock, Float64* outRate)
    cdef OSStatus CAClockArm(CAClockRef inCAClock)
    cdef OSStatus CAClockDisarm(CAClockRef inCAClock)
    cdef OSStatus CAClockStart(CAClockRef inCAClock)
    cdef OSStatus CAClockStop(CAClockRef inCAClock)

    # Note: Some functions like CAClockSetCurrentTempo, CAClockSecondsToBeats may not be
    # available in all macOS versions or may have different signatures

    # Error codes
    cdef enum:
        kCAClock_UnknownPropertyError           = -66816
        kCAClock_InvalidPropertySizeError       = -66815
        kCAClock_InvalidTimeFormatError         = -66814
        kCAClock_InvalidSyncModeError           = -66813
        kCAClock_InvalidSyncSourceError         = -66812
        kCAClock_InvalidTimebaseError           = -66811
        kCAClock_InvalidTimebaseSourceError     = -66810
        kCAClock_InvalidSMPTEFormatError        = -66809
        kCAClock_InvalidSMPTEOffsetError        = -66808
        kCAClock_InvalidUnitError               = -66807
        kCAClock_InvalidPlayRateError           = -66806
        kCAClock_CannotSetTimeError             = -66805
