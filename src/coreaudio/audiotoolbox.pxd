# audiotoolbox.pxd
# AudioToolbox framework declarations for cycoreaudio

from .corefoundation cimport *
from .coreaudiotypes cimport *
from .coreaudio cimport *

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
    ctypedef void (*AudioQueueInputCallback)(void* inUserData, AudioQueueRef inAQ, AudioQueueBufferRef inBuffer, const AudioTimeStamp* inStartTime, UInt32 inNumberPacketDescriptions)

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
    cdef OSStatus AudioQueueEnqueueBuffer(AudioQueueRef inAQ, AudioQueueBufferRef inBuffer, UInt32 inNumPacketDescs, const void* inPacketDescs)
    cdef OSStatus AudioQueueEnqueueBufferWithParameters(AudioQueueRef inAQ, AudioQueueBufferRef inBuffer, UInt32 inNumPacketDescs, const void* inPacketDescs, UInt32 inTrimFramesAtStart, UInt32 inTrimFramesAtEnd, UInt32 inNumParamValues, const AudioQueueParameterEvent* inParamValues, const AudioTimeStamp* inStartTime, AudioTimeStamp* outActualStartTime)
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

# -----------------------------------------------------------------------------

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

    # Deprecated functions (macOS only)
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