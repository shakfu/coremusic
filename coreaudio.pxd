cdef extern from *:
    """
    #define FOURCC_ARGS(x)  (char)((x & 0xff000000) >> 24), \
        (char)((x & 0xff0000) >> 16),                   \
        (char)((x & 0xff00) >> 8), (char)((x) & 0xff)
    """
    ctypedef unsigned long uint64_t
    ctypedef long int64_t
    cdef char[] FOURCC_ARGS(SInt32)

cdef extern from "CoreFoundation/CFBase.h":
    ctypedef float          Float32
    ctypedef double         Float64
    ctypedef unsigned char  Boolean
    ctypedef unsigned char  UInt8
    ctypedef signed char    SInt8
    ctypedef unsigned short UInt16
    ctypedef signed short   SInt16
    ctypedef unsigned int   UInt32
    ctypedef signed int     SInt32
    ctypedef uint64_t       UInt64
    ctypedef int64_t        SInt64
    ctypedef SInt32         OSStatus

    ctypedef CFDictionaryRef

cdef extern from "CoreAudio/CoreAudioTypes.h":

    ctypedef enum SMPTETimeType:
        kSMPTETimeType24        = 0
        kSMPTETimeType25        = 1
        kSMPTETimeType30Drop    = 2
        kSMPTETimeType30        = 3
        kSMPTETimeType2997      = 4
        kSMPTETimeType2997Drop  = 5
        kSMPTETimeType60        = 6
        kSMPTETimeType5994      = 7
        kSMPTETimeType60Drop    = 8
        kSMPTETimeType5994Drop  = 9
        kSMPTETimeType50        = 10
        kSMPTETimeType2398      = 11

    ctypedef enum SMPTETimeFlags:
        kSMPTETimeUnknown   = 0
        kSMPTETimeValid     = 1
        kSMPTETimeRunning   = 2

    ctypedef struct SMPTETime:
        SInt16          mSubframes
        SInt16          mSubframeDivisor
        UInt32          mCounter
        SMPTETimeType   mType
        SMPTETimeFlags  mFlags
        SInt16          mHours
        SInt16          mMinutes
        SInt16          mSeconds
        SInt16          mFrames

    ctypedef enum AudioTimeStampFlags:
        kAudioTimeStampNothingValid         = 0
        kAudioTimeStampSampleTimeValid      = 1
        kAudioTimeStampHostTimeValid        = 2
        kAudioTimeStampRateScalarValid      = 4
        kAudioTimeStampWordClockTimeValid   = 8
        kAudioTimeStampSMPTETimeValid       = 16
        kAudioTimeStampSampleHostTimeValid  = 3

    ctypedef struct AudioTimeStamp:
        Float64             mSampleTime
        UInt64              mHostTime
        Float64             mRateScalar
        UInt64              mWordClockTime
        SMPTETime           mSMPTETime
        AudioTimeStampFlags mFlags
        UInt32              mReserved


# see headers/corefoundation/CFAvaiability.h for CF_ENUM explanation


# cdef extern from *:
#     cdef enum Dummy:
#         Plug = 1886156135


cdef extern from "CoreAudio/AudioHardwareBase.h":
    ctypedef UInt32 AudioObjectID
    ctypedef UInt32 AudioClassID
    ctypedef UInt32 AudioObjectPropertySelector
    ctypedef UInt32 AudioObjectPropertyScope
    ctypedef UInt32 AudioObjectPropertyElement

    ctypedef struct AudioObjectPropertyAddress:
        AudioObjectPropertySelector mSelector
        AudioObjectPropertyScope    mScope
        AudioObjectPropertyElement  mElement


cdef extern from "CoreAudio/AudioHardware.h":

    ctypedef enum:
        kAudioObjectSystemObject = 1

    ctypedef enum:
        kAudioObjectPropertyCreator             = 1869638759
        kAudioObjectPropertyListenerAdded       = 1818850145
        kAudioObjectPropertyListenerRemoved     = 1818850162

    ctypedef OSStatus (*AudioObjectPropertyListenerProc)(AudioObjectID inObjectID, UInt32 inNumberAddresses, const AudioObjectPropertyAddress* inAddresses, void* inClientData)
    ctypedef void (*AudioObjectPropertyListenerBlock)( UInt32 inNumberAddresses, const AudioObjectPropertyAddress* inAddresses)

    cdef void AudioObjectShow(AudioObjectID inObjectID)
    cdef Boolean AudioObjectHasProperty(AudioObjectID inObjectID, const AudioObjectPropertyAddress* inAddress) 
    cdef OSStatus AudioObjectIsPropertySettable(AudioObjectID inObjectID, const AudioObjectPropertyAddress* inAddress, Boolean* outIsSettable)   
    cdef OSStatus AudioObjectGetPropertyDataSize(AudioObjectID inObjectID, const AudioObjectPropertyAddress* inAddress, UInt32 inQualifierDataSize, const void* inQualifierData, UInt32* outDataSize)
    cdef OSStatus AudioObjectGetPropertyData(AudioObjectID inObjectID, const AudioObjectPropertyAddress* inAddress, UInt32 inQualifierDataSize, const void* inQualifierData, UInt32* ioDataSize, void* outData)
    cdef OSStatus AudioObjectSetPropertyData( AudioObjectID inObjectID, const AudioObjectPropertyAddress* inAddress, UInt32 inQualifierDataSize, const void* inQualifierData, UInt32 inDataSize, const void* inData)
    cdef OSStatus AudioObjectAddPropertyListener( AudioObjectID inObjectID, const AudioObjectPropertyAddress* inAddress, AudioObjectPropertyListenerProc inListener, void* inClientData)
    cdef OSStatus AudioObjectRemovePropertyListener( AudioObjectID inObjectID, const AudioObjectPropertyAddress* inAddress, AudioObjectPropertyListenerProc inListener, void*  inClientData)
    # cdef OSStatus AudioObjectAddPropertyListenerBlock( AudioObjectID inObjectID, const AudioObjectPropertyAddress* inAddress, dispatch_queue_t inDispatchQueue, AudioObjectPropertyListenerBlock inListener)
    # cdef OSStatus AudioObjectRemovePropertyListenerBlock( AudioObjectID inObjectID, const AudioObjectPropertyAddress* inAddress, dispatch_queue_t inDispatchQueue, AudioObjectPropertyListenerBlock inListener)

    ctypedef enum:
        kAudioSystemObjectClassID = 1634957683

    ctypedef UInt32 AudioHardwarePowerHint

    ctypedef enum:
        kAudioHardwarePowerHintNone = 0
        kAudioHardwarePowerHintFavorSavingPower = 1

    cdef OSStatus AudioHardwareUnload()

    cdef OSStatus AudioHardwareCreateAggregateDevice( CFDictionaryRef inDescription, AudioObjectID* outDeviceID)
    cdef OSStatus AudioHardwareDestroyAggregateDevice(AudioObjectID inDeviceID)

    ctypedef enum :
        kAudioPlugInCreateAggregateDevice = 1667327847
        kAudioPlugInDestroyAggregateDevice = 1684105063

    ctypedef enum :
        kAudioTransportManagerCreateEndPointDevice = 1667523958
        kAudioTransportManagerDestroyEndPointDevice = 1684301174
    
    ctypedef OSStatus (*AudioDeviceIOProc)( AudioObjectID inDevice,
                            const AudioTimeStamp* inNow,
                            const AudioBufferList* inInputData,
                            const AudioTimeStamp* inInputTime,
                            AudioBufferList* outOutputData,
                            const AudioTimeStamp* inOutputTime,
                            void* inClientData)

    ctypedef AudioDeviceIOProc AudioDeviceIOProcID

    cdef OSStatus AudioDeviceCreateIOProcID( AudioObjectID inDevice, AudioDeviceIOProc inProc, void* inClientData, AudioDeviceIOProcID * outIOProcID)
    # cdef OSStatus AudioDeviceCreateIOProcIDWithBlock( AudioDeviceIOProcID * outIOProcID, AudioObjectID inDevice, dispatch_queue_t inDispatchQueue, AudioDeviceIOBlock inIOBlock)
    cdef OSStatus AudioDeviceDestroyIOProcID( AudioObjectID inDevice, AudioDeviceIOProcID inIOProcID)
    cdef OSStatus AudioDeviceStart( AudioObjectID inDevice, AudioDeviceIOProcID inProcID)
    cdef OSStatus AudioDeviceStartAtTime( AudioObjectID inDevice, AudioDeviceIOProcID inProcID, AudioTimeStamp* ioRequestedStartTime, UInt32 inFlags)
    cdef OSStatus AudioDeviceStop( AudioObjectID inDevice, AudioDeviceIOProcID inProcID)
    cdef OSStatus AudioDeviceGetCurrentTime( AudioObjectID inDevice, AudioTimeStamp* outTime)
    cdef OSStatus AudioDeviceTranslateTime( AudioObjectID inDevice, const AudioTimeStamp* inTime, AudioTimeStamp* outTime)
    cdef OSStatus AudioDeviceGetNearestStartTime( AudioObjectID inDevice, AudioTimeStamp* ioRequestedStartTime, UInt32 inFlags)

    ctypedef enum:
        kAudioHardwarePropertyDevices = 1684370979
        kAudioHardwarePropertyDefaultInputDevice = 1682533920
        kAudioHardwarePropertyDefaultOutputDevice = 1682929012
        kAudioHardwarePropertyDefaultSystemOutputDevice = 1934587252
        kAudioHardwarePropertyTranslateUIDToDevice = 1969841252
        kAudioHardwarePropertyMixStereoToMono = 1937010031
        kAudioHardwarePropertyPlugInList = 1886152483
        kAudioHardwarePropertyTranslateBundleIDToPlugIn = 1651074160
        kAudioHardwarePropertyTransportManagerList = 1953326883
        kAudioHardwarePropertyTranslateBundleIDToTransportManager = 1953325673
        kAudioHardwarePropertyBoxList = 1651472419
        kAudioHardwarePropertyTranslateUIDToBox = 1969841250
        kAudioHardwarePropertyClockDeviceList = 1668049699
        kAudioHardwarePropertyTranslateUIDToClockDevice = 1969841251
        kAudioHardwarePropertyProcessIsMain = 1835100526
        kAudioHardwarePropertyIsInitingOrExiting = 1768845172
        kAudioHardwarePropertyUserIDChanged = 1702193508
        kAudioHardwarePropertyProcessInputMute = 1886218606
        kAudioHardwarePropertyProcessIsAudible = 1886221684
        kAudioHardwarePropertySleepingIsAllowed = 1936483696
        kAudioHardwarePropertyUnloadingIsAllowed = 1970170980
        kAudioHardwarePropertyHogModeIsAllowed = 1752131442
        kAudioHardwarePropertyUserSessionIsActiveOrHeadless = 1970496882
        kAudioHardwarePropertyServiceRestarted = 1936880500
        kAudioHardwarePropertyPowerHint = 1886353256
        kAudioHardwarePropertyProcessObjectList = 1886548771
        kAudioHardwarePropertyTranslatePIDToProcessObject = 1768174192
        kAudioHardwarePropertyTapList = 1953526563
        kAudioHardwarePropertyTranslateUIDToTap = 1969841268        


    ctypedef enum:
        kAudioDevicePropertyPlugIn = 1886156135
        kAudioDevicePropertyDeviceHasChanged = 1684629094
        kAudioDevicePropertyDeviceIsRunningSomewhere = 1735356005
        kAudioDeviceProcessorOverload = 1870030194
        kAudioDevicePropertyIOStoppedAbnormally = 1937010788
        kAudioDevicePropertyHogMode = 1869180523
        kAudioDevicePropertyBufferFrameSize = 1718839674
        kAudioDevicePropertyBufferFrameSizeRange = 1718843939
        kAudioDevicePropertyUsesVariableBufferFrameSizes = 1986425722
        kAudioDevicePropertyIOCycleUsage = 1852012899
        kAudioDevicePropertyStreamConfiguration = 1936482681
        kAudioDevicePropertyIOProcStreamUsage = 1937077093
        kAudioDevicePropertyActualSampleRate = 1634955892
        kAudioDevicePropertyClockDevice = 1634755428
        kAudioDevicePropertyIOThreadOSWorkgroup = 1869838183
        kAudioDevicePropertyProcessMute = 1634758765


    ctypedef enum:
        kAudioDevicePropertyJackIsConnected = 1784767339
        kAudioDevicePropertyVolumeScalar = 1987013741
        kAudioDevicePropertyVolumeDecibels = 1987013732
        kAudioDevicePropertyVolumeRangeDecibels = 1986290211
        kAudioDevicePropertyVolumeScalarToDecibels = 1983013986
        kAudioDevicePropertyVolumeDecibelsToScalar = 1684157046
        kAudioDevicePropertyStereoPan = 1936744814
        kAudioDevicePropertyStereoPanChannels = 1936748067
        kAudioDevicePropertyMute = 1836414053
        kAudioDevicePropertySolo = 1936682095
        kAudioDevicePropertyPhantomPower = 1885888878
        kAudioDevicePropertyPhaseInvert = 1885893481
        kAudioDevicePropertyClipLight = 1668049264
        kAudioDevicePropertyTalkback = 1952541794
        kAudioDevicePropertyListenback = 1819504226
        kAudioDevicePropertyDataSource = 1936945763
        kAudioDevicePropertyDataSources = 1936941859
        kAudioDevicePropertyDataSourceNameForIDCFString = 1819501422
        kAudioDevicePropertyDataSourceKindForID = 1936941931
        kAudioDevicePropertyClockSource = 1668510307
        kAudioDevicePropertyClockSources = 1668506403
        kAudioDevicePropertyClockSourceNameForIDCFString = 1818456942
        kAudioDevicePropertyClockSourceKindForID = 1668506475
        kAudioDevicePropertyPlayThru = 1953002101
        kAudioDevicePropertyPlayThruSolo = 1953002099
        kAudioDevicePropertyPlayThruVolumeScalar = 1836479331
        kAudioDevicePropertyPlayThruVolumeDecibels = 1836475490
        kAudioDevicePropertyPlayThruVolumeRangeDecibels = 1836475427
        kAudioDevicePropertyPlayThruVolumeScalarToDecibels = 1836462692
        kAudioDevicePropertyPlayThruVolumeDecibelsToScalar = 1836462707
        kAudioDevicePropertyPlayThruStereoPan = 1836281966
        kAudioDevicePropertyPlayThruStereoPanChannels = 1836281891
        kAudioDevicePropertyPlayThruDestination = 1835295859
        kAudioDevicePropertyPlayThruDestinations = 1835295779
        kAudioDevicePropertyPlayThruDestinationNameForIDCFString = 1835295843
        kAudioDevicePropertyChannelNominalLineLevel = 1852601964
        kAudioDevicePropertyChannelNominalLineLevels = 1852601891
        kAudioDevicePropertyChannelNominalLineLevelNameForIDCFString = 1818455660
        kAudioDevicePropertyHighPassFilterSetting = 1751740518
        kAudioDevicePropertyHighPassFilterSettings = 1751740451
        kAudioDevicePropertyHighPassFilterSettingNameForIDCFString = 1751740524
        kAudioDevicePropertySubVolumeScalar = 1937140845
        kAudioDevicePropertySubVolumeDecibels = 1937140836
        kAudioDevicePropertySubVolumeRangeDecibels = 1937138723
        kAudioDevicePropertySubVolumeScalarToDecibels = 1937125988
        kAudioDevicePropertySubVolumeDecibelsToScalar = 1935946358
        kAudioDevicePropertySubMute = 1936553332
        kAudioDevicePropertyVoiceActivityDetectionEnable = 1983996971
        kAudioDevicePropertyVoiceActivityDetectionState = 1983997011

    ctypedef enum:
        kAudioAggregateDevicePropertyFullSubDeviceList = 1735554416
        kAudioAggregateDevicePropertyActiveSubDeviceList = 1634169456
        kAudioAggregateDevicePropertyComposition = 1633906541
        kAudioAggregateDevicePropertyMainSubDevice = 1634562932
        kAudioAggregateDevicePropertyClockDevice = 1634755428
        kAudioAggregateDevicePropertyTapList = 1952542755
        kAudioAggregateDevicePropertySubTapList = 1635017072





cdef extern from "CoreAudio/CoreAudio.h":
    
    # from CoreAudiBaseTypes.h

    cdef enum:
        kAudio_UnimplementedError     = -4
        kAudio_FileNotFoundError      = -43
        kAudio_FilePermissionError    = -54
        kAudio_TooManyFilesOpenError  = -42
        kAudio_BadFilePathError       = 561017960
        kAudio_ParamError             = -50
        kAudio_MemFullError           = -108

    ctypedef struct AudioValueTranslation:
        void*  mInputData
        UInt32 mInputDataSize
        void*  mOutputData
        UInt32 mOutputDataSize

    ctypedef struct AudioValueRange:
        Float64 mMinimum
        Float64 mMaximum

    ctypedef struct AudioBuffer:
        UInt32 mNumberChannels
        UInt32 mDataByteSize
        void*  mData

    ctypedef struct AudioBufferList:
        UInt32      mNumberBuffers
        AudioBuffer mBuffers[1]

    ctypedef UInt32  AudioFormatID
    ctypedef UInt32  AudioFormatFlags

    ctypedef struct AudioStreamBasicDescription:
        Float64             mSampleRate
        AudioFormatID       mFormatID
        AudioFormatFlags    mFormatFlags
        UInt32              mBytesPerPacket
        UInt32              mFramesPerPacket
        UInt32              mBytesPerFrame
        UInt32              mChannelsPerFrame
        UInt32              mBitsPerChannel
        UInt32              mReserved

    cdef Float64    kAudioStreamAnyRate = 0.0

    ctypedef enum:
        kAudioFormatLinearPCM = 1819304813
        kLinearPCMFormatFlagIsFloat = 1
        kLinearPCMFormatFlagIsBigEndian = 2
        kLinearPCMFormatFlagIsSignedInteger = 4
        kLinearPCMFormatFlagIsPacked = 8
        kLinearPCMFormatFlagIsAlignedHigh = 16
        kLinearPCMFormatFlagIsNonInterleaved = 32
        kLinearPCMFormatFlagIsNonMixable = 64


cdef extern from "CoreFoundation/CoreFoundation.h":
    ctypedef struct __CFURL
    ctypedef __CFURL* CFURLRef
    ctypedef struct __CFRunLoop
    ctypedef __CFRunLoop* CFRunLoopRef
    ctypedef struct __CFString
    ctypedef __CFString* CFStringRef
    ctypedef UInt8 AudioFilePermissions
    ctypedef long CFIndex
    ctypedef void* CFAllocatorRef
    ctypedef void* CFTypeRef
    
    cdef CFURLRef CFURLCreateFromFileSystemRepresentation(CFAllocatorRef allocator, const UInt8* buffer, CFIndex bufLen, Boolean isDirectory)
    cdef void CFRelease(CFTypeRef cf)
    cdef CFAllocatorRef kCFAllocatorDefault


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


# C audio player integration is available in the separate audio_player.c/h files
# and demonstrated in actual_audio_player.py


    
