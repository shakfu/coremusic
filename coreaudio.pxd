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
        kAudioHardwarePowerHintNone = 0,
        kAudioHardwarePowerHintFavorSavingPower = 1

    ctypedef enum:
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

    
