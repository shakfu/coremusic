from .coreaudiotypes cimport *
from .corefoundation cimport *


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

    # Note: Block-based property listener APIs are not exposed due to Cython limitations
    # with Objective-C blocks and dispatch_queue_t. Use the callback-based versions above:
    #   AudioObjectAddPropertyListener / AudioObjectRemovePropertyListener
    # Original signatures:
    #   OSStatus AudioObjectAddPropertyListenerBlock(AudioObjectID, const AudioObjectPropertyAddress*,
    #       dispatch_queue_t, AudioObjectPropertyListenerBlock)
    #   OSStatus AudioObjectRemovePropertyListenerBlock(AudioObjectID, const AudioObjectPropertyAddress*,
    #       dispatch_queue_t, AudioObjectPropertyListenerBlock)

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

    # Note: Block-based IO proc API not exposed due to Cython limitations with
    # Objective-C blocks and dispatch_queue_t. Use AudioDeviceCreateIOProcID above.
    # Original signature:
    #   OSStatus AudioDeviceCreateIOProcIDWithBlock(AudioDeviceIOProcID*, AudioObjectID,
    #       dispatch_queue_t, AudioDeviceIOBlock)
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

    # Common AudioObject properties (for devices)
    ctypedef enum:
        kAudioObjectPropertyName = 1819173229  # 'lnam'
        kAudioObjectPropertyManufacturer = 1819107691  # 'lmak'
        kAudioObjectPropertyElementName = 1818454126  # 'lchn'
        kAudioObjectPropertyOwnedObjects = 1870098020  # 'ownd'

    # AudioObject property scopes
    ctypedef enum:
        kAudioObjectPropertyScopeGlobal = 1735159650  # 'glob'
        kAudioObjectPropertyScopeInput = 1768845428  # 'inpt'
        kAudioObjectPropertyScopeOutput = 1869968496  # 'outp'
        kAudioObjectPropertyScopePlayThrough = 1886681200  # 'ptru'

    # AudioObject property elements
    ctypedef enum:
        kAudioObjectPropertyElementMain = 0  # Main element
        # DEPRECATED: Use kAudioObjectPropertyElementMain instead (macOS 12.0+)
        kAudioObjectPropertyElementMaster = 0  # Deprecated alias for Main

    ctypedef enum:
        kAudioDevicePropertyDeviceUID = 1969841184  # 'uid '
        kAudioDevicePropertyModelUID = 1836411236  # 'muid'
        kAudioDevicePropertyTransportType = 1953653102  # 'tran'
        kAudioDevicePropertyDeviceIsAlive = 1819898990  # 'livn'
        kAudioDevicePropertyDeviceCanBeDefaultDevice = 1684629862  # 'dflt'
        kAudioDevicePropertyNominalSampleRate = 1853059700  # 'nsrt'
        kAudioDevicePropertyAvailableNominalSampleRates = 1853059619  # 'nsr#'
        kAudioDevicePropertyIsHidden = 1751737454  # 'hidn'
        kAudioDevicePropertyPreferredChannelsForStereo = 1684237420  # 'dch2'
        kAudioDevicePropertyPreferredChannelLayout = 1936879468  # 'srnd'

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


# C audio player integration is available in the separate audio_player.c/h files
# and demonstrated in actual_audio_player.py
