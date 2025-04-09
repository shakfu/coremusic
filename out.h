



enum
{
    kAudioObjectSystemObject = 1
};
typedef OSStatus
(*AudioObjectPropertyListenerProc)( AudioObjectID inObjectID,
                                    UInt32 inNumberAddresses,
                                    const AudioObjectPropertyAddress* inAddresses,
                                    void* inClientData);
typedef void
(^AudioObjectPropertyListenerBlock)( UInt32 inNumberAddresses,
                                        const AudioObjectPropertyAddress* inAddresses);
enum
{
    kAudioObjectPropertyCreator = 'oplg',
    kAudioObjectPropertyListenerAdded = 'lisa',
    kAudioObjectPropertyListenerRemoved = 'lisr'
};

void AudioObjectShow( AudioObjectID inObjectID);

Boolean AudioObjectHasProperty( AudioObjectID inObjectID,
                        const AudioObjectPropertyAddress* inAddress);

OSStatus AudioObjectIsPropertySettable( AudioObjectID inObjectID,
                                const AudioObjectPropertyAddress* inAddress,
                                Boolean* outIsSettable);

OSStatus AudioObjectGetPropertyDataSize( AudioObjectID inObjectID,
                                const AudioObjectPropertyAddress* inAddress,
                                UInt32 inQualifierDataSize,
                                const void* inQualifierData,
                                UInt32* outDataSize);

OSStatus AudioObjectGetPropertyData( AudioObjectID inObjectID,
                            const AudioObjectPropertyAddress* inAddress,
                            UInt32 inQualifierDataSize,
                            const void* inQualifierData,
                            UInt32* ioDataSize,
                            void* outData);

OSStatus AudioObjectSetPropertyData( AudioObjectID inObjectID,
                            const AudioObjectPropertyAddress* inAddress,
                            UInt32 inQualifierDataSize,
                            const void* inQualifierData,
                            UInt32 inDataSize,
                            const void* inData);

OSStatus AudioObjectAddPropertyListener( AudioObjectID inObjectID,
                                const AudioObjectPropertyAddress* inAddress,
                                AudioObjectPropertyListenerProc inListener,
                                void* inClientData);

OSStatus AudioObjectRemovePropertyListener( AudioObjectID inObjectID,
                                    const AudioObjectPropertyAddress* inAddress,
                                    AudioObjectPropertyListenerProc inListener,
                                    void* inClientData);

OSStatus AudioObjectAddPropertyListenerBlock( AudioObjectID inObjectID,
                                        const AudioObjectPropertyAddress* inAddress,
                                        dispatch_queue_t inDispatchQueue,
                                        AudioObjectPropertyListenerBlock inListener);

OSStatus AudioObjectRemovePropertyListenerBlock( AudioObjectID inObjectID,
                                        const AudioObjectPropertyAddress* inAddress,
                                        dispatch_queue_t inDispatchQueue,
                                        AudioObjectPropertyListenerBlock inListener);
enum
{
    kAudioSystemObjectClassID = 'asys'
};

typedef UInt32 AudioHardwarePowerHint

; enum
{
    kAudioHardwarePowerHintNone = 0,
    kAudioHardwarePowerHintFavorSavingPower = 1
};

enum
{
    kAudioHardwarePropertyDevices = 'dev#',
    kAudioHardwarePropertyDefaultInputDevice = 'dIn ',
    kAudioHardwarePropertyDefaultOutputDevice = 'dOut',
    kAudioHardwarePropertyDefaultSystemOutputDevice = 'sOut',
    kAudioHardwarePropertyTranslateUIDToDevice = 'uidd',
    kAudioHardwarePropertyMixStereoToMono = 'stmo',
    kAudioHardwarePropertyPlugInList = 'plg#',
    kAudioHardwarePropertyTranslateBundleIDToPlugIn = 'bidp',
    kAudioHardwarePropertyTransportManagerList = 'tmg#',
    kAudioHardwarePropertyTranslateBundleIDToTransportManager = 'tmbi',
    kAudioHardwarePropertyBoxList = 'box#',
    kAudioHardwarePropertyTranslateUIDToBox = 'uidb',
    kAudioHardwarePropertyClockDeviceList = 'clk#',
    kAudioHardwarePropertyTranslateUIDToClockDevice = 'uidc',
    kAudioHardwarePropertyProcessIsMain = 'main',
    kAudioHardwarePropertyIsInitingOrExiting = 'inot',
    kAudioHardwarePropertyUserIDChanged = 'euid',
    kAudioHardwarePropertyProcessInputMute = 'pmin',
    kAudioHardwarePropertyProcessIsAudible = 'pmut',
    kAudioHardwarePropertySleepingIsAllowed = 'slep',
    kAudioHardwarePropertyUnloadingIsAllowed = 'unld',
    kAudioHardwarePropertyHogModeIsAllowed = 'hogr',
    kAudioHardwarePropertyUserSessionIsActiveOrHeadless = 'user',
    kAudioHardwarePropertyServiceRestarted = 'srst',
    kAudioHardwarePropertyPowerHint = 'powh',
    kAudioHardwarePropertyProcessObjectList = 'prs#',
    kAudioHardwarePropertyTranslatePIDToProcessObject = 'id2p',
    kAudioHardwarePropertyTapList = 'tps#',
    kAudioHardwarePropertyTranslateUIDToTap = 'uidt',
};

OSStatus AudioHardwareUnload(void) __attribute__((availability(macosx,introduced=10.1)));
OSStatus AudioHardwareCreateAggregateDevice( CFDictionaryRef inDescription,
                                    AudioObjectID* outDeviceID) __attribute__((availability(macosx,introduced=10.9)));
OSStatus AudioHardwareDestroyAggregateDevice(AudioObjectID inDeviceID) __attribute__((availability(macosx,introduced=10.9)));

enum
{
    kAudioPlugInCreateAggregateDevice = 'cagg',
    kAudioPlugInDestroyAggregateDevice = 'dagg'
};

enum
{
    kAudioTransportManagerCreateEndPointDevice = 'cdev',
    kAudioTransportManagerDestroyEndPointDevice = 'ddev'

};
typedef OSStatus
(*AudioDeviceIOProc)( AudioObjectID inDevice,
                        const AudioTimeStamp* inNow,
                        const AudioBufferList* inInputData,
                        const AudioTimeStamp* inInputTime,
                        AudioBufferList* outOutputData,
                        const AudioTimeStamp* inOutputTime,
                        void* inClientData);
typedef void
(^AudioDeviceIOBlock)( const AudioTimeStamp* inNow,
                        const AudioBufferList* inInputData,
                        const AudioTimeStamp* inInputTime,
                        AudioBufferList* outOutputData,
                        const AudioTimeStamp* inOutputTime);
typedef AudioDeviceIOProc AudioDeviceIOProcID;
struct AudioHardwareIOProcStreamUsage
{
    void* mIOProc;
    UInt32 mNumberStreams;
    UInt32 mStreamIsOn[1];
};
typedef struct AudioHardwareIOProcStreamUsage AudioHardwareIOProcStreamUsage;
enum
{
    kAudioDeviceStartTimeIsInputFlag = (1 << 0),
    kAudioDeviceStartTimeDontConsultDeviceFlag = (1 << 1),
    kAudioDeviceStartTimeDontConsultHALFlag = (1 << 2)
};
enum
{
    kAudioDevicePropertyPlugIn = 'plug',
    kAudioDevicePropertyDeviceHasChanged = 'diff',
    kAudioDevicePropertyDeviceIsRunningSomewhere = 'gone',
    kAudioDeviceProcessorOverload = 'over',
    kAudioDevicePropertyIOStoppedAbnormally = 'stpd',
    kAudioDevicePropertyHogMode = 'oink',
    kAudioDevicePropertyBufferFrameSize = 'fsiz',
    kAudioDevicePropertyBufferFrameSizeRange = 'fsz#',
    kAudioDevicePropertyUsesVariableBufferFrameSizes = 'vfsz',
    kAudioDevicePropertyIOCycleUsage = 'ncyc',
    kAudioDevicePropertyStreamConfiguration = 'slay',
    kAudioDevicePropertyIOProcStreamUsage = 'suse',
    kAudioDevicePropertyActualSampleRate = 'asrt',
    kAudioDevicePropertyClockDevice = 'apcd',
    kAudioDevicePropertyIOThreadOSWorkgroup = 'oswg',
    kAudioDevicePropertyProcessMute = 'appm'
};
enum
{
    kAudioDevicePropertyJackIsConnected = 'jack',
    kAudioDevicePropertyVolumeScalar = 'volm',
    kAudioDevicePropertyVolumeDecibels = 'vold',
    kAudioDevicePropertyVolumeRangeDecibels = 'vdb#',
    kAudioDevicePropertyVolumeScalarToDecibels = 'v2db',
    kAudioDevicePropertyVolumeDecibelsToScalar = 'db2v',
    kAudioDevicePropertyStereoPan = 'span',
    kAudioDevicePropertyStereoPanChannels = 'spn#',
    kAudioDevicePropertyMute = 'mute',
    kAudioDevicePropertySolo = 'solo',
    kAudioDevicePropertyPhantomPower = 'phan',
    kAudioDevicePropertyPhaseInvert = 'phsi',
    kAudioDevicePropertyClipLight = 'clip',
    kAudioDevicePropertyTalkback = 'talb',
    kAudioDevicePropertyListenback = 'lsnb',
    kAudioDevicePropertyDataSource = 'ssrc',
    kAudioDevicePropertyDataSources = 'ssc#',
    kAudioDevicePropertyDataSourceNameForIDCFString = 'lscn',
    kAudioDevicePropertyDataSourceKindForID = 'ssck',
    kAudioDevicePropertyClockSource = 'csrc',
    kAudioDevicePropertyClockSources = 'csc#',
    kAudioDevicePropertyClockSourceNameForIDCFString = 'lcsn',
    kAudioDevicePropertyClockSourceKindForID = 'csck',
    kAudioDevicePropertyPlayThru = 'thru',
    kAudioDevicePropertyPlayThruSolo = 'thrs',
    kAudioDevicePropertyPlayThruVolumeScalar = 'mvsc',
    kAudioDevicePropertyPlayThruVolumeDecibels = 'mvdb',
    kAudioDevicePropertyPlayThruVolumeRangeDecibels = 'mvd#',
    kAudioDevicePropertyPlayThruVolumeScalarToDecibels = 'mv2d',
    kAudioDevicePropertyPlayThruVolumeDecibelsToScalar = 'mv2s',
    kAudioDevicePropertyPlayThruStereoPan = 'mspn',
    kAudioDevicePropertyPlayThruStereoPanChannels = 'msp#',
    kAudioDevicePropertyPlayThruDestination = 'mdds',
    kAudioDevicePropertyPlayThruDestinations = 'mdd#',
    kAudioDevicePropertyPlayThruDestinationNameForIDCFString = 'mddc',
    kAudioDevicePropertyChannelNominalLineLevel = 'nlvl',
    kAudioDevicePropertyChannelNominalLineLevels = 'nlv#',
    kAudioDevicePropertyChannelNominalLineLevelNameForIDCFString = 'lcnl',
    kAudioDevicePropertyHighPassFilterSetting = 'hipf',
    kAudioDevicePropertyHighPassFilterSettings = 'hip#',
    kAudioDevicePropertyHighPassFilterSettingNameForIDCFString = 'hipl',
    kAudioDevicePropertySubVolumeScalar = 'svlm',
    kAudioDevicePropertySubVolumeDecibels = 'svld',
    kAudioDevicePropertySubVolumeRangeDecibels = 'svd#',
    kAudioDevicePropertySubVolumeScalarToDecibels = 'sv2d',
    kAudioDevicePropertySubVolumeDecibelsToScalar = 'sd2v',
    kAudioDevicePropertySubMute = 'smut',
    kAudioDevicePropertyVoiceActivityDetectionEnable = 'vAd+',
    kAudioDevicePropertyVoiceActivityDetectionState = 'vAdS'

};
OSStatus
AudioDeviceCreateIOProcID( AudioObjectID inDevice,
                            AudioDeviceIOProc inProc,
                            void* inClientData,
                            AudioDeviceIOProcID * _Nonnull outIOProcID);
OSStatus
AudioDeviceCreateIOProcIDWithBlock( AudioDeviceIOProcID * _Nonnull outIOProcID,
                                    AudioObjectID inDevice,
                                    dispatch_queue_t inDispatchQueue,
                                    AudioDeviceIOBlock inIOBlock);
OSStatus
AudioDeviceDestroyIOProcID( AudioObjectID inDevice,
                            AudioDeviceIOProcID inIOProcID);
OSStatus
AudioDeviceStart( AudioObjectID inDevice,
                    AudioDeviceIOProcID inProcID);
OSStatus
AudioDeviceStartAtTime( AudioObjectID inDevice,
                        AudioDeviceIOProcID inProcID,
                        AudioTimeStamp* ioRequestedStartTime,
                        UInt32 inFlags) __attribute__((availability(macosx,introduced=10.3)));
OSStatus
AudioDeviceStop( AudioObjectID inDevice,
                    AudioDeviceIOProcID inProcID);
OSStatus
AudioDeviceGetCurrentTime( AudioObjectID inDevice,
                            AudioTimeStamp* outTime);
OSStatus
AudioDeviceTranslateTime( AudioObjectID inDevice,
                            const AudioTimeStamp* inTime,
                            AudioTimeStamp* outTime);
OSStatus
AudioDeviceGetNearestStartTime( AudioObjectID inDevice,
                                AudioTimeStamp* ioRequestedStartTime,
                                UInt32 inFlags) __attribute__((availability(macosx,introduced=10.3)));
enum
{
    kAudioAggregateDeviceClassID = 'aagg'
};
enum
{
    kAudioAggregateDevicePropertyFullSubDeviceList = 'grup',
    kAudioAggregateDevicePropertyActiveSubDeviceList = 'agrp',
    kAudioAggregateDevicePropertyComposition = 'acom',
    kAudioAggregateDevicePropertyMainSubDevice = 'amst',
    kAudioAggregateDevicePropertyClockDevice = 'apcd',
    kAudioAggregateDevicePropertyTapList = 'tap#',
    kAudioAggregateDevicePropertySubTapList = 'atap',
};
enum
{
 kAudioAggregateDriftCompensationMinQuality = 0,
 kAudioAggregateDriftCompensationLowQuality = 0x20,
 kAudioAggregateDriftCompensationMediumQuality = 0x40,
 kAudioAggregateDriftCompensationHighQuality = 0x60,
 kAudioAggregateDriftCompensationMaxQuality = 0x7F
};
enum
{
    kAudioSubDeviceClassID = 'asub'
};
enum
{
    kAudioSubDeviceDriftCompensationMinQuality __attribute__((availability(macos,introduced=13.0,deprecated=100000,replacement="kAudioAggregateDriftCompensationMinQuality"))) __attribute__((availability(ios,introduced=16.0,deprecated=100000,replacement="kAudioAggregateDriftCompensationMinQuality"))) = 0,
    kAudioSubDeviceDriftCompensationLowQuality __attribute__((availability(macos,introduced=13.0,deprecated=100000,replacement="kAudioAggregateDriftCompensationLowQuality"))) __attribute__((availability(ios,introduced=16.0,deprecated=100000,replacement="kAudioAggregateDriftCompensationLowQuality"))) = 0x20,
    kAudioSubDeviceDriftCompensationMediumQuality __attribute__((availability(macos,introduced=13.0,deprecated=100000,replacement="kAudioAggregateDriftCompensationMediumQuality"))) __attribute__((availability(ios,introduced=16.0,deprecated=100000,replacement="kAudioAggregateDriftCompensationMediumQuality"))) = 0x40,
    kAudioSubDeviceDriftCompensationHighQuality __attribute__((availability(macos,introduced=13.0,deprecated=100000,replacement="kAudioAggregateDriftCompensationHighQuality"))) __attribute__((availability(ios,introduced=16.0,deprecated=100000,replacement="kAudioAggregateDriftCompensationHighQuality"))) = 0x60,
    kAudioSubDeviceDriftCompensationMaxQuality __attribute__((availability(macos,introduced=13.0,deprecated=100000,replacement="kAudioAggregateDriftCompensationMaxQuality"))) __attribute__((availability(ios,introduced=16.0,deprecated=100000,replacement="kAudioAggregateDriftCompensationMaxQuality"))) = 0x7F
};
enum
{
    kAudioSubDevicePropertyExtraLatency = 'xltc',
    kAudioSubDevicePropertyDriftCompensation = 'drft',
    kAudioSubDevicePropertyDriftCompensationQuality = 'drfq'
};
enum
{
    kAudioSubTapClassID = 'stap'
};
enum
{
    kAudioSubTapPropertyExtraLatency = 'xltc',
    kAudioSubTapPropertyDriftCompensation = 'drft',
    kAudioSubTapPropertyDriftCompensationQuality = 'drfq'
};
enum
{
    kAudioProcessClassID = 'clnt'
};
enum
{
    kAudioProcessPropertyPID = 'ppid',
    kAudioProcessPropertyBundleID = 'pbid',
    kAudioProcessPropertyDevices = 'pdv#',
    kAudioProcessPropertyIsRunning = 'pir?',
 kAudioProcessPropertyIsRunningInput = 'piri',
 kAudioProcessPropertyIsRunningOutput = 'piro',
};
enum
{
    kAudioTapClassID = 'tcls'
};
enum
{
    kAudioTapPropertyUID = 'tuid',
    kAudioTapPropertyDescription = 'tdsc',
    kAudioTapPropertyFormat = 'tfmt',
};




enum
{
    kAudioDevicePropertyScopeInput = kAudioObjectPropertyScopeInput,
    kAudioDevicePropertyScopeOutput = kAudioObjectPropertyScopeOutput,
    kAudioDevicePropertyScopePlayThrough = kAudioObjectPropertyScopePlayThrough
};
enum
{
    kAudioPropertyWildcardPropertyID = kAudioObjectPropertySelectorWildcard
};

enum
{
    kAudioPropertyWildcardSection = 0xFF
};

enum
{
    kAudioPropertyWildcardChannel = kAudioObjectPropertyElementWildcard
};
enum
{
    kAudioISubOwnerControlClassID = 'atch'
};
enum
{
    kAudioLevelControlPropertyDecibelsToScalarTransferFunction = 'lctf'
};
typedef UInt32 AudioLevelControlTransferFunction; enum
{
    kAudioLevelControlTranferFunctionLinear = 0,
    kAudioLevelControlTranferFunction1Over3 = 1,
    kAudioLevelControlTranferFunction1Over2 = 2,
    kAudioLevelControlTranferFunction3Over4 = 3,
    kAudioLevelControlTranferFunction3Over2 = 4,
    kAudioLevelControlTranferFunction2Over1 = 5,
    kAudioLevelControlTranferFunction3Over1 = 6,
    kAudioLevelControlTranferFunction4Over1 = 7,
    kAudioLevelControlTranferFunction5Over1 = 8,
    kAudioLevelControlTranferFunction6Over1 = 9,
    kAudioLevelControlTranferFunction7Over1 = 10,
    kAudioLevelControlTranferFunction8Over1 = 11,
    kAudioLevelControlTranferFunction9Over1 = 12,
    kAudioLevelControlTranferFunction10Over1 = 13,
    kAudioLevelControlTranferFunction11Over1 = 14,
    kAudioLevelControlTranferFunction12Over1 = 15
};
typedef AudioObjectPropertySelector AudioHardwarePropertyID;
typedef OSStatus
(*AudioHardwarePropertyListenerProc)( AudioHardwarePropertyID inPropertyID,
                                        void* inClientData);
enum
{
    kAudioHardwarePropertyRunLoop = 'rnlp',
    kAudioHardwarePropertyDeviceForUID = 'duid',
    kAudioHardwarePropertyPlugInForBundleID = 'pibi',
    kAudioHardwarePropertyProcessIsMaster __attribute__((availability(macos,introduced=10.0,deprecated=12.0,replacement="kAudioHardwarePropertyProcessIsMain"))) __attribute__((availability(ios,introduced=2.0,deprecated=15.0,replacement="kAudioHardwarePropertyProcessIsMain"))) __attribute__((availability(watchos,introduced=1.0,deprecated=8.0,replacement="kAudioHardwarePropertyProcessIsMain"))) __attribute__((availability(tvos,introduced=9.0,deprecated=15.0,replacement="kAudioHardwarePropertyProcessIsMain"))) = 'mast'
};
enum
{
    kAudioHardwarePropertyBootChimeVolumeScalar = 'bbvs',
    kAudioHardwarePropertyBootChimeVolumeDecibels = 'bbvd',
    kAudioHardwarePropertyBootChimeVolumeRangeDecibels = 'bbd#',
    kAudioHardwarePropertyBootChimeVolumeScalarToDecibels = 'bv2d',
    kAudioHardwarePropertyBootChimeVolumeDecibelsToScalar = 'bd2v',
    kAudioHardwarePropertyBootChimeVolumeDecibelsToScalarTransferFunction = 'bvtf'
};
OSStatus
AudioHardwareAddRunLoopSource(CFRunLoopSourceRef inRunLoopSource) __attribute__((availability(macosx,introduced=10.3,deprecated=10.7)));
OSStatus
AudioHardwareRemoveRunLoopSource(CFRunLoopSourceRef inRunLoopSource) __attribute__((availability(macosx,introduced=10.3,deprecated=10.7)));
OSStatus
AudioHardwareGetPropertyInfo( AudioHardwarePropertyID inPropertyID,
                                UInt32* outSize,
                                Boolean* outWritable) __attribute__((availability(macosx,introduced=10.0,deprecated=10.6)));
OSStatus
AudioHardwareGetProperty( AudioHardwarePropertyID inPropertyID,
                            UInt32* ioPropertyDataSize,
                            void* outPropertyData) __attribute__((availability(macosx,introduced=10.0,deprecated=10.6)));
OSStatus
AudioHardwareSetProperty( AudioHardwarePropertyID inPropertyID,
                            UInt32 inPropertyDataSize,
                            const void* inPropertyData) __attribute__((availability(macosx,introduced=10.0,deprecated=10.6)));
OSStatus
AudioHardwareAddPropertyListener( AudioHardwarePropertyID inPropertyID,
                                    AudioHardwarePropertyListenerProc inProc,
                                    void* inClientData) __attribute__((availability(macosx,introduced=10.0,deprecated=10.6)));
OSStatus
AudioHardwareRemovePropertyListener( AudioHardwarePropertyID inPropertyID,
                                        AudioHardwarePropertyListenerProc inProc) __attribute__((availability(macosx,introduced=10.0,deprecated=10.6)));
typedef AudioObjectID AudioDeviceID;






typedef AudioObjectPropertySelector AudioDevicePropertyID;
typedef OSStatus
(*AudioDevicePropertyListenerProc)( AudioDeviceID inDevice,
                                    UInt32 inChannel,
                                    Boolean isInput,
                                    AudioDevicePropertyID inPropertyID,
                                    void* inClientData);
enum
{
    kAudioDeviceUnknown = kAudioObjectUnknown
};
enum
{
    kAudioDeviceTransportTypeAutoAggregate = 'fgrp'
};
enum
{
    kAudioDevicePropertyVolumeDecibelsToScalarTransferFunction = 'vctf',
    kAudioDevicePropertyPlayThruVolumeDecibelsToScalarTransferFunction = 'mvtf',
    kAudioDevicePropertyDriverShouldOwniSub = 'isub',
    kAudioDevicePropertySubVolumeDecibelsToScalarTransferFunction = 'svtf'
};
enum
{
    kAudioDevicePropertyDeviceName = 'name',
    kAudioDevicePropertyDeviceNameCFString = kAudioObjectPropertyName,
    kAudioDevicePropertyDeviceManufacturer = 'makr',
    kAudioDevicePropertyDeviceManufacturerCFString = kAudioObjectPropertyManufacturer,
    kAudioDevicePropertyRegisterBufferList = 'rbuf',
    kAudioDevicePropertyBufferSize = 'bsiz',
    kAudioDevicePropertyBufferSizeRange = 'bsz#',
    kAudioDevicePropertyChannelName = 'chnm',
    kAudioDevicePropertyChannelNameCFString = kAudioObjectPropertyElementName,
    kAudioDevicePropertyChannelCategoryName = 'ccnm',
    kAudioDevicePropertyChannelCategoryNameCFString = kAudioObjectPropertyElementCategoryName,
    kAudioDevicePropertyChannelNumberName = 'cnnm',
    kAudioDevicePropertyChannelNumberNameCFString = kAudioObjectPropertyElementNumberName,
    kAudioDevicePropertySupportsMixing = 'mix?',
    kAudioDevicePropertyStreamFormat = 'sfmt',
    kAudioDevicePropertyStreamFormats = 'sfm#',
    kAudioDevicePropertyStreamFormatSupported = 'sfm?',
    kAudioDevicePropertyStreamFormatMatch = 'sfmm',
    kAudioDevicePropertyDataSourceNameForID = 'sscn',
    kAudioDevicePropertyClockSourceNameForID = 'cscn',
    kAudioDevicePropertyPlayThruDestinationNameForID = 'mddn',
    kAudioDevicePropertyChannelNominalLineLevelNameForID = 'cnlv',
    kAudioDevicePropertyHighPassFilterSettingNameForID = 'chip'
};
OSStatus
AudioDeviceAddIOProc( AudioDeviceID inDevice,
                        AudioDeviceIOProc inProc,
                        void* inClientData) __attribute__((availability(macosx,introduced=10.0,deprecated=10.5)));
OSStatus
AudioDeviceRemoveIOProc( AudioDeviceID inDevice,
                            AudioDeviceIOProc inProc) __attribute__((availability(macosx,introduced=10.0,deprecated=10.5)));
OSStatus
AudioDeviceRead( AudioDeviceID inDevice,
                    const AudioTimeStamp* inStartTime,
                    AudioBufferList* outData) __attribute__((availability(macosx,introduced=10.1,deprecated=10.5)));
OSStatus
AudioDeviceGetPropertyInfo( AudioDeviceID inDevice,
                            UInt32 inChannel,
                            Boolean isInput,
                            AudioDevicePropertyID inPropertyID,
                            UInt32* outSize,
                            Boolean* outWritable) __attribute__((availability(macosx,introduced=10.0,deprecated=10.6)));
OSStatus
AudioDeviceGetProperty( AudioDeviceID inDevice,
                        UInt32 inChannel,
                        Boolean isInput,
                        AudioDevicePropertyID inPropertyID,
                        UInt32* ioPropertyDataSize,
                        void* outPropertyData) __attribute__((availability(macosx,introduced=10.0,deprecated=10.6)));
OSStatus
AudioDeviceSetProperty( AudioDeviceID inDevice,
                        const AudioTimeStamp* inWhen,
                        UInt32 inChannel,
                        Boolean isInput,
                        AudioDevicePropertyID inPropertyID,
                        UInt32 inPropertyDataSize,
                        const void* inPropertyData) __attribute__((availability(macosx,introduced=10.0,deprecated=10.6)));
OSStatus
AudioDeviceAddPropertyListener( AudioDeviceID inDevice,
                                UInt32 inChannel,
                                Boolean isInput,
                                AudioDevicePropertyID inPropertyID,
                                AudioDevicePropertyListenerProc inProc,
                                void* inClientData) __attribute__((availability(macosx,introduced=10.0,deprecated=10.6)));
OSStatus
AudioDeviceRemovePropertyListener( AudioDeviceID inDevice,
                                    UInt32 inChannel,
                                    Boolean isInput,
                                    AudioDevicePropertyID inPropertyID,
                                    AudioDevicePropertyListenerProc inProc) __attribute__((availability(macosx,introduced=10.0,deprecated=10.6)));
enum
{
    kAudioAggregateDevicePropertyMasterSubDevice __attribute__((availability(macos,introduced=10.0,deprecated=12.0,replacement="kAudioAggregateDevicePropertyMainSubDevice"))) __attribute__((availability(ios,introduced=2.0,deprecated=15.0,replacement="kAudioAggregateDevicePropertyMainSubDevice"))) __attribute__((availability(watchos,introduced=1.0,deprecated=8.0,replacement="kAudioAggregateDevicePropertyMainSubDevice"))) __attribute__((availability(tvos,introduced=9.0,deprecated=15.0,replacement="kAudioAggregateDevicePropertyMainSubDevice"))) = kAudioAggregateDevicePropertyMainSubDevice
};
typedef AudioObjectID AudioStreamID;
typedef OSStatus
(*AudioStreamPropertyListenerProc)( AudioStreamID inStream,
                                    UInt32 inChannel,
                                    AudioDevicePropertyID inPropertyID,
                                    void* inClientData);
enum
{
    kAudioStreamUnknown = kAudioObjectUnknown
};
enum
{
    kAudioStreamPropertyOwningDevice = kAudioObjectPropertyOwner,
    kAudioStreamPropertyPhysicalFormats = 'pft#',
    kAudioStreamPropertyPhysicalFormatSupported = 'pft?',
    kAudioStreamPropertyPhysicalFormatMatch = 'pftm'
};
OSStatus
AudioStreamGetPropertyInfo( AudioStreamID inStream,
                            UInt32 inChannel,
                            AudioDevicePropertyID inPropertyID,
                            UInt32* outSize,
                            Boolean* outWritable) __attribute__((availability(macosx,introduced=10.1,deprecated=10.6)));
OSStatus
AudioStreamGetProperty( AudioStreamID inStream,
                        UInt32 inChannel,
                        AudioDevicePropertyID inPropertyID,
                        UInt32* ioPropertyDataSize,
                        void* outPropertyData) __attribute__((availability(macosx,introduced=10.1,deprecated=10.6)));
OSStatus
AudioStreamSetProperty( AudioStreamID inStream,
                        const AudioTimeStamp* inWhen,
                        UInt32 inChannel,
                        AudioDevicePropertyID inPropertyID,
                        UInt32 inPropertyDataSize,
                        const void* inPropertyData) __attribute__((availability(macosx,introduced=10.1,deprecated=10.6)));
OSStatus
AudioStreamAddPropertyListener( AudioStreamID inStream,
                                UInt32 inChannel,
                                AudioDevicePropertyID inPropertyID,
                                AudioStreamPropertyListenerProc inProc,
                                void* inClientData) __attribute__((availability(macosx,introduced=10.1,deprecated=10.6)));
OSStatus
AudioStreamRemovePropertyListener( AudioStreamID inStream,
                                    UInt32 inChannel,
                                    AudioDevicePropertyID inPropertyID,
                                    AudioStreamPropertyListenerProc inProc) __attribute__((availability(macosx,introduced=10.1,deprecated=10.6)));
enum
{
    kAudioBootChimeVolumeControlClassID = 'pram'
};
enum
{
    kAudioControlPropertyVariant = 'cvar'
};
enum
{
    kAudioClockSourceControlPropertyItemKind = kAudioSelectorControlPropertyItemKind
};

