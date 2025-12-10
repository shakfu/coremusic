# CoreMIDI API declarations for Cython
# Based on headers/coremidi/MIDIServices.h

from libc.stdint cimport *

from .coreaudio cimport *
from .coreaudiotypes cimport *
from .corefoundation cimport *


# CoreMIDI external declarations
cdef extern from "CoreMIDI/MIDIServices.h":

    # Basic types
    ctypedef UInt32 MIDIObjectRef
    ctypedef MIDIObjectRef MIDIClientRef
    ctypedef MIDIObjectRef MIDIPortRef
    ctypedef MIDIObjectRef MIDIDeviceRef
    ctypedef MIDIObjectRef MIDIEntityRef
    ctypedef MIDIObjectRef MIDIEndpointRef

    ctypedef UInt32 MIDIProtocolID
    ctypedef UInt32 MIDITimeStamp
    ctypedef UInt32 MIDIUniqueID

    # Error constants
    ctypedef enum:
        kMIDIInvalidClient = -10830
        kMIDIInvalidPort = -10831
        kMIDIWrongEndpointType = -10832
        kMIDINoConnection = -10833
        kMIDIUnknownEndpoint = -10834
        kMIDIUnknownProperty = -10835
        kMIDIWrongPropertyType = -10836
        kMIDINoCurrentSetup = -10837
        kMIDIMessageSendErr = -10838
        kMIDIServerStartErr = -10839
        kMIDISetupFormatErr = -10840
        kMIDIWrongThread = -10841
        kMIDIObjectNotFound = -10842
        kMIDIIDNotUnique = -10843
        kMIDINotPermitted = -10844
        kMIDIUnknownError = -10845

    # Object types
    ctypedef enum:
        kMIDIObjectType_Other = -1
        kMIDIObjectType_Device = 0
        kMIDIObjectType_Entity = 1
        kMIDIObjectType_Source = 2
        kMIDIObjectType_Destination = 3
        kMIDIObjectType_ExternalDevice = 0x10
        kMIDIObjectType_ExternalEntity = 0x11
        kMIDIObjectType_ExternalSource = 0x12
        kMIDIObjectType_ExternalDestination = 0x13

    # Protocol IDs
    ctypedef enum:
        kMIDIProtocol_1_0 = 1
        kMIDIProtocol_2_0 = 2

    # MIDI Packet structure (legacy)
    ctypedef struct MIDIPacket:
        MIDITimeStamp timeStamp
        UInt16 length
        UInt8 data[256]

    ctypedef struct MIDIPacketList:
        UInt32 numPackets
        MIDIPacket packet[1]

    # UMP structures (MIDI 2.0)
    ctypedef struct MIDIEventPacket:
        MIDITimeStamp timeStamp
        UInt32 wordCount
        UInt32 words[64]

    ctypedef struct MIDIEventList:
        MIDIProtocolID protocol
        UInt16 numPackets
        MIDIEventPacket packet[1]

    # Callback types
    ctypedef void (*MIDINotifyProc)(
        const void* message,
        void* refCon)

    ctypedef void (*MIDIReadProc)(
        const MIDIPacketList* pktlist,
        void* readProcRefCon,
        void* srcConnRefCon)

    ctypedef void (*MIDICompletionProc)(
        void* request)

    # Block types (forward declarations)
    # Note: These are Objective-C blocks that cannot be directly represented in Cython.
    # Actual signatures:
    #   MIDINotifyBlock = void (^)(const MIDINotification *message)
    #   MIDIReceiveBlock = void (^)(const MIDIPacketList *pktlist, void *srcConnRefCon)
    # Use MIDINotifyProc/MIDIReadProc callback functions instead for Cython compatibility.
    ctypedef void* MIDINotifyBlock
    ctypedef void* MIDIReceiveBlock

    # Client functions
    cdef OSStatus MIDIClientCreate(
        CFStringRef name,
        MIDINotifyProc notifyProc,
        void* notifyRefCon,
        MIDIClientRef* outClient)

    cdef OSStatus MIDIClientDispose(MIDIClientRef client)

    # Port functions
    cdef OSStatus MIDIInputPortCreate(
        MIDIClientRef client,
        CFStringRef portName,
        MIDIReadProc readProc,
        void* refCon,
        MIDIPortRef* outPort)

    cdef OSStatus MIDIOutputPortCreate(
        MIDIClientRef client,
        CFStringRef portName,
        MIDIPortRef* outPort)

    cdef OSStatus MIDIPortDispose(MIDIPortRef port)

    cdef OSStatus MIDIPortConnectSource(
        MIDIPortRef port,
        MIDIEndpointRef source,
        void* connRefCon)

    cdef OSStatus MIDIPortDisconnectSource(
        MIDIPortRef port,
        MIDIEndpointRef source)

    # Device and endpoint functions
    cdef UInt32 MIDIGetNumberOfDevices()

    cdef MIDIDeviceRef MIDIGetDevice(UInt32 deviceIndex0)

    cdef UInt32 MIDIDeviceGetNumberOfEntities(MIDIDeviceRef device)

    cdef MIDIEntityRef MIDIDeviceGetEntity(
        MIDIDeviceRef device,
        UInt32 entityIndex0)

    cdef UInt32 MIDIEntityGetNumberOfSources(MIDIEntityRef entity)

    cdef MIDIEndpointRef MIDIEntityGetSource(
        MIDIEntityRef entity,
        UInt32 sourceIndex0)

    cdef UInt32 MIDIEntityGetNumberOfDestinations(MIDIEntityRef entity)

    cdef MIDIEndpointRef MIDIEntityGetDestination(
        MIDIEntityRef entity,
        UInt32 destIndex0)

    cdef UInt32 MIDIGetNumberOfSources()

    cdef MIDIEndpointRef MIDIGetSource(UInt32 sourceIndex0)

    cdef UInt32 MIDIGetNumberOfDestinations()

    cdef MIDIEndpointRef MIDIGetDestination(UInt32 destIndex0)

    # Virtual endpoint functions
    cdef OSStatus MIDISourceCreate(
        MIDIClientRef client,
        CFStringRef name,
        MIDIEndpointRef* outSrc)

    cdef OSStatus MIDIDestinationCreate(
        MIDIClientRef client,
        CFStringRef name,
        MIDIReadProc readProc,
        void* refCon,
        MIDIEndpointRef* outDest)

    cdef OSStatus MIDIEndpointDispose(MIDIEndpointRef endpt)

    # Property functions
    cdef OSStatus MIDIObjectGetIntegerProperty(
        MIDIObjectRef obj,
        CFStringRef propertyID,
        SInt32* outValue)

    cdef OSStatus MIDIObjectSetIntegerProperty(
        MIDIObjectRef obj,
        CFStringRef propertyID,
        SInt32 value)

    cdef OSStatus MIDIObjectGetStringProperty(
        MIDIObjectRef obj,
        CFStringRef propertyID,
        CFStringRef* outString)

    cdef OSStatus MIDIObjectSetStringProperty(
        MIDIObjectRef obj,
        CFStringRef propertyID,
        CFStringRef string)

    # Note: Dictionary and data property functions omitted for now due to type complexity

    # Send functions
    cdef OSStatus MIDISend(
        MIDIPortRef port,
        MIDIEndpointRef dest,
        const MIDIPacketList* pktlist)

    cdef OSStatus MIDISendSysex(
        void* request)

    cdef OSStatus MIDIReceived(
        MIDIEndpointRef src,
        const MIDIPacketList* pktlist)

    # Packet list utilities
    cdef MIDIPacket* MIDIPacketListInit(MIDIPacketList* pktlist)

    cdef MIDIPacket* MIDIPacketListAdd(
        MIDIPacketList* pktlist,
        UInt32 listSize,
        MIDIPacket* curPacket,
        MIDITimeStamp time,
        UInt32 nData,
        const UInt8* data)

    # Property constants
    cdef extern const CFStringRef kMIDIPropertyName
    cdef extern const CFStringRef kMIDIPropertyManufacturer
    cdef extern const CFStringRef kMIDIPropertyModel
    cdef extern const CFStringRef kMIDIPropertyUniqueID
    cdef extern const CFStringRef kMIDIPropertyDeviceID
    cdef extern const CFStringRef kMIDIPropertyReceiveChannels
    cdef extern const CFStringRef kMIDIPropertyTransmitChannels
    cdef extern const CFStringRef kMIDIPropertyMaxSysExSpeed
    cdef extern const CFStringRef kMIDIPropertyAdvanceScheduleTimeMuSec
    cdef extern const CFStringRef kMIDIPropertyIsEmbeddedEntity
    cdef extern const CFStringRef kMIDIPropertyIsBroadcast
    cdef extern const CFStringRef kMIDIPropertySingleRealtimeEntity
    cdef extern const CFStringRef kMIDIPropertyConnectionUniqueID
    cdef extern const CFStringRef kMIDIPropertyOffline
    cdef extern const CFStringRef kMIDIPropertyPrivate
    cdef extern const CFStringRef kMIDIPropertyDriverOwner
    cdef extern const CFStringRef kMIDIPropertySupportsMMC
    cdef extern const CFStringRef kMIDIPropertySupportsGeneralMIDI
    cdef extern const CFStringRef kMIDIPropertySupportsShowControl
    cdef extern const CFStringRef kMIDIPropertyImage
    cdef extern const CFStringRef kMIDIPropertyDriverVersion
    cdef extern const CFStringRef kMIDIPropertyDisplayName

# CoreMIDI MIDIMessages.h API declarations
cdef extern from "CoreMIDI/MIDIMessages.h":

    # MIDI Universal Packet message type nibbles
    ctypedef enum MIDIMessageType:
        kMIDIMessageTypeUtility = 0x0
        kMIDIMessageTypeSystem = 0x1
        kMIDIMessageTypeChannelVoice1 = 0x2
        kMIDIMessageTypeSysEx = 0x3
        kMIDIMessageTypeChannelVoice2 = 0x4
        kMIDIMessageTypeData128 = 0x5
        kMIDIMessageTypeUnknownF = 0xF

    # Channel Voice status nibbles
    ctypedef enum MIDICVStatus:
        # MIDI 1.0
        kMIDICVStatusNoteOff = 0x8
        kMIDICVStatusNoteOn = 0x9
        kMIDICVStatusPolyPressure = 0xA
        kMIDICVStatusControlChange = 0xB
        kMIDICVStatusProgramChange = 0xC
        kMIDICVStatusChannelPressure = 0xD
        kMIDICVStatusPitchBend = 0xE
        # MIDI 2.0
        kMIDICVStatusRegisteredPNC = 0x0
        kMIDICVStatusAssignablePNC = 0x1
        kMIDICVStatusRegisteredControl = 0x2
        kMIDICVStatusAssignableControl = 0x3
        kMIDICVStatusRelRegisteredControl = 0x4
        kMIDICVStatusRelAssignableControl = 0x5
        kMIDICVStatusPerNotePitchBend = 0x6
        kMIDICVStatusPerNoteMgmt = 0xF

    # System status bytes
    ctypedef enum MIDISystemStatus:
        kMIDIStatusStartOfExclusive = 0xF0
        kMIDIStatusEndOfExclusive = 0xF7
        kMIDIStatusMTC = 0xF1
        kMIDIStatusSongPosPointer = 0xF2
        kMIDIStatusSongSelect = 0xF3
        kMIDIStatusTuneRequest = 0xF6
        kMIDIStatusTimingClock = 0xF8
        kMIDIStatusStart = 0xFA
        kMIDIStatusContinue = 0xFB
        kMIDIStatusStop = 0xFC
        kMIDIStatusActiveSending = 0xFE
        kMIDIStatusActiveSensing = 0xFE
        kMIDIStatusSystemReset = 0xFF

    # SysEx status nibbles
    ctypedef enum MIDISysExStatus:
        kMIDISysExStatusComplete = 0x0
        kMIDISysExStatusStart = 0x1
        kMIDISysExStatusContinue = 0x2
        kMIDISysExStatusEnd = 0x3
        kMIDISysExStatusMixedDataSetHeader = 0x8
        kMIDISysExStatusMixedDataSetPayload = 0x9

    # Utility status nibbles
    ctypedef enum MIDIUtilityStatus:
        kMIDIUtilityStatusNOOP = 0x0
        kMIDIUtilityStatusJitterReductionClock = 0x1
        kMIDIUtilityStatusJitterReductionTimestamp = 0x2

    # MIDI 2.0 Note Attribute Types
    ctypedef enum MIDINoteAttribute:
        kMIDINoteAttributeNone = 0x0
        kMIDINoteAttributeManufacturerSpecific = 0x1
        kMIDINoteAttributeProfileSpecific = 0x2
        kMIDINoteAttributePitch = 0x3

    # MIDI 2.0 Program Change Options
    ctypedef enum MIDIProgramChangeOptions:
        kMIDIProgramChangeBankValid = 0x1

    # MIDI 2.0 Per Note Management Options
    ctypedef enum MIDIPerNoteManagementOptions:
        kMIDIPerNoteManagementReset = 0x1
        kMIDIPerNoteManagementDetach = 0x2

    # Universal MIDI Packet structs
    ctypedef UInt32 MIDIMessage_32

    ctypedef struct MIDIMessage_64:
        UInt32 word0
        UInt32 word1

    ctypedef struct MIDIMessage_96:
        UInt32 word0
        UInt32 word1
        UInt32 word2

    ctypedef struct MIDIMessage_128:
        UInt32 word0
        UInt32 word1
        UInt32 word2
        UInt32 word3

    # Universal MIDI Packet message helper functions
    cdef MIDIMessageType MIDIMessageTypeForUPWord(const UInt32 word)

    # MIDI 1.0 Universal MIDI Packet helper functions
    cdef MIDIMessage_32 MIDI1UPChannelVoiceMessage(UInt8 group, UInt8 status, UInt8 channel, UInt8 data1, UInt8 data2)
    cdef MIDIMessage_32 MIDI1UPNoteOff(UInt8 group, UInt8 channel, UInt8 noteNumber, UInt8 velocity)
    cdef MIDIMessage_32 MIDI1UPNoteOn(UInt8 group, UInt8 channel, UInt8 noteNumber, UInt8 velocity)
    cdef MIDIMessage_32 MIDI1UPControlChange(UInt8 group, UInt8 channel, UInt8 index, UInt8 data)
    cdef MIDIMessage_32 MIDI1UPPitchBend(UInt8 group, UInt8 channel, UInt8 lsb, UInt8 msb)
    cdef MIDIMessage_32 MIDI1UPSystemCommon(UInt8 group, UInt8 status, UInt8 byte1, UInt8 byte2)
    cdef MIDIMessage_64 MIDI1UPSysEx(UInt8 group, UInt8 status, UInt8 bytesUsed, UInt8 byte1, UInt8 byte2, UInt8 byte3, UInt8 byte4, UInt8 byte5, UInt8 byte6)
    cdef MIDIMessage_64 MIDI1UPSysExArray(UInt8 group, UInt8 status, const UInt8* begin, const UInt8* end)

    # MIDI 2.0 Channel Voice Message helper functions
    cdef MIDIMessage_64 MIDI2ChannelVoiceMessage(UInt8 group, UInt8 status, UInt8 channel, UInt16 index, UInt32 value)
    cdef MIDIMessage_64 MIDI2NoteOn(UInt8 group, UInt8 channel, UInt8 noteNumber, UInt8 attributeType, UInt16 attributeData, UInt16 velocity)
    cdef MIDIMessage_64 MIDI2NoteOff(UInt8 group, UInt8 channel, UInt8 noteNumber, UInt8 attributeType, UInt16 attributeData, UInt16 velocity)
    cdef MIDIMessage_64 MIDI2PolyPressure(UInt8 group, UInt8 channel, UInt8 noteNumber, UInt32 value)
    cdef MIDIMessage_64 MIDI2RegisteredPNC(UInt8 group, UInt8 channel, UInt8 noteNumber, UInt8 index, UInt32 value)
    cdef MIDIMessage_64 MIDI2AssignablePNC(UInt8 group, UInt8 channel, UInt8 noteNumber, UInt8 index, UInt32 value)
    cdef MIDIMessage_64 MIDI2PerNoteManagment(UInt8 group, UInt8 channel, UInt8 noteNumber, bint detachPNCs, bint resetPNCsToDefault)
    cdef MIDIMessage_64 MIDI2ControlChange(UInt8 group, UInt8 channel, UInt8 index, UInt32 value)
    cdef MIDIMessage_64 MIDI2RegisteredControl(UInt8 group, UInt8 channel, UInt8 bank, UInt8 index, UInt32 value)
    cdef MIDIMessage_64 MIDI2AssignableControl(UInt8 group, UInt8 channel, UInt8 bank, UInt8 index, UInt32 value)
    cdef MIDIMessage_64 MIDI2RelRegisteredControl(UInt8 group, UInt8 channel, UInt8 bank, UInt8 index, UInt32 value)
    cdef MIDIMessage_64 MIDI2RelAssignableControl(UInt8 group, UInt8 channel, UInt8 bank, UInt8 index, UInt32 value)
    cdef MIDIMessage_64 MIDI2ProgramChange(UInt8 group, UInt8 channel, bint bankIsValid, UInt8 program, UInt8 bank_msb, UInt8 bank_lsb)
    cdef MIDIMessage_64 MIDI2ChannelPressure(UInt8 group, UInt8 channel, UInt32 value)
    cdef MIDIMessage_64 MIDI2PitchBend(UInt8 group, UInt8 channel, UInt32 value)
    cdef MIDIMessage_64 MIDI2PerNotePitchBend(UInt8 group, UInt8 channel, UInt8 noteNumber, UInt32 value)

    # Universal Message structure and visitor types
    ctypedef struct MIDIUniversalMessage:
        MIDIMessageType type
        UInt8 group
        UInt8 reserved[3]
        # Union members are simplified for Cython - full structure is complex

    ctypedef void (*MIDIEventVisitor)(void* context, MIDITimeStamp timeStamp, MIDIUniversalMessage message)

    # Event list parsing function
    cdef void MIDIEventListForEachEvent(
        const MIDIEventList* evtlist,
        MIDIEventVisitor visitor,
        void* visitorContext)

# CoreMIDI MIDISetup.h API declarations
cdef extern from "CoreMIDI/MIDISetup.h":

    # MIDISetup type (typedef from MIDIObjectRef)
    ctypedef MIDIObjectRef MIDISetupRef

    # Item count type for endpoints
    ctypedef UInt32 ItemCount

    # Device and Entity Management Functions (Available, non-deprecated)

    # MIDIDeviceNewEntity - Available from macOS 11.0+, iOS 14.0+
    cdef OSStatus MIDIDeviceNewEntity(
        MIDIDeviceRef device,
        CFStringRef name,
        MIDIProtocolID protocol,
        Boolean embedded,
        ItemCount numSourceEndpoints,
        ItemCount numDestinationEndpoints,
        MIDIEntityRef* newEntity)

    # DEPRECATED: Use MIDIDeviceNewEntity instead (macOS 11.0+, iOS 14.0+)
    # MIDIDeviceAddEntity doesn't support MIDI 2.0 protocol specification.
    # Kept for backwards compatibility with older macOS/iOS versions.
    cdef OSStatus MIDIDeviceAddEntity(
        MIDIDeviceRef device,
        CFStringRef name,
        Boolean embedded,
        ItemCount numSourceEndpoints,
        ItemCount numDestinationEndpoints,
        MIDIEntityRef* newEntity)

    # MIDIDeviceRemoveEntity - Available from macOS 10.1+, iOS 4.2+
    cdef OSStatus MIDIDeviceRemoveEntity(
        MIDIDeviceRef device,
        MIDIEntityRef entity)

    # MIDIEntityAddOrRemoveEndpoints - Available from macOS 10.2+, iOS 4.2+
    cdef OSStatus MIDIEntityAddOrRemoveEndpoints(
        MIDIEntityRef entity,
        ItemCount numSourceEndpoints,
        ItemCount numDestinationEndpoints)

    # Setup Device Management Functions

    # MIDISetupAddDevice - Available from macOS 10.1+, iOS 4.2+
    cdef OSStatus MIDISetupAddDevice(MIDIDeviceRef device)

    # MIDISetupRemoveDevice - Available from macOS 10.1+, iOS 4.2+
    cdef OSStatus MIDISetupRemoveDevice(MIDIDeviceRef device)

    # MIDISetupAddExternalDevice - Available from macOS 10.1+, iOS 4.2+
    cdef OSStatus MIDISetupAddExternalDevice(MIDIDeviceRef device)

    # MIDISetupRemoveExternalDevice - Available from macOS 10.1+, iOS 4.2+
    cdef OSStatus MIDISetupRemoveExternalDevice(MIDIDeviceRef device)

    # External Device Creation

    # MIDIExternalDeviceCreate - Available from macOS 10.1+, iOS 4.2+
    cdef OSStatus MIDIExternalDeviceCreate(
        CFStringRef name,
        CFStringRef manufacturer,
        CFStringRef model,
        MIDIDeviceRef* outDevice)

    # Note: The following functions are deprecated and not included:
    # - MIDISetupCreate (deprecated in macOS 10.6)
    # - MIDISetupDispose (deprecated in macOS 10.6)
    # - MIDISetupInstall (deprecated in macOS 10.6)
    # - MIDISetupGetCurrent (deprecated in macOS 10.6)
    # - MIDISetupToData (deprecated in macOS 10.6)
    # - MIDISetupFromData (deprecated in macOS 10.6)
    # - MIDIGetSerialPortOwner (deprecated in macOS 10.6)
    # - MIDISetSerialPortOwner (deprecated in macOS 10.6)
    # - MIDIGetSerialPortDrivers (deprecated in macOS 10.6)

# CoreMIDI MIDIDriver.h API declarations
cdef extern from "CoreMIDI/MIDIDriver.h":

    # Driver and device list types
    ctypedef void* MIDIDriverRef
    ctypedef MIDIObjectRef MIDIDeviceListRef

    # Device Creation and Management Functions (Available to non-drivers)

    # MIDIDeviceCreate - Available from macOS 10.0+, iOS 4.2+
    cdef OSStatus MIDIDeviceCreate(
        MIDIDriverRef owner,
        CFStringRef name,
        CFStringRef manufacturer,
        CFStringRef model,
        MIDIDeviceRef* outDevice)

    # MIDIDeviceDispose - Available from macOS 10.3+, iOS 4.2+
    cdef OSStatus MIDIDeviceDispose(MIDIDeviceRef device)

    # Device List Management Functions

    # MIDIDeviceListGetNumberOfDevices - Available from macOS 10.0+, iOS 4.2+
    cdef ItemCount MIDIDeviceListGetNumberOfDevices(MIDIDeviceListRef devList)

    # MIDIDeviceListGetDevice - Available from macOS 10.0+, iOS 4.2+
    cdef MIDIDeviceRef MIDIDeviceListGetDevice(
        MIDIDeviceListRef devList,
        ItemCount index0)

    # MIDIDeviceListAddDevice - Available from macOS 10.0+, iOS 4.2+
    cdef OSStatus MIDIDeviceListAddDevice(
        MIDIDeviceListRef devList,
        MIDIDeviceRef dev)

    # MIDIDeviceListDispose - Available from macOS 10.1+, iOS 4.2+
    cdef OSStatus MIDIDeviceListDispose(MIDIDeviceListRef devList)

    # Endpoint RefCon Management Functions

    # MIDIEndpointSetRefCons - Available from macOS 10.0+, iOS 4.2+
    cdef OSStatus MIDIEndpointSetRefCons(
        MIDIEndpointRef endpt,
        void* ref1,
        void* ref2)

    # MIDIEndpointGetRefCons - Available from macOS 10.0+, iOS 4.2+
    cdef OSStatus MIDIEndpointGetRefCons(
        MIDIEndpointRef endpt,
        void** ref1,
        void** ref2)

    # Driver Utility Functions

    # MIDIGetDriverIORunLoop - Available from macOS 10.0+, iOS 4.2+
    cdef CFRunLoopRef MIDIGetDriverIORunLoop()

    # MIDIGetDriverDeviceList - Available from macOS 10.1+, iOS 4.2+
    cdef MIDIDeviceListRef MIDIGetDriverDeviceList(MIDIDriverRef driver)

    # MIDIDriverEnableMonitoring - Available from macOS 10.1+
    cdef OSStatus MIDIDriverEnableMonitoring(
        MIDIDriverRef driver,
        Boolean enabled)

    # Note: Complex driver interface structures and COM-style functions are omitted
    # as they are primarily used in CFPlugIn driver development, not general use

# CoreMIDI MIDIThruConnection.h API declarations
cdef extern from "CoreMIDI/MIDIThruConnection.h":

    # Thru connection reference type
    ctypedef MIDIObjectRef MIDIThruConnectionRef

    # Constants
    enum:
        kMIDIThruConnection_MaxEndpoints = 8

    # Transform types
    ctypedef enum MIDITransformType:
        kMIDITransform_None = 0
        kMIDITransform_FilterOut = 1
        kMIDITransform_MapControl = 2
        kMIDITransform_Add = 8
        kMIDITransform_Scale = 9
        kMIDITransform_MinValue = 10
        kMIDITransform_MaxValue = 11
        kMIDITransform_MapValue = 12

    # Control types
    ctypedef enum MIDITransformControlType:
        kMIDIControlType_7Bit = 0
        kMIDIControlType_14Bit = 1
        kMIDIControlType_7BitRPN = 2
        kMIDIControlType_14BitRPN = 3
        kMIDIControlType_7BitNRPN = 4
        kMIDIControlType_14BitNRPN = 5

    # Value mapping structure
    ctypedef struct MIDIValueMap:
        UInt8 value[128]

    # Transform structures
    ctypedef struct MIDITransform:
        MIDITransformType transform
        SInt16 param

    ctypedef struct MIDIControlTransform:
        MIDITransformControlType controlType
        MIDITransformControlType remappedControlType
        UInt16 controlNumber
        MIDITransformType transform
        SInt16 param

    # Endpoint description
    ctypedef struct MIDIThruConnectionEndpoint:
        MIDIEndpointRef endpointRef
        MIDIUniqueID uniqueID

    # Main connection parameters structure
    ctypedef struct MIDIThruConnectionParams:
        UInt32 version
        UInt32 numSources
        MIDIThruConnectionEndpoint sources[8]  # kMIDIThruConnection_MaxEndpoints
        UInt32 numDestinations
        MIDIThruConnectionEndpoint destinations[8]  # kMIDIThruConnection_MaxEndpoints

        UInt8 channelMap[16]
        UInt8 lowVelocity
        UInt8 highVelocity
        UInt8 lowNote
        UInt8 highNote
        MIDITransform noteNumber
        MIDITransform velocity
        MIDITransform keyPressure
        MIDITransform channelPressure
        MIDITransform programChange
        MIDITransform pitchBend

        UInt8 filterOutSysEx
        UInt8 filterOutMTC
        UInt8 filterOutBeatClock
        UInt8 filterOutTuneRequest
        UInt8 reserved2[3]
        UInt8 filterOutAllControls

        UInt16 numControlTransforms
        UInt16 numMaps
        UInt16 reserved3[4]

    # Function declarations

    # MIDIThruConnectionParamsInitialize - Available from macOS 10.2+, iOS 4.2+
    cdef void MIDIThruConnectionParamsInitialize(MIDIThruConnectionParams* inConnectionParams)

    # MIDIThruConnectionCreate - Available from macOS 10.2+, iOS 4.2+
    cdef OSStatus MIDIThruConnectionCreate(
        CFStringRef inPersistentOwnerID,
        CFDataRef inConnectionParams,
        MIDIThruConnectionRef* outConnection)

    # MIDIThruConnectionDispose - Available from macOS 10.2+, iOS 4.2+
    cdef OSStatus MIDIThruConnectionDispose(MIDIThruConnectionRef connection)

    # MIDIThruConnectionGetParams - Available from macOS 10.2+, iOS 4.2+
    cdef OSStatus MIDIThruConnectionGetParams(
        MIDIThruConnectionRef connection,
        CFDataRef* outConnectionParams)

    # MIDIThruConnectionSetParams - Available from macOS 10.2+, iOS 4.2+
    cdef OSStatus MIDIThruConnectionSetParams(
        MIDIThruConnectionRef connection,
        CFDataRef inConnectionParams)

    # MIDIThruConnectionFind - Available from macOS 10.2+, iOS 4.2+
    cdef OSStatus MIDIThruConnectionFind(
        CFStringRef inPersistentOwnerID,
        CFDataRef* outConnectionList)