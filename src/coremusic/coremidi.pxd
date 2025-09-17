# CoreMIDI API declarations for Cython
# Based on headers/coremidi/MIDIServices.h

from libc.stdint cimport *
from .corefoundation cimport *
from .coreaudiotypes cimport *
from .coreaudio cimport *

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

    # Note: Property constants omitted for now due to CFStringRef extern issues

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

    # MIDIDeviceAddEntity - Deprecated but still available for compatibility
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