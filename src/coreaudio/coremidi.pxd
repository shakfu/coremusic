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