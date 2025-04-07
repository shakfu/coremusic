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
    # ctypedef UInt32 AudioObjectPropertySelector
    ctypedef UInt32 AudioObjectPropertyScope
    ctypedef UInt32 AudioObjectPropertyElement

    ctypedef struct AudioObjectPropertyAddress:
        AudioObjectPropertySelector mSelector
        AudioObjectPropertyScope    mScope
        AudioObjectPropertyElement  mElement


cdef extern from "CoreAudio/AudioHardware.h":

    ctypedef enum AudioObjectPropertySelector:
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
    # cdef OSStatus AudioObjectAddPropertyListenerBlock( AudioObjectID inObjectID, const AudioObjectPropertyAddress* inAddress, dispatch_queue_t  inDispatchQueue, AudioObjectPropertyListenerBlock inListener)

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

    
