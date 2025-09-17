# corefoundation.pxd
# CoreFoundation framework declarations for coremusic

cdef extern from *:
    """
    #define FOURCC_ARGS(x)  (char)((x & 0xff000000) >> 24), \
        (char)((x & 0xff0000) >> 16),                   \
        (char)((x & 0xff00) >> 8), (char)((x) & 0xff)
    """
    ctypedef unsigned long uint64_t
    ctypedef long int64_t
    cdef char[] FOURCC_ARGS(SInt32)

# -----------------------------------------------------------------------------

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
    ctypedef UInt32          FourCharCode
    ctypedef FourCharCode    OSType

cdef extern from "CoreFoundation/CoreFoundation.h":
    ctypedef struct __CFURL
    ctypedef __CFURL* CFURLRef
    ctypedef struct __CFRunLoop
    ctypedef __CFRunLoop* CFRunLoopRef
    ctypedef struct __CFString
    ctypedef __CFString* CFStringRef
    ctypedef struct __CFData
    ctypedef __CFData* CFDataRef
    ctypedef UInt8 AudioFilePermissions
    ctypedef long CFIndex
    ctypedef void* CFAllocatorRef
    ctypedef void* CFTypeRef

    cdef CFURLRef CFURLCreateFromFileSystemRepresentation(CFAllocatorRef allocator, const UInt8* buffer, CFIndex bufLen, Boolean isDirectory)
    cdef void CFRelease(CFTypeRef cf)
    cdef CFAllocatorRef kCFAllocatorDefault

    # String creation and encoding
    ctypedef UInt32 CFStringEncoding
    cdef CFStringEncoding kCFStringEncodingUTF8
    cdef CFStringRef CFStringCreateWithCString(CFAllocatorRef allocator, const char* cStr, CFStringEncoding encoding)
    cdef char* CFStringGetCStringPtr(CFStringRef theString, CFStringEncoding encoding)
    cdef CFIndex CFStringGetLength(CFStringRef theString)
    cdef CFIndex CFStringGetMaximumSizeForEncoding(CFIndex length, CFStringEncoding encoding)
    cdef Boolean CFStringGetCString(CFStringRef theString, char* buffer, CFIndex bufferSize, CFStringEncoding encoding)

    # CFData functions
    cdef CFDataRef CFDataCreate(CFAllocatorRef allocator, const UInt8* bytes, CFIndex length)
    cdef CFIndex CFDataGetLength(CFDataRef theData)
    cdef UInt8* CFDataGetBytePtr(CFDataRef theData)