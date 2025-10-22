from . cimport corefoundation as cf
from . cimport coreaudiotypes as ct
from . cimport audiotoolbox as at
from . cimport coreaudio as ca
from . cimport coremidi as cm

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

def fourchar_to_int(code: str) -> int:
   """Convert fourcc chars to an int

   >>> fourchar_to_int('TEXT')
   1413830740
   """
   assert len(code) == 4, "should be four characters only"
   return ((ord(code[0]) << 24) | (ord(code[1]) << 16) |
           (ord(code[2]) << 8)  | ord(code[3]))

def int_to_fourchar(n: int) -> str:
    """convert int to fourcc 4 chars

    >>> int_to_fourchar(1413830740)
    'TEXT'
    """
    return (
          chr((n >> 24) & 255)
        + chr((n >> 16) & 255)
        + chr((n >> 8) & 255)
        + chr((n & 255))
    )

def audio_object_show(int object_id):
    """Show an AudioObject"""
    ca.AudioObjectShow(object_id)

def audio_hardware_unload() -> int:
    """Unload the audio hardware"""
    return ca.AudioHardwareUnload()


def audio_hardware_destroy_aggregate_device(int in_device_id) -> int:
    """Destroy an audio hardware aggregate device"""
    return ca.AudioHardwareDestroyAggregateDevice(in_device_id)


# Audio Hardware Device Functions
def audio_object_get_property_data(int object_id, int property_selector, int scope, int element):
    """Get property data from an AudioObject

    WARNING: For string properties (like device names), use audio_object_get_property_string instead.
    This function returns raw bytes which may be a CFStringRef pointer for string properties.
    """
    cdef ca.AudioObjectPropertyAddress address
    cdef cf.UInt32 data_size = 0
    cdef cf.OSStatus status
    cdef char* buffer
    cdef bytes result

    address.mSelector = property_selector
    address.mScope = scope
    address.mElement = element

    # Get the data size
    status = ca.AudioObjectGetPropertyDataSize(object_id, &address, 0, <void*>0, &data_size)
    if status != 0:
        raise RuntimeError(f"AudioObjectGetPropertyDataSize failed with status: {status}")

    if data_size == 0:
        return b''

    # Allocate buffer
    buffer = <char*>malloc(data_size)
    if buffer == <char*>0:
        raise MemoryError("Failed to allocate buffer")

    try:
        status = ca.AudioObjectGetPropertyData(object_id, &address, 0, <void*>0, &data_size, buffer)
        if status != 0:
            raise RuntimeError(f"AudioObjectGetPropertyData failed with status: {status}")

        # Copy to Python bytes object
        result = buffer[:data_size]
        return result
    finally:
        free(buffer)


def audio_object_get_property_string(int object_id, int property_selector, int scope, int element):
    """Get a string property from an AudioObject

    This properly handles CFStringRef properties by dereferencing the pointer
    and extracting the actual string content using CoreFoundation APIs.
    """
    cdef ca.AudioObjectPropertyAddress address
    cdef cf.UInt32 data_size = sizeof(cf.CFStringRef)
    cdef cf.OSStatus status
    cdef cf.CFStringRef string_ref
    cdef char buffer[1024]
    cdef bytes result
    cdef cf.CFIndex length
    cdef cf.CFIndex max_size
    cdef char* large_buffer

    address.mSelector = property_selector
    address.mScope = scope
    address.mElement = element

    # Get the CFStringRef
    status = ca.AudioObjectGetPropertyData(object_id, &address, 0, <void*>0, &data_size, &string_ref)
    if status != 0:
        raise RuntimeError(f"AudioObjectGetPropertyData failed with status: {status}")

    if string_ref == <cf.CFStringRef>0:
        return b''

    # Extract the C string from the CFStringRef
    if cf.CFStringGetCString(string_ref, buffer, sizeof(buffer), cf.kCFStringEncodingUTF8):
        result = buffer
        return result

    # Fallback: try to get length and allocate larger buffer
    length = cf.CFStringGetLength(string_ref)
    max_size = cf.CFStringGetMaximumSizeForEncoding(length, cf.kCFStringEncodingUTF8)
    large_buffer = <char*>malloc(max_size + 1)

    if large_buffer == <char*>0:
        return b''

    try:
        if cf.CFStringGetCString(string_ref, large_buffer, max_size + 1, cf.kCFStringEncodingUTF8):
            result = large_buffer
            return result
        else:
            return b''
    finally:
        free(large_buffer)


def audio_hardware_get_devices() -> list:
    """Get list of all audio device IDs"""
    cdef ca.AudioObjectPropertyAddress address
    cdef cf.UInt32 data_size = 0
    cdef cf.OSStatus status
    cdef ca.AudioObjectID* device_ids
    cdef int device_count
    cdef list devices = []

    address.mSelector = ca.kAudioHardwarePropertyDevices
    address.mScope = 0  # kAudioObjectPropertyScopeGlobal
    address.mElement = 0  # kAudioObjectPropertyElementMain

    # Get the data size
    status = ca.AudioObjectGetPropertyDataSize(ca.kAudioObjectSystemObject, &address, 0, NULL, &data_size)
    if status != 0:
        return []

    device_count = data_size // sizeof(ca.AudioObjectID)
    if device_count == 0:
        return []

    # Allocate buffer
    device_ids = <ca.AudioObjectID*>malloc(data_size)
    if device_ids == NULL:
        return []

    try:
        status = ca.AudioObjectGetPropertyData(ca.kAudioObjectSystemObject, &address, 0, NULL, &data_size, device_ids)
        if status == 0:
            for i in range(device_count):
                devices.append(device_ids[i])
    finally:
        free(device_ids)

    return devices


def audio_hardware_get_default_output_device() -> int:
    """Get the default output device ID"""
    cdef ca.AudioObjectPropertyAddress address
    cdef cf.UInt32 data_size = sizeof(ca.AudioObjectID)
    cdef ca.AudioObjectID device_id = 0
    cdef cf.OSStatus status

    address.mSelector = ca.kAudioHardwarePropertyDefaultOutputDevice
    address.mScope = 0  # kAudioObjectPropertyScopeGlobal
    address.mElement = 0  # kAudioObjectPropertyElementMain

    status = ca.AudioObjectGetPropertyData(ca.kAudioObjectSystemObject, &address, 0, NULL, &data_size, &device_id)
    if status != 0:
        return 0

    return device_id


def audio_hardware_get_default_input_device() -> int:
    """Get the default input device ID"""
    cdef ca.AudioObjectPropertyAddress address
    cdef cf.UInt32 data_size = sizeof(ca.AudioObjectID)
    cdef ca.AudioObjectID device_id = 0
    cdef cf.OSStatus status

    address.mSelector = ca.kAudioHardwarePropertyDefaultInputDevice
    address.mScope = 0  # kAudioObjectPropertyScopeGlobal
    address.mElement = 0  # kAudioObjectPropertyElementMain

    status = ca.AudioObjectGetPropertyData(ca.kAudioObjectSystemObject, &address, 0, NULL, &data_size, &device_id)
    if status != 0:
        return 0

    return device_id


# Audio File Functions
def audio_file_open_url(str file_path, int permissions=1, int file_type_hint=0):
    """Open an audio file at the given path"""
    cdef at.AudioFileID audio_file
    cdef cf.CFURLRef url_ref
    cdef bytes path_bytes = file_path.encode('utf-8')

    url_ref = cf.CFURLCreateFromFileSystemRepresentation(
        cf.kCFAllocatorDefault,
        <const cf.UInt8*>path_bytes,
        len(path_bytes),
        False
    )

    if not url_ref:
        raise ValueError("Could not create URL from file path")

    cdef cf.OSStatus status = at.AudioFileOpenURL(
        url_ref,
        <at.AudioFilePermissions>permissions,
        <at.AudioFileTypeID>file_type_hint,
        &audio_file
    )

    cf.CFRelease(url_ref)

    if status != 0:
        raise RuntimeError(f"AudioFileOpenURL failed with status: {status}")

    return <long>audio_file


def audio_file_close(long audio_file_id):
    """Close an audio file"""
    cdef at.AudioFileID audio_file = <at.AudioFileID>audio_file_id
    cdef cf.OSStatus status = at.AudioFileClose(audio_file)
    if status != 0:
        raise RuntimeError(f"AudioFileClose failed with status: {status}")
    return status


def audio_file_get_property(long audio_file_id, int property_id):
    """Get a property from an audio file"""
    cdef at.AudioFileID audio_file = <at.AudioFileID>audio_file_id
    cdef cf.UInt32 data_size = 0
    cdef cf.UInt32 writable = 0

    # Get the size of the property data
    cdef cf.OSStatus status = at.AudioFileGetPropertyInfo(
        audio_file,
        <at.AudioFilePropertyID>property_id,
        &data_size,
        &writable
    )

    if status != 0:
        raise RuntimeError(f"AudioFileGetPropertyInfo failed with status: {status}")

    # Allocate buffer and get the property data
    cdef char* buffer = <char*>malloc(data_size)
    if not buffer:
        raise MemoryError("Could not allocate buffer for property data")

    try:
        status = at.AudioFileGetProperty(
            audio_file,
            <at.AudioFilePropertyID>property_id,
            &data_size,
            buffer
        )

        if status != 0:
            raise RuntimeError(f"AudioFileGetProperty failed with status: {status}")

        # Return the data as bytes
        return buffer[:data_size]

    finally:
        free(buffer)


def audio_file_read_packets(long audio_file_id, long start_packet, int num_packets):
    """Read packets from an audio file"""
    cdef at.AudioFileID audio_file = <at.AudioFileID>audio_file_id
    cdef cf.UInt32 num_bytes = 0
    cdef cf.UInt32 packet_count = <cf.UInt32>num_packets

    # First get the maximum packet size to determine buffer size
    cdef cf.UInt32 max_packet_size = 0
    cdef cf.UInt32 prop_size = sizeof(cf.UInt32)

    cdef cf.OSStatus status = at.AudioFileGetProperty(
        audio_file,
        at.kAudioFilePropertyMaximumPacketSize,
        &prop_size,
        &max_packet_size
    )

    if status != 0:
        raise RuntimeError(f"Could not get maximum packet size: {status}")

    # Allocate buffer
    cdef cf.UInt32 buffer_size = max_packet_size * packet_count
    cdef char* buffer = <char*>malloc(buffer_size)
    if not buffer:
        raise MemoryError("Could not allocate buffer for packet data")

    try:
        num_bytes = buffer_size
        status = at.AudioFileReadPackets(
            audio_file,
            False,  # don't use cache
            &num_bytes,
            NULL,   # no packet descriptions
            <cf.SInt64>start_packet,
            &packet_count,
            buffer
        )

        if status != 0:
            raise RuntimeError(f"AudioFileReadPackets failed with status: {status}")

        return buffer[:num_bytes], packet_count

    finally:
        free(buffer)


# AudioFileStream Functions
# Dummy callback functions to avoid NULL pointer issues
cdef void dummy_property_listener(void* client_data, at.AudioFileStreamID stream,
                                 at.AudioFileStreamPropertyID property_id,
                                 at.AudioFileStreamPropertyFlags* flags) noexcept:
    """Dummy property listener callback"""
    pass

cdef void dummy_packets_callback(void* client_data, cf.UInt32 num_bytes,
                                cf.UInt32 num_packets, const void* input_data,
                                ct.AudioStreamPacketDescription* packet_descriptions) noexcept:
    """Dummy packets callback"""
    pass

def audio_file_stream_open(file_type_hint=0):
    """Open an AudioFileStream parser for streaming audio data"""
    cdef at.AudioFileStreamID stream_id

    cdef cf.OSStatus status = at.AudioFileStreamOpen(
        NULL,  # client data
        dummy_property_listener,  # property listener proc
        dummy_packets_callback,  # packets proc
        <at.AudioFileTypeID>file_type_hint,
        &stream_id
    )

    if status != 0:
        raise RuntimeError(f"AudioFileStreamOpen failed with status: {status}")

    return <long>stream_id


def audio_file_stream_close(long stream_id):
    """Close an AudioFileStream parser"""
    cdef at.AudioFileStreamID stream = <at.AudioFileStreamID>stream_id
    cdef cf.OSStatus status = at.AudioFileStreamClose(stream)
    if status != 0:
        raise RuntimeError(f"AudioFileStreamClose failed with status: {status}")
    return status


def audio_file_stream_parse_bytes(long stream_id, bytes data, int flags=0):
    """Parse bytes through the AudioFileStream parser"""
    cdef at.AudioFileStreamID stream = <at.AudioFileStreamID>stream_id
    cdef char* data_ptr = <char*>data
    cdef cf.UInt32 data_size = len(data)

    cdef cf.OSStatus status = at.AudioFileStreamParseBytes(
        stream,
        data_size,
        <const void*>data_ptr,
        <at.AudioFileStreamParseFlags>flags
    )

    if status != 0:
        raise RuntimeError(f"AudioFileStreamParseBytes failed with status: {status}")

    return status


def audio_file_stream_get_property(long stream_id, int property_id):
    """Get a property from an AudioFileStream parser"""
    cdef at.AudioFileStreamID stream = <at.AudioFileStreamID>stream_id
    cdef cf.UInt32 data_size = 0
    cdef cf.Boolean writable = 0

    # Get the size of the property data
    cdef cf.OSStatus status = at.AudioFileStreamGetPropertyInfo(
        stream,
        <at.AudioFileStreamPropertyID>property_id,
        &data_size,
        &writable
    )

    if status != 0:
        raise RuntimeError(f"AudioFileStreamGetPropertyInfo failed with status: {status}")

    # Allocate buffer and get the property data
    cdef char* buffer = <char*>malloc(data_size)
    if not buffer:
        raise MemoryError("Could not allocate memory for property data")

    cdef cf.UInt32 actual_size = data_size
    cdef at.AudioStreamBasicDescription* asbd
    try:
        status = at.AudioFileStreamGetProperty(
            stream,
            <at.AudioFileStreamPropertyID>property_id,
            &actual_size,
            buffer
        )

        if status != 0:
            raise RuntimeError(f"AudioFileStreamGetProperty failed with status: {status}")

        # Handle different property types
        if property_id == at.kAudioFileStreamProperty_DataFormat:
            # Return AudioStreamBasicDescription as dict
            desc = <at.AudioStreamBasicDescription*>buffer
            return {
                'sample_rate': desc.mSampleRate,
                'format_id': desc.mFormatID,
                'format_flags': desc.mFormatFlags,
                'bytes_per_packet': desc.mBytesPerPacket,
                'frames_per_packet': desc.mFramesPerPacket,
                'bytes_per_frame': desc.mBytesPerFrame,
                'channels_per_frame': desc.mChannelsPerFrame,
                'bits_per_channel': desc.mBitsPerChannel,
                'reserved': desc.mReserved
            }
        elif property_id in [at.kAudioFileStreamProperty_ReadyToProducePackets,
                           at.kAudioFileStreamProperty_FileFormat,
                           at.kAudioFileStreamProperty_MaximumPacketSize,
                           at.kAudioFileStreamProperty_AudioDataPacketCount,
                           at.kAudioFileStreamProperty_BitRate]:
            # Return scalar values
            if data_size == 4:
                return (<cf.UInt32*>buffer)[0]
            elif data_size == 8:
                return (<cf.UInt64*>buffer)[0]
        elif property_id in [at.kAudioFileStreamProperty_AudioDataByteCount,
                           at.kAudioFileStreamProperty_DataOffset]:
            # Return 64-bit values
            return (<cf.UInt64*>buffer)[0]
        else:
            # Return raw bytes for other properties
            return buffer[:actual_size]

    finally:
        free(buffer)


def audio_file_stream_seek(long stream_id, long packet_offset):
    """Seek to a packet offset in the AudioFileStream"""
    cdef at.AudioFileStreamID stream = <at.AudioFileStreamID>stream_id
    cdef cf.SInt64 byte_offset = 0
    cdef at.AudioFileStreamSeekFlags flags
    flags = <at.AudioFileStreamSeekFlags>0

    cdef cf.OSStatus status = at.AudioFileStreamSeek(
        stream,
        <cf.SInt64>packet_offset,
        &byte_offset,
        &flags
    )

    if status != 0:
        raise RuntimeError(f"AudioFileStreamSeek failed with status: {status}")

    return {
        'byte_offset': byte_offset,
        'flags': flags,
        'is_estimated': bool(flags & at.kAudioFileStreamSeekFlag_OffsetIsEstimated)
    }


# ============================================================================
# AudioConverter Functions
# ============================================================================

def audio_converter_new(source_format: dict, dest_format: dict) -> int:
    """Create a new AudioConverter

    Args:
        source_format: Source audio format dictionary (AudioStreamBasicDescription)
        dest_format: Destination audio format dictionary (AudioStreamBasicDescription)

    Returns:
        AudioConverter ID
    """
    cdef at.AudioStreamBasicDescription src_format
    cdef at.AudioStreamBasicDescription dst_format
    cdef at.AudioConverterRef converter

    # Set up source format
    src_format.mSampleRate = source_format.get('sample_rate', 44100.0)
    src_format.mFormatID = source_format.get('format_id', ca.kAudioFormatLinearPCM)
    src_format.mFormatFlags = source_format.get('format_flags',
        ca.kLinearPCMFormatFlagIsSignedInteger | ca.kLinearPCMFormatFlagIsPacked)
    src_format.mBytesPerPacket = source_format.get('bytes_per_packet', 4)
    src_format.mFramesPerPacket = source_format.get('frames_per_packet', 1)
    src_format.mBytesPerFrame = source_format.get('bytes_per_frame', 4)
    src_format.mChannelsPerFrame = source_format.get('channels_per_frame', 2)
    src_format.mBitsPerChannel = source_format.get('bits_per_channel', 16)
    src_format.mReserved = 0

    # Set up destination format
    dst_format.mSampleRate = dest_format.get('sample_rate', 44100.0)
    dst_format.mFormatID = dest_format.get('format_id', ca.kAudioFormatLinearPCM)
    dst_format.mFormatFlags = dest_format.get('format_flags',
        ca.kLinearPCMFormatFlagIsSignedInteger | ca.kLinearPCMFormatFlagIsPacked)
    dst_format.mBytesPerPacket = dest_format.get('bytes_per_packet', 4)
    dst_format.mFramesPerPacket = dest_format.get('frames_per_packet', 1)
    dst_format.mBytesPerFrame = dest_format.get('bytes_per_frame', 4)
    dst_format.mChannelsPerFrame = dest_format.get('channels_per_frame', 2)
    dst_format.mBitsPerChannel = dest_format.get('bits_per_channel', 16)
    dst_format.mReserved = 0

    cdef cf.OSStatus status = at.AudioConverterNew(&src_format, &dst_format, &converter)
    if status != 0:
        raise RuntimeError(f"AudioConverterNew failed with status: {status}")

    return <long>converter


def audio_converter_dispose(long converter_id):
    """Dispose of an AudioConverter"""
    cdef at.AudioConverterRef converter = <at.AudioConverterRef>converter_id
    cdef cf.OSStatus status = at.AudioConverterDispose(converter)
    if status != 0:
        raise RuntimeError(f"AudioConverterDispose failed with status: {status}")


def audio_converter_convert_buffer(long converter_id, bytes input_data) -> bytes:
    """Convert audio data using AudioConverter

    Args:
        converter_id: AudioConverter ID
        input_data: Input audio data bytes

    Returns:
        Converted audio data bytes
    """
    cdef at.AudioConverterRef converter = <at.AudioConverterRef>converter_id
    cdef cf.UInt32 input_data_size = len(input_data)
    cdef cf.UInt32 output_data_size = input_data_size * 4  # Allocate extra space
    cdef char* input_buffer = <char*><bytes>input_data
    cdef char* output_buffer = <char*>malloc(output_data_size)
    cdef cf.OSStatus status

    if output_buffer == <char*>0:
        raise MemoryError("Failed to allocate output buffer")

    try:
        status = at.AudioConverterConvertBuffer(
            converter,
            input_data_size,
            input_buffer,
            &output_data_size,
            output_buffer
        )

        if status != 0:
            raise RuntimeError(f"AudioConverterConvertBuffer failed with status: {status}")

        result = output_buffer[:output_data_size]
        return result
    finally:
        free(output_buffer)


def audio_converter_get_property(long converter_id, int property_id) -> bytes:
    """Get a property from an AudioConverter

    Args:
        converter_id: AudioConverter ID
        property_id: Property ID

    Returns:
        Property data as bytes
    """
    cdef at.AudioConverterRef converter = <at.AudioConverterRef>converter_id
    cdef cf.UInt32 data_size = 0
    cdef cf.OSStatus status

    # Get property size
    status = at.AudioConverterGetPropertyInfo(converter, property_id, &data_size, NULL)
    if status != 0:
        raise RuntimeError(f"AudioConverterGetPropertyInfo failed with status: {status}")

    if data_size == 0:
        return b''

    cdef char* buffer = <char*>malloc(data_size)
    if buffer == <char*>0:
        raise MemoryError("Failed to allocate buffer")

    try:
        status = at.AudioConverterGetProperty(converter, property_id, &data_size, buffer)
        if status != 0:
            raise RuntimeError(f"AudioConverterGetProperty failed with status: {status}")

        result = buffer[:data_size]
        return result
    finally:
        free(buffer)


def audio_converter_set_property(long converter_id, int property_id, bytes data):
    """Set a property on an AudioConverter

    Args:
        converter_id: AudioConverter ID
        property_id: Property ID
        data: Property data as bytes
    """
    cdef at.AudioConverterRef converter = <at.AudioConverterRef>converter_id
    cdef cf.UInt32 data_size = len(data)
    cdef char* buffer = <char*><bytes>data

    cdef cf.OSStatus status = at.AudioConverterSetProperty(converter, property_id, data_size, buffer)
    if status != 0:
        raise RuntimeError(f"AudioConverterSetProperty failed with status: {status}")


def audio_converter_reset(long converter_id):
    """Reset an AudioConverter to its initial state"""
    cdef at.AudioConverterRef converter = <at.AudioConverterRef>converter_id
    cdef cf.OSStatus status = at.AudioConverterReset(converter)
    if status != 0:
        raise RuntimeError(f"AudioConverterReset failed with status: {status}")


# ============================================================================
# AudioConverter Callback-based Complex Conversion
# ============================================================================

cdef struct AudioConverterCallbackData:
    char* input_buffer         # Pointer to input data
    cf.UInt32 input_buffer_size # Total size of input buffer
    cf.UInt32 packets_read     # Packets consumed so far
    cf.UInt32 total_packets    # Total packets available
    cf.UInt32 packet_size      # Size of each packet in bytes
    at.AudioStreamBasicDescription source_format  # Input format


cdef at.OSStatus audio_converter_input_callback(
    at.AudioConverterRef converter,
    cf.UInt32* num_packets,
    at.AudioBufferList* buffer_list,
    at.AudioStreamPacketDescription** packet_descriptions,
    void* user_data
) noexcept nogil:
    """Callback that provides input data to AudioConverter on demand

    This function is called by AudioConverter when it needs more input data.
    It must:
    1. Read the requested number of packets from input
    2. Fill the AudioBufferList with data
    3. Update num_packets with actual count provided
    4. Return 0 for success, or error code
    """
    cdef AudioConverterCallbackData* data
    cdef cf.UInt32 requested_packets
    cdef cf.UInt32 available_packets
    cdef cf.UInt32 packets_to_provide
    cdef cf.UInt32 bytes_to_copy
    cdef char* source_ptr
    cdef char* dest_ptr
    cdef cf.UInt32 offset

    if user_data == NULL:
        return -50  # paramErr

    data = <AudioConverterCallbackData*>user_data
    requested_packets = num_packets[0]

    # Calculate how many packets we can provide
    available_packets = data.total_packets - data.packets_read
    if available_packets == 0:
        num_packets[0] = 0
        return 0  # End of data

    packets_to_provide = requested_packets if requested_packets < available_packets else available_packets
    bytes_to_copy = packets_to_provide * data.packet_size

    # Calculate offset into input buffer
    offset = data.packets_read * data.packet_size

    # Get pointer to input data
    source_ptr = data.input_buffer + offset

    # Fill AudioBufferList
    if buffer_list.mNumberBuffers > 0:
        dest_ptr = <char*>buffer_list.mBuffers[0].mData
        memcpy(dest_ptr, source_ptr, bytes_to_copy)
        buffer_list.mBuffers[0].mDataByteSize = bytes_to_copy
        buffer_list.mBuffers[0].mNumberChannels = data.source_format.mChannelsPerFrame

    # Update packets read
    data.packets_read += packets_to_provide
    num_packets[0] = packets_to_provide

    return 0  # noErr


def audio_converter_fill_complex_buffer(
    long converter_id,
    bytes input_data,
    int input_packet_count,
    int output_packet_count,
    dict source_format_dict
) -> tuple:
    """Convert audio using callback-based API for complex conversions

    This method supports all types of conversions including:
    - Sample rate changes (e.g., 44.1kHz -> 48kHz)
    - Bit depth changes (e.g., 16-bit -> 24-bit)
    - Channel count changes (stereo <-> mono)
    - Codec conversions (e.g., MP3 -> PCM)

    Args:
        converter_id: AudioConverter ID
        input_data: Input audio data as bytes
        input_packet_count: Number of packets in input data
        output_packet_count: Number of output packets to produce
        source_format_dict: Source AudioStreamBasicDescription as dict

    Returns:
        (output_data, actual_packet_count) tuple

    Raises:
        RuntimeError: If conversion fails
        MemoryError: If buffer allocation fails
    """
    cdef at.AudioConverterRef converter = <at.AudioConverterRef>converter_id
    cdef AudioConverterCallbackData callback_data
    cdef at.AudioBufferList* output_buffer_list
    cdef cf.UInt32 output_data_packet_size = output_packet_count
    cdef cf.OSStatus status
    cdef char* output_buffer
    cdef cf.UInt32 output_buffer_size
    cdef bytes result
    cdef char* input_ptr

    # Get pointer to input data (Python bytes object)
    input_ptr = <char*><bytes>input_data

    # Initialize callback data
    callback_data.input_buffer = input_ptr
    callback_data.input_buffer_size = len(input_data)
    callback_data.packets_read = 0
    callback_data.total_packets = input_packet_count
    callback_data.packet_size = source_format_dict['bytes_per_packet']

    # Fill source format structure
    callback_data.source_format.mSampleRate = source_format_dict['sample_rate']
    callback_data.source_format.mFormatID = source_format_dict['format_id']
    callback_data.source_format.mFormatFlags = source_format_dict['format_flags']
    callback_data.source_format.mBytesPerPacket = source_format_dict['bytes_per_packet']
    callback_data.source_format.mFramesPerPacket = source_format_dict['frames_per_packet']
    callback_data.source_format.mBytesPerFrame = source_format_dict['bytes_per_frame']
    callback_data.source_format.mChannelsPerFrame = source_format_dict['channels_per_frame']
    callback_data.source_format.mBitsPerChannel = source_format_dict['bits_per_channel']
    callback_data.source_format.mReserved = 0

    # Allocate output buffer (estimate size generously)
    # For sample rate conversion, output size = input_size * (output_rate / input_rate)
    output_buffer_size = len(input_data) * 4  # 4x for safety
    output_buffer = <char*>malloc(output_buffer_size)
    if output_buffer == NULL:
        raise MemoryError("Failed to allocate output buffer")

    # Allocate and initialize AudioBufferList
    output_buffer_list = <at.AudioBufferList*>malloc(sizeof(at.AudioBufferList) + sizeof(at.AudioBuffer))
    if output_buffer_list == NULL:
        free(output_buffer)
        raise MemoryError("Failed to allocate AudioBufferList")

    try:
        output_buffer_list.mNumberBuffers = 1
        output_buffer_list.mBuffers[0].mNumberChannels = callback_data.source_format.mChannelsPerFrame
        output_buffer_list.mBuffers[0].mDataByteSize = output_buffer_size
        output_buffer_list.mBuffers[0].mData = output_buffer

        # Call AudioConverterFillComplexBuffer
        status = at.AudioConverterFillComplexBuffer(
            converter,
            audio_converter_input_callback,  # Our callback function
            &callback_data,                   # User data
            &output_data_packet_size,         # In/Out: packet count
            output_buffer_list,               # Output buffer
            NULL                              # Packet descriptions (optional)
        )

        if status != 0:
            raise RuntimeError(f"AudioConverterFillComplexBuffer failed with status: {status}")

        # Extract result
        actual_bytes = output_buffer_list.mBuffers[0].mDataByteSize
        result = output_buffer[:actual_bytes]

        return (result, output_data_packet_size)

    finally:
        free(output_buffer)
        free(output_buffer_list)


# AudioConverter property getters
def get_audio_converter_property_min_input_buffer_size() -> int:
    """Get the minimum input buffer size for an audio converter"""
    return at.kAudioConverterPropertyMinimumInputBufferSize

def get_audio_converter_property_min_output_buffer_size() -> int:
    """Get the minimum output buffer size for an audio converter"""
    return at.kAudioConverterPropertyMinimumOutputBufferSize

def get_audio_converter_property_max_input_packet_size() -> int:
    """Get the maximum input packet size for an audio converter"""
    return at.kAudioConverterPropertyMaximumInputPacketSize

def get_audio_converter_property_max_output_packet_size() -> int:
    """Get the maximum output packet size for an audio converter"""
    return at.kAudioConverterPropertyMaximumOutputPacketSize

def get_audio_converter_property_sample_rate_converter_quality() -> int:
    """Get the sample rate converter quality for an audio converter"""
    return at.kAudioConverterSampleRateConverterQuality

def get_audio_converter_property_codec_quality() -> int:
    """Get the codec quality for an audio converter"""
    return at.kAudioConverterCodecQuality

def get_audio_converter_quality_max() -> int:
    """Get the maximum quality for an audio converter"""
    return at.kAudioConverterQuality_Max

def get_audio_converter_quality_high() -> int:
    """Get the high quality for an audio converter"""
    return at.kAudioConverterQuality_High

def get_audio_converter_quality_medium() -> int:
    """Get the medium quality for an audio converter"""
    return at.kAudioConverterQuality_Medium

def get_audio_converter_quality_low() -> int:
    """Get the low quality for an audio converter"""
    return at.kAudioConverterQuality_Low

def get_audio_converter_quality_min() -> int:
    """Get the minimum quality for an audio converter"""
    return at.kAudioConverterQuality_Min


# ============================================================================
# ExtendedAudioFile Functions
# ============================================================================

def extended_audio_file_open_url(str file_path) -> int:
    """Open an audio file with ExtendedAudioFile

    Args:
        file_path: Path to audio file

    Returns:
        ExtAudioFile ID
    """
    cdef cf.CFStringRef cf_path = cf.CFStringCreateWithCString(NULL, file_path.encode('utf-8'), cf.kCFStringEncodingUTF8)
    cdef cf.CFURLRef url = cf.CFURLCreateWithFileSystemPath(NULL, cf_path, cf.kCFURLPOSIXPathStyle, False)
    cdef at.ExtAudioFileRef ext_file

    if url == NULL:
        cf.CFRelease(cf_path)
        raise ValueError(f"Failed to create URL from path: {file_path}")

    cdef cf.OSStatus status = at.ExtAudioFileOpenURL(url, &ext_file)

    cf.CFRelease(url)
    cf.CFRelease(cf_path)

    if status != 0:
        raise RuntimeError(f"ExtAudioFileOpenURL failed with status: {status}")

    return <long>ext_file


def extended_audio_file_create_with_url(str file_path, int file_type, source_format: dict,
                                         int codec_manufacturer=0) -> int:
    """Create a new audio file with ExtendedAudioFile

    Args:
        file_path: Path for new audio file
        file_type: Audio file type (e.g., kAudioFileWAVEType)
        source_format: Audio format dictionary (AudioStreamBasicDescription)
        codec_manufacturer: Codec manufacturer code (default: 0)

    Returns:
        ExtAudioFile ID
    """
    cdef cf.CFStringRef cf_path = cf.CFStringCreateWithCString(NULL, file_path.encode('utf-8'), cf.kCFStringEncodingUTF8)
    cdef cf.CFURLRef url = cf.CFURLCreateWithFileSystemPath(NULL, cf_path, cf.kCFURLPOSIXPathStyle, False)
    cdef at.AudioStreamBasicDescription format
    cdef at.ExtAudioFileRef ext_file

    if url == NULL:
        cf.CFRelease(cf_path)
        raise ValueError(f"Failed to create URL from path: {file_path}")

    # Set up audio format
    format.mSampleRate = source_format.get('sample_rate', 44100.0)
    format.mFormatID = source_format.get('format_id', ca.kAudioFormatLinearPCM)
    format.mFormatFlags = source_format.get('format_flags',
        ca.kLinearPCMFormatFlagIsSignedInteger | ca.kLinearPCMFormatFlagIsPacked)
    format.mBytesPerPacket = source_format.get('bytes_per_packet', 4)
    format.mFramesPerPacket = source_format.get('frames_per_packet', 1)
    format.mBytesPerFrame = source_format.get('bytes_per_frame', 4)
    format.mChannelsPerFrame = source_format.get('channels_per_frame', 2)
    format.mBitsPerChannel = source_format.get('bits_per_channel', 16)
    format.mReserved = 0

    cdef cf.OSStatus status = at.ExtAudioFileCreateWithURL(
        url,
        file_type,
        &format,
        NULL,  # channel layout
        at.kAudioFileFlags_EraseFile,
        &ext_file
    )

    cf.CFRelease(url)
    cf.CFRelease(cf_path)

    if status != 0:
        raise RuntimeError(f"ExtAudioFileCreateWithURL failed with status: {status}")

    return <long>ext_file


def extended_audio_file_dispose(long ext_file_id):
    """Dispose of an ExtendedAudioFile"""
    cdef at.ExtAudioFileRef ext_file = <at.ExtAudioFileRef>ext_file_id
    cdef cf.OSStatus status = at.ExtAudioFileDispose(ext_file)
    if status != 0:
        raise RuntimeError(f"ExtAudioFileDispose failed with status: {status}")


def extended_audio_file_read(long ext_file_id, int num_frames) -> tuple:
    """Read audio frames from ExtendedAudioFile

    Args:
        ext_file_id: ExtAudioFile ID
        num_frames: Number of frames to read

    Returns:
        Tuple of (audio_data_bytes, frames_read)
    """
    cdef at.ExtAudioFileRef ext_file = <at.ExtAudioFileRef>ext_file_id
    cdef cf.UInt32 frames_to_read = num_frames
    cdef cf.UInt32 buffer_size = num_frames * 4 * 2  # Assume stereo 16-bit for now
    cdef char* buffer = <char*>malloc(buffer_size)
    cdef at.AudioBufferList buffer_list
    cdef cf.OSStatus status

    if buffer == <char*>0:
        raise MemoryError("Failed to allocate read buffer")

    try:
        buffer_list.mNumberBuffers = 1
        buffer_list.mBuffers[0].mNumberChannels = 2
        buffer_list.mBuffers[0].mDataByteSize = buffer_size
        buffer_list.mBuffers[0].mData = buffer

        status = at.ExtAudioFileRead(ext_file, &frames_to_read, &buffer_list)
        if status != 0:
            raise RuntimeError(f"ExtAudioFileRead failed with status: {status}")

        actual_size = buffer_list.mBuffers[0].mDataByteSize
        result = buffer[:actual_size]
        return (result, frames_to_read)
    finally:
        free(buffer)


def extended_audio_file_write(long ext_file_id, int num_frames, bytes audio_data):
    """Write audio frames to ExtendedAudioFile

    Args:
        ext_file_id: ExtAudioFile ID
        num_frames: Number of frames to write
        audio_data: Audio data bytes
    """
    cdef at.ExtAudioFileRef ext_file = <at.ExtAudioFileRef>ext_file_id
    cdef cf.UInt32 frames = num_frames
    cdef char* buffer = <char*><bytes>audio_data
    cdef cf.UInt32 buffer_size = len(audio_data)

    cdef at.AudioBufferList buffer_list
    buffer_list.mNumberBuffers = 1
    buffer_list.mBuffers[0].mNumberChannels = 2
    buffer_list.mBuffers[0].mDataByteSize = buffer_size
    buffer_list.mBuffers[0].mData = buffer

    cdef cf.OSStatus status = at.ExtAudioFileWrite(ext_file, frames, &buffer_list)
    if status != 0:
        raise RuntimeError(f"ExtAudioFileWrite failed with status: {status}")


def extended_audio_file_get_property(long ext_file_id, int property_id) -> bytes:
    """Get a property from an ExtendedAudioFile

    Args:
        ext_file_id: ExtAudioFile ID
        property_id: Property ID

    Returns:
        Property data as bytes
    """
    cdef at.ExtAudioFileRef ext_file = <at.ExtAudioFileRef>ext_file_id
    cdef cf.UInt32 data_size = sizeof(at.AudioStreamBasicDescription)  # Start with a reasonable size
    cdef char* buffer = <char*>malloc(data_size)
    cdef cf.OSStatus status

    if buffer == <char*>0:
        raise MemoryError("Failed to allocate buffer")

    try:
        status = at.ExtAudioFileGetProperty(ext_file, property_id, &data_size, buffer)
        if status != 0:
            raise RuntimeError(f"ExtAudioFileGetProperty failed with status: {status}")

        result = buffer[:data_size]
        return result
    finally:
        free(buffer)


def extended_audio_file_set_property(long ext_file_id, int property_id, bytes data):
    """Set a property on an ExtendedAudioFile

    Args:
        ext_file_id: ExtAudioFile ID
        property_id: Property ID
        data: Property data as bytes
    """
    cdef at.ExtAudioFileRef ext_file = <at.ExtAudioFileRef>ext_file_id
    cdef cf.UInt32 data_size = len(data)
    cdef char* buffer = <char*><bytes>data

    cdef cf.OSStatus status = at.ExtAudioFileSetProperty(ext_file, property_id, data_size, buffer)
    if status != 0:
        raise RuntimeError(f"ExtAudioFileSetProperty failed with status: {status}")


# ExtendedAudioFile property getters
def get_extended_audio_file_property_file_data_format() -> int:
    """Get the file data format for an extended audio file"""
    return at.kExtAudioFileProperty_FileDataFormat

def get_extended_audio_file_property_client_data_format() -> int:
    """Get the client data format for an extended audio file"""
    return at.kExtAudioFileProperty_ClientDataFormat

def get_extended_audio_file_property_file_channel_layout() -> int:
    """Get the file channel layout for an extended audio file"""
    return at.kExtAudioFileProperty_FileChannelLayout

def get_extended_audio_file_property_client_channel_layout() -> int:
    """Get the client channel layout for an extended audio file"""
    return at.kExtAudioFileProperty_ClientChannelLayout

def get_extended_audio_file_property_codec_manufacturer() -> int:
    """Get the codec manufacturer for an extended audio file"""
    return at.kExtAudioFileProperty_CodecManufacturer

def get_extended_audio_file_property_audio_file() -> int:
    """Get the audio file for an extended audio file"""
    return at.kExtAudioFileProperty_AudioFile

def get_extended_audio_file_property_file_length_frames() -> int:
    """Get the file length frames for an extended audio file"""
    return at.kExtAudioFileProperty_FileLengthFrames


# Audio Queue Functions
cdef void audio_queue_output_callback(void* user_data, at.AudioQueueRef queue, at.AudioQueueBufferRef buffer) noexcept:
    """C callback function for audio queue output"""
    # This will be called by CoreAudio when it needs more audio data
    # For now, we'll just enqueue the buffer again to keep playing
    cdef cf.OSStatus status = at.AudioQueueEnqueueBuffer(queue, buffer, 0, NULL)


def audio_queue_new_output(audio_format):
    """Create a new audio output queue"""
    cdef at.AudioStreamBasicDescription format
    cdef at.AudioQueueRef queue

    # Set up the audio format
    format.mSampleRate = audio_format.get('sample_rate', 44100.0)
    format.mFormatID = audio_format.get('format_id', ca.kAudioFormatLinearPCM)
    format.mFormatFlags = audio_format.get('format_flags',
        ca.kLinearPCMFormatFlagIsSignedInteger | ca.kLinearPCMFormatFlagIsPacked)
    format.mBytesPerPacket = audio_format.get('bytes_per_packet', 4)
    format.mFramesPerPacket = audio_format.get('frames_per_packet', 1)
    format.mBytesPerFrame = audio_format.get('bytes_per_frame', 4)
    format.mChannelsPerFrame = audio_format.get('channels_per_frame', 2)
    format.mBitsPerChannel = audio_format.get('bits_per_channel', 16)
    format.mReserved = 0

    cdef cf.OSStatus status = at.AudioQueueNewOutput(
        &format,
        audio_queue_output_callback,
        NULL,  # user data
        NULL,  # run loop
        NULL,  # run loop mode
        0,     # flags
        &queue
    )

    if status != 0:
        raise RuntimeError(f"AudioQueueNewOutput failed with status: {status}")

    return <long>queue


def audio_queue_allocate_buffer(long queue_id, int buffer_size):
    """Allocate a buffer for an audio queue"""
    cdef at.AudioQueueRef queue = <at.AudioQueueRef>queue_id
    cdef at.AudioQueueBufferRef buffer

    cdef cf.OSStatus status = at.AudioQueueAllocateBuffer(
        queue,
        <cf.UInt32>buffer_size,
        &buffer
    )

    if status != 0:
        raise RuntimeError(f"AudioQueueAllocateBuffer failed with status: {status}")

    return <long>buffer


def audio_queue_enqueue_buffer(long queue_id, long buffer_id):
    """Enqueue a buffer to an audio queue"""
    cdef at.AudioQueueRef queue = <at.AudioQueueRef>queue_id
    cdef at.AudioQueueBufferRef buffer = <at.AudioQueueBufferRef>buffer_id

    cdef cf.OSStatus status = at.AudioQueueEnqueueBuffer(queue, buffer, 0, NULL)

    if status != 0:
        raise RuntimeError(f"AudioQueueEnqueueBuffer failed with status: {status}")

    return status


def audio_queue_start(long queue_id):
    """Start an audio queue"""
    cdef at.AudioQueueRef queue = <at.AudioQueueRef>queue_id

    cdef cf.OSStatus status = at.AudioQueueStart(queue, NULL)

    if status != 0:
        raise RuntimeError(f"AudioQueueStart failed with status: {status}")

    return status


def audio_queue_stop(long queue_id, bint immediate=True):
    """Stop an audio queue"""
    cdef at.AudioQueueRef queue = <at.AudioQueueRef>queue_id

    cdef cf.OSStatus status = at.AudioQueueStop(queue, immediate)

    if status != 0:
        raise RuntimeError(f"AudioQueueStop failed with status: {status}")

    return status


def audio_queue_dispose(long queue_id, bint immediate=True):
    """Dispose of an audio queue"""
    cdef at.AudioQueueRef queue = <at.AudioQueueRef>queue_id

    cdef cf.OSStatus status = at.AudioQueueDispose(queue, immediate)

    if status != 0:
        raise RuntimeError(f"AudioQueueDispose failed with status: {status}")

    return status


# Constant getter functions
def get_audio_format_linear_pcm():
    """Get the linear PCM format for an audio format"""
    return ca.kAudioFormatLinearPCM

def get_linear_pcm_format_flag_is_signed_integer():
    """Get the signed integer flag for a linear PCM format"""
    return ca.kLinearPCMFormatFlagIsSignedInteger

def get_linear_pcm_format_flag_is_packed():
    """Get the packed flag for a linear PCM format"""
    return ca.kLinearPCMFormatFlagIsPacked

def get_audio_file_wave_type():
    """Get the wave type for an audio file"""
    return at.kAudioFileWAVEType

def get_audio_file_read_permission():
    """Get the read permission for an audio file"""
    return at.kAudioFileReadPermission

def get_audio_file_property_data_format():
    """Get the data format for an audio file"""
    return at.kAudioFilePropertyDataFormat

def get_audio_file_property_maximum_packet_size():
    """Get the maximum packet size for an audio file"""
    return at.kAudioFilePropertyMaximumPacketSize

def get_audio_file_property_audio_data_packet_count():
    """Get the audio data packet count for an audio file"""
    return at.kAudioFilePropertyAudioDataPacketCount

def get_audio_file_property_estimated_duration():
    """Get the estimated duration for an audio file"""
    return at.kAudioFilePropertyEstimatedDuration

# AudioObject/Device property constants
def get_audio_object_property_name():
    """Get the name for an audio object"""
    return ca.kAudioObjectPropertyName

def get_audio_object_property_manufacturer():
    """Get the manufacturer for an audio object"""
    return ca.kAudioObjectPropertyManufacturer

def get_audio_device_property_device_uid():
    """Get the device UID for an audio device"""
    return ca.kAudioDevicePropertyDeviceUID

def get_audio_device_property_model_uid():
    """Get the model UID for an audio device"""
    return ca.kAudioDevicePropertyModelUID

def get_audio_device_property_transport_type():
    """Get the transport type for an audio device"""
    return ca.kAudioDevicePropertyTransportType

def get_audio_device_property_device_is_alive():
    """Get the device is alive for an audio device"""
    return ca.kAudioDevicePropertyDeviceIsAlive

def get_audio_device_property_nominal_sample_rate():
    """Get the nominal sample rate for an audio device"""
    return ca.kAudioDevicePropertyNominalSampleRate

def get_audio_device_property_available_nominal_sample_rates():
    """Get the available nominal sample rates for an audio device"""
    return ca.kAudioDevicePropertyAvailableNominalSampleRates

def get_audio_device_property_is_hidden():
    """Get the is hidden for an audio device"""
    return ca.kAudioDevicePropertyIsHidden

def get_audio_device_property_preferred_channels_for_stereo():
    """Get the preferred channels for stereo for an audio device"""
    return ca.kAudioDevicePropertyPreferredChannelsForStereo

def get_audio_device_property_stream_configuration():
    """Get the stream configuration for an audio device"""
    return ca.kAudioDevicePropertyStreamConfiguration

# AudioObject scope constants
def get_audio_object_property_scope_global():
    """Get the global scope for an audio object"""
    return ca.kAudioObjectPropertyScopeGlobal

def get_audio_object_property_scope_input():
    """Get the input scope for an audio object"""
    return ca.kAudioObjectPropertyScopeInput

def get_audio_object_property_scope_output():
    """Get the output scope for an audio object"""
    return ca.kAudioObjectPropertyScopeOutput

def get_audio_object_property_element_main():
    """Get the main element for an audio object"""
    return ca.kAudioObjectPropertyElementMain


# AudioComponent Functions
def audio_component_find_next(description_dict, long previous_component_id=0):
    """Find an audio component matching the description

    Args:
        description_dict: Dictionary with keys 'type', 'subtype', 'manufacturer', 'flags', 'flags_mask'
        previous_component_id: Optional component ID to continue searching from (for iteration)

    Returns:
        Component ID (long) or None if no more components found
    """
    cdef at.AudioComponentDescription desc
    cdef at.AudioComponent component
    cdef at.AudioComponent previous = NULL

    if previous_component_id != 0:
        previous = <at.AudioComponent>previous_component_id

    desc.componentType = description_dict.get('type', 0)
    desc.componentSubType = description_dict.get('subtype', 0)
    desc.componentManufacturer = description_dict.get('manufacturer', 0)
    desc.componentFlags = description_dict.get('flags', 0)
    desc.componentFlagsMask = description_dict.get('flags_mask', 0)

    component = at.AudioComponentFindNext(previous, &desc)

    if component == NULL:
        return None
    return <long>component


def audio_component_instance_new(long component_id):
    """Create a new instance of an audio component"""
    cdef at.AudioComponent component = <at.AudioComponent>component_id
    cdef at.AudioComponentInstance instance

    cdef cf.OSStatus status = at.AudioComponentInstanceNew(component, &instance)
    if status != 0:
        raise RuntimeError(f"AudioComponentInstanceNew failed with status: {status}")

    return <long>instance


def audio_component_instance_dispose(long instance_id):
    """Dispose of an audio component instance"""
    cdef at.AudioComponentInstance instance = <at.AudioComponentInstance>instance_id

    cdef cf.OSStatus status = at.AudioComponentInstanceDispose(instance)
    if status != 0:
        raise RuntimeError(f"AudioComponentInstanceDispose failed with status: {status}")

    return status


def audio_component_copy_name(long component_id):
    """Get the name of an audio component

    Args:
        component_id: AudioComponent ID

    Returns:
        str: The name of the audio component, or None if failed
    """
    cdef at.AudioComponent component = <at.AudioComponent>component_id
    cdef cf.CFStringRef name_ref = NULL

    cdef cf.OSStatus status = at.AudioComponentCopyName(component, &name_ref)
    if status != 0 or name_ref == NULL:
        return None

    # Convert CFString to Python string
    cdef char buffer[256]
    cdef str name = None

    if cf.CFStringGetCString(name_ref, buffer, 256, cf.kCFStringEncodingUTF8):
        name = buffer.decode('utf-8')

    cf.CFRelease(name_ref)
    return name


def audio_component_get_description(long component_id):
    """Get the description of an audio component

    Args:
        component_id: AudioComponent ID

    Returns:
        dict: AudioComponentDescription as a dictionary with keys:
              'type', 'subtype', 'manufacturer', 'flags', 'flags_mask'
    """
    cdef at.AudioComponent component = <at.AudioComponent>component_id
    cdef at.AudioComponentDescription desc

    cdef cf.OSStatus status = at.AudioComponentGetDescription(component, &desc)
    if status != 0:
        raise RuntimeError(f"AudioComponentGetDescription failed with status: {status}")

    return {
        'type': desc.componentType,
        'subtype': desc.componentSubType,
        'manufacturer': desc.componentManufacturer,
        'flags': desc.componentFlags,
        'flags_mask': desc.componentFlagsMask
    }


# AudioUnit Functions
def audio_unit_initialize(long audio_unit_id):
    """Initialize an audio unit"""
    cdef at.AudioUnit unit = <at.AudioUnit>audio_unit_id

    cdef cf.OSStatus status = at.AudioUnitInitialize(unit)
    if status != 0:
        raise RuntimeError(f"AudioUnitInitialize failed with status: {status}")

    return status


def audio_unit_uninitialize(long audio_unit_id):
    """Uninitialize an audio unit"""
    cdef at.AudioUnit unit = <at.AudioUnit>audio_unit_id

    cdef cf.OSStatus status = at.AudioUnitUninitialize(unit)
    if status != 0:
        raise RuntimeError(f"AudioUnitUninitialize failed with status: {status}")

    return status


def audio_unit_set_property(long audio_unit_id, int property_id, int scope, int element, data):
    """Set a property on an audio unit"""
    cdef at.AudioUnit unit = <at.AudioUnit>audio_unit_id
    cdef cf.OSStatus status

    if isinstance(data, bytes):
        # Handle raw bytes data
        status = at.AudioUnitSetProperty(unit,
                                         <at.AudioUnitPropertyID>property_id,
                                         <at.AudioUnitScope>scope,
                                         <at.AudioUnitElement>element,
                                         <const char*>data,
                                         <cf.UInt32>len(data))
    else:
        raise ValueError("data must be bytes")

    if status != 0:
        raise RuntimeError(f"AudioUnitSetProperty failed with status: {status}")

    return status


def audio_unit_get_property(long audio_unit_id, int property_id, int scope, int element):
    """Get a property from an audio unit"""
    cdef at.AudioUnit unit = <at.AudioUnit>audio_unit_id
    cdef cf.UInt32 data_size = 0
    cdef cf.Boolean writable = 0
    cdef cf.OSStatus status

    # Get the size of the property
    status = at.AudioUnitGetPropertyInfo(unit,
                                         <at.AudioUnitPropertyID>property_id,
                                         <at.AudioUnitScope>scope,
                                         <at.AudioUnitElement>element,
                                         &data_size,
                                         &writable)
    if status != 0:
        raise RuntimeError(f"AudioUnitGetPropertyInfo failed with status: {status}")

    # Allocate buffer and get the property
    cdef char* buffer = <char*>malloc(data_size)
    if not buffer:
        raise MemoryError("Could not allocate buffer for property data")

    try:
        status = at.AudioUnitGetProperty(unit,
                                         <at.AudioUnitPropertyID>property_id,
                                         <at.AudioUnitScope>scope,
                                         <at.AudioUnitElement>element,
                                         buffer,
                                         &data_size)

        if status != 0:
            raise RuntimeError(f"AudioUnitGetProperty failed with status: {status}")

        return buffer[:data_size]

    finally:
        free(buffer)


def audio_output_unit_start(long audio_unit_id):
    """Start an output audio unit"""
    cdef at.AudioUnit unit = <at.AudioUnit>audio_unit_id

    cdef cf.OSStatus status = at.AudioOutputUnitStart(unit)
    if status != 0:
        raise RuntimeError(f"AudioOutputUnitStart failed with status: {status}")

    return status


def audio_output_unit_stop(long audio_unit_id):
    """Stop an output audio unit"""
    cdef at.AudioUnit unit = <at.AudioUnit>audio_unit_id

    cdef cf.OSStatus status = at.AudioOutputUnitStop(unit)
    if status != 0:
        raise RuntimeError(f"AudioOutputUnitStop failed with status: {status}")

    return status


# AudioUnit constant getter functions
def get_audio_unit_type_output():
    """Get the output type for an audio unit"""
    return at.kAudioUnitType_Output

def get_audio_component_type_music_device():
    """Get the music device type for an audio unit"""
    return at.kAudioUnitType_MusicDevice

def get_audio_unit_subtype_default_output():
    """Get the default output type for an audio unit"""
    return at.kAudioUnitSubType_DefaultOutput

def get_audio_unit_manufacturer_apple():
    """Get the Apple manufacturer for an audio unit"""
    return at.kAudioUnitManufacturer_Apple

def get_audio_unit_property_stream_format():
    """Get the stream format for an audio unit"""
    return at.kAudioUnitProperty_StreamFormat

def get_audio_unit_property_set_render_callback():
    """Get the set render callback for an audio unit"""
    return at.kAudioUnitProperty_SetRenderCallback

def get_audio_unit_scope_input():
    """Get the input scope for an audio unit"""
    return at.kAudioUnitScope_Input

def get_audio_unit_scope_output():
    """Get the output scope for an audio unit"""
    return at.kAudioUnitScope_Output

def get_audio_unit_scope_global():
    """Get the global scope for an audio unit"""
    return at.kAudioUnitScope_Global

def get_linear_pcm_format_flag_is_non_interleaved():
    """Get the non-interleaved flag for a linear PCM format"""
    return ca.kLinearPCMFormatFlagIsNonInterleaved


# ===== PURE CYTHON AUDIO PLAYER IMPLEMENTATION =====

# Constants for audio format
DEF SAMPLE_RATE = 44100.0
DEF CHANNELS = 2
DEF BITS_PER_CHANNEL = 32  # Float32

# Audio player state structure
cdef struct AudioPlayerData:
    ca.AudioBufferList* buffer_list
    cf.UInt32 total_frames
    cf.UInt32 current_frame
    cf.Boolean playing
    cf.Boolean loop

# Render callback function
cdef cf.OSStatus audio_player_render_callback(
    void* user_data,
    at.AudioUnitRenderActionFlags* action_flags,
    const ca.AudioTimeStamp* time_stamp,
    cf.UInt32 bus_number,
    cf.UInt32 num_frames,
    ca.AudioBufferList* io_data
) noexcept nogil:
    """Render callback that provides audio data to the output unit"""
    cdef AudioPlayerData* player_data
    cdef cf.UInt32 current_frame
    cdef cf.UInt32 max_frames
    cdef cf.Float32* output_data
    cdef cf.Float32* input_data
    cdef cf.UInt32 frame, out_sample, in_sample
    cdef cf.UInt32 buffer_idx

    if user_data == NULL or io_data == NULL:
        return 0  # noErr

    player_data = <AudioPlayerData*>user_data

    # Clear output buffers first
    for buffer_idx in range(io_data.mNumberBuffers):
        memset(io_data.mBuffers[buffer_idx].mData, 0, io_data.mBuffers[buffer_idx].mDataByteSize)

    # Check if we have data and are playing
    if not player_data.playing or player_data.buffer_list == NULL or player_data.total_frames == 0:
        return 0  # Return silence

    current_frame = player_data.current_frame
    max_frames = player_data.total_frames

    output_data = <cf.Float32*>io_data.mBuffers[0].mData
    input_data = <cf.Float32*>player_data.buffer_list.mBuffers[0].mData

    # Copy audio data frame by frame
    for frame in range(num_frames):
        if current_frame >= max_frames:
            if player_data.loop:
                current_frame = 0  # Loop back to start
            else:
                player_data.playing = False
                break  # Stop playing

        # Copy stereo frame (2 channels)
        out_sample = frame * 2
        in_sample = current_frame * 2

        if current_frame < max_frames:
            output_data[out_sample] = input_data[in_sample]          # Left channel
            output_data[out_sample + 1] = input_data[in_sample + 1]  # Right channel

        current_frame += 1

    player_data.current_frame = current_frame

    return 0  # noErr


cdef class AudioPlayer:
    """Pure Cython audio player using AudioUnit for playback"""
    cdef AudioPlayerData player_data
    cdef at.AudioUnit output_unit
    cdef bint initialized

    def __init__(self):
        """Initialize the AudioPlayer"""
        memset(&self.player_data, 0, sizeof(AudioPlayerData))
        self.output_unit = NULL
        self.initialized = False

    def load_file(self, str file_path):
        """Load an audio file for playback using ExtAudioFile"""
        # Declare all C variables at the beginning
        cdef bytes path_bytes = file_path.encode('utf-8')
        cdef cf.CFURLRef url_ref
        cdef at.ExtAudioFileRef audio_file
        cdef cf.OSStatus status
        cdef at.AudioStreamBasicDescription file_format
        cdef at.AudioStreamBasicDescription client_format
        cdef cf.UInt32 size
        cdef cf.SInt64 file_length_frames
        cdef cf.UInt32 bytes_per_buffer
        cdef ca.AudioBufferList* buffer_list
        cdef cf.UInt32 read_frames = 0
        cdef cf.UInt32 frames_to_read
        cdef ca.AudioBufferList temp_buffer_list

        url_ref = cf.CFURLCreateFromFileSystemRepresentation(
            cf.kCFAllocatorDefault,
            <const cf.UInt8*>path_bytes,
            len(path_bytes),
            False
        )

        if url_ref == NULL:
            raise ValueError(f"Could not create URL for file: {file_path}")

        # Open audio file
        status = at.ExtAudioFileOpenURL(url_ref, &audio_file)
        cf.CFRelease(url_ref)

        if status != 0:
            raise RuntimeError(f"Could not open audio file: {status}")

        try:
            # Get file format
            size = sizeof(file_format)
            status = at.ExtAudioFileGetProperty(
                audio_file,
                at.kExtAudioFileProperty_FileDataFormat,
                &size,
                &file_format
            )
            if status != 0:
                raise RuntimeError(f"Could not get file format: {status}")

            # Set client format (what we want the data converted to)
            client_format.mSampleRate = SAMPLE_RATE
            client_format.mFormatID = ca.kAudioFormatLinearPCM
            client_format.mFormatFlags = ca.kLinearPCMFormatFlagIsFloat
            client_format.mBytesPerPacket = sizeof(cf.Float32) * CHANNELS
            client_format.mFramesPerPacket = 1
            client_format.mBytesPerFrame = CHANNELS * sizeof(cf.Float32)
            client_format.mChannelsPerFrame = CHANNELS
            client_format.mBitsPerChannel = BITS_PER_CHANNEL
            client_format.mReserved = 0

            status = at.ExtAudioFileSetProperty(
                audio_file,
                at.kExtAudioFileProperty_ClientDataFormat,
                sizeof(client_format),
                &client_format
            )
            if status != 0:
                raise RuntimeError(f"Could not set client format: {status}")

            # Get file length in frames
            size = sizeof(file_length_frames)
            status = at.ExtAudioFileGetProperty(
                audio_file,
                at.kExtAudioFileProperty_FileLengthFrames,
                &size,
                &file_length_frames
            )
            if status != 0:
                raise RuntimeError(f"Could not get file length: {status}")

            # Account for sample rate conversion
            file_length_frames = <cf.SInt64>(file_length_frames * (SAMPLE_RATE / file_format.mSampleRate))

            # Allocate buffer for audio data
            bytes_per_buffer = client_format.mBytesPerFrame * <cf.UInt32>file_length_frames

            # Allocate AudioBufferList
            buffer_list = <ca.AudioBufferList*>malloc(
                sizeof(ca.AudioBufferList) + 0 * sizeof(ca.AudioBuffer)  # Single buffer
            )
            if buffer_list == NULL:
                raise MemoryError("Could not allocate AudioBufferList")

            buffer_list.mNumberBuffers = 1
            buffer_list.mBuffers[0].mData = malloc(bytes_per_buffer)
            if buffer_list.mBuffers[0].mData == NULL:
                free(buffer_list)
                raise MemoryError("Could not allocate audio buffer")

            memset(buffer_list.mBuffers[0].mData, 0, bytes_per_buffer)
            buffer_list.mBuffers[0].mDataByteSize = bytes_per_buffer
            buffer_list.mBuffers[0].mNumberChannels = client_format.mChannelsPerFrame

            # Read audio data in chunks
            read_frames = 0
            while read_frames < <cf.UInt32>file_length_frames:
                frames_to_read = <cf.UInt32>file_length_frames - read_frames
                if frames_to_read > 16384:
                    frames_to_read = 16384  # Read in chunks

                # Setup temporary buffer list for this chunk
                temp_buffer_list.mNumberBuffers = 1
                temp_buffer_list.mBuffers[0].mNumberChannels = client_format.mChannelsPerFrame
                temp_buffer_list.mBuffers[0].mData = <char*>buffer_list.mBuffers[0].mData + (read_frames * client_format.mBytesPerFrame)
                temp_buffer_list.mBuffers[0].mDataByteSize = frames_to_read * client_format.mBytesPerFrame

                status = at.ExtAudioFileRead(audio_file, &frames_to_read, &temp_buffer_list)
                if frames_to_read == 0:
                    break

                read_frames += frames_to_read

            # Clean up old buffer if exists
            if self.player_data.buffer_list != NULL:
                if self.player_data.buffer_list.mBuffers[0].mData != NULL:
                    free(self.player_data.buffer_list.mBuffers[0].mData)
                free(self.player_data.buffer_list)

            # Set up player data
            self.player_data.buffer_list = buffer_list
            self.player_data.total_frames = read_frames
            self.player_data.current_frame = 0
            self.player_data.playing = False
            self.player_data.loop = False

        finally:
            at.ExtAudioFileDispose(audio_file)

        return 0  # noErr

    def setup_output(self):
        """Setup the audio output unit"""
        cdef at.AudioComponentDescription output_desc
        cdef at.AudioComponent comp
        cdef cf.OSStatus status

        # Find default output AudioComponent
        output_desc.componentType = at.kAudioUnitType_Output
        output_desc.componentSubType = at.kAudioUnitSubType_DefaultOutput
        output_desc.componentManufacturer = at.kAudioUnitManufacturer_Apple
        output_desc.componentFlags = 0
        output_desc.componentFlagsMask = 0

        comp = at.AudioComponentFindNext(NULL, &output_desc)
        if comp == NULL:
            raise RuntimeError("Cannot find default output AudioComponent")

        # Create AudioUnit instance
        status = at.AudioComponentInstanceNew(comp, &self.output_unit)
        if status != 0:
            raise RuntimeError(f"Could not create AudioUnit instance: {status}")

        # Set stream format
        cdef at.AudioStreamBasicDescription stream_format
        stream_format.mSampleRate = SAMPLE_RATE
        stream_format.mFormatID = ca.kAudioFormatLinearPCM
        stream_format.mFormatFlags = ca.kLinearPCMFormatFlagIsFloat
        stream_format.mBytesPerPacket = sizeof(cf.Float32) * CHANNELS
        stream_format.mFramesPerPacket = 1
        stream_format.mBytesPerFrame = CHANNELS * sizeof(cf.Float32)
        stream_format.mChannelsPerFrame = CHANNELS
        stream_format.mBitsPerChannel = BITS_PER_CHANNEL
        stream_format.mReserved = 0

        status = at.AudioUnitSetProperty(
            self.output_unit,
            at.kAudioUnitProperty_StreamFormat,
            at.kAudioUnitScope_Input,
            0,
            &stream_format,
            sizeof(stream_format)
        )
        if status != 0:
            raise RuntimeError(f"Could not set stream format: {status}")

        # Set render callback
        cdef at.AURenderCallbackStruct callback_struct
        callback_struct.inputProc = audio_player_render_callback
        callback_struct.inputProcRefCon = &self.player_data

        status = at.AudioUnitSetProperty(
            self.output_unit,
            at.kAudioUnitProperty_SetRenderCallback,
            at.kAudioUnitScope_Global,
            0,
            &callback_struct,
            sizeof(callback_struct)
        )
        if status != 0:
            raise RuntimeError(f"Could not set render callback: {status}")

        # Initialize AudioUnit
        status = at.AudioUnitInitialize(self.output_unit)
        if status != 0:
            raise RuntimeError(f"Could not initialize AudioUnit: {status}")

        self.initialized = True
        return 0  # noErr

    def start(self):
        """Start audio playback"""
        if not self.initialized:
            raise RuntimeError("AudioPlayer not initialized. Call setup_output() first.")

        self.player_data.playing = True
        self.player_data.current_frame = 0

        cdef cf.OSStatus status = at.AudioOutputUnitStart(self.output_unit)
        if status != 0:
            self.player_data.playing = False
            raise RuntimeError(f"Could not start AudioUnit: {status}")

        return 0  # noErr

    def stop(self):
        """Stop audio playback"""
        if not self.initialized:
            return 0  # noErr

        self.player_data.playing = False

        cdef cf.OSStatus status = at.AudioOutputUnitStop(self.output_unit)
        if status != 0:
            raise RuntimeError(f"Could not stop AudioUnit: {status}")

        return 0  # noErr

    def set_looping(self, bint loop):
        """Enable/disable looping playback"""
        self.player_data.loop = loop

    def reset_playback(self):
        """Reset playback to beginning"""
        self.player_data.current_frame = 0

    def is_playing(self):
        """Check if audio is currently playing"""
        return bool(self.player_data.playing)

    def get_progress(self):
        """Get current playback progress as a float (0.0 to 1.0)"""
        if self.player_data.total_frames == 0:
            return 0.0
        return <float>self.player_data.current_frame / <float>self.player_data.total_frames

    def __dealloc__(self):
        """Clean up resources when the object is destroyed"""
        if self.initialized and self.output_unit != NULL:
            at.AudioOutputUnitStop(self.output_unit)
            at.AudioUnitUninitialize(self.output_unit)
            at.AudioComponentInstanceDispose(self.output_unit)

        if self.player_data.buffer_list != NULL:
            if self.player_data.buffer_list.mBuffers[0].mData != NULL:
                free(self.player_data.buffer_list.mBuffers[0].mData)
            free(self.player_data.buffer_list)
            self.player_data.buffer_list = NULL


# AudioFileStream constant getter functions
def get_audio_file_stream_property_ready_to_produce_packets():
    """Get the ready to produce packets for an audio file stream"""
    return at.kAudioFileStreamProperty_ReadyToProducePackets

def get_audio_file_stream_property_file_format():
    """Get the file format for an audio file stream"""
    return at.kAudioFileStreamProperty_FileFormat

def get_audio_file_stream_property_data_format():
    """Get the data format for an audio file stream"""
    return at.kAudioFileStreamProperty_DataFormat

def get_audio_file_stream_property_format_list():
    """Get the format list for an audio file stream"""
    return at.kAudioFileStreamProperty_FormatList

def get_audio_file_stream_property_magic_cookie_data():
    """Get the magic cookie data for an audio file stream"""
    return at.kAudioFileStreamProperty_MagicCookieData

def get_audio_file_stream_property_audio_data_byte_count():
    """Get the audio data byte count for an audio file stream"""
    return at.kAudioFileStreamProperty_AudioDataByteCount

def get_audio_file_stream_property_audio_data_packet_count():
    """Get the audio data packet count for an audio file stream"""
    return at.kAudioFileStreamProperty_AudioDataPacketCount

def get_audio_file_stream_property_maximum_packet_size():
    """Get the maximum packet size for an audio file stream"""
    return at.kAudioFileStreamProperty_MaximumPacketSize

def get_audio_file_stream_property_data_offset():
    """Get the data offset for an audio file stream"""
    return at.kAudioFileStreamProperty_DataOffset

def get_audio_file_stream_property_channel_layout():
    """Get the channel layout for an audio file stream"""
    return at.kAudioFileStreamProperty_ChannelLayout

def get_audio_file_stream_property_packet_to_frame():
    """Get the packet to frame for an audio file stream"""
    return at.kAudioFileStreamProperty_PacketToFrame

def get_audio_file_stream_property_frame_to_packet():
    """Get the frame to packet for an audio file stream"""
    return at.kAudioFileStreamProperty_FrameToPacket

def get_audio_file_stream_property_packet_to_byte():
    """Get the packet to byte for an audio file stream"""
    return at.kAudioFileStreamProperty_PacketToByte

def get_audio_file_stream_property_byte_to_packet():
    """Get the byte to packet for an audio file stream"""
    return at.kAudioFileStreamProperty_ByteToPacket

def get_audio_file_stream_property_packet_table_info():
    """Get the packet table info for an audio file stream"""
    return at.kAudioFileStreamProperty_PacketTableInfo

def get_audio_file_stream_property_packet_size_upper_bound():
    """Get the packet size upper bound for an audio file stream"""
    return at.kAudioFileStreamProperty_PacketSizeUpperBound

def get_audio_file_stream_property_average_bytes_per_packet():
    """Get the average bytes per packet for an audio file stream"""
    return at.kAudioFileStreamProperty_AverageBytesPerPacket

def get_audio_file_stream_property_bit_rate():
    """Get the bit rate for an audio file stream"""
    return at.kAudioFileStreamProperty_BitRate

def get_audio_file_stream_property_info_dictionary():
    """Get the info dictionary for an audio file stream"""
    return at.kAudioFileStreamProperty_InfoDictionary

# AudioFileStream flag getter functions
def get_audio_file_stream_property_flag_property_is_cached():
    """Get the property is cached flag for an audio file stream"""
    return at.kAudioFileStreamPropertyFlag_PropertyIsCached

def get_audio_file_stream_property_flag_cache_property():
    """Get the cache property flag for an audio file stream"""
    return at.kAudioFileStreamPropertyFlag_CacheProperty

def get_audio_file_stream_parse_flag_discontinuity():
    """Get the discontinuity flag for an audio file stream"""
    return at.kAudioFileStreamParseFlag_Discontinuity

def get_audio_file_stream_seek_flag_offset_is_estimated():
    """Get the offset is estimated flag for an audio file stream"""
    return at.kAudioFileStreamSeekFlag_OffsetIsEstimated

# AudioFileStream error code getter functions
def get_audio_file_stream_error_unsupported_file_type():
    """Get the unsupported file type error for an audio file stream"""
    return at.kAudioFileStreamError_UnsupportedFileType

def get_audio_file_stream_error_unsupported_data_format():
    """Get the unsupported data format error for an audio file stream"""
    return at.kAudioFileStreamError_UnsupportedDataFormat

def get_audio_file_stream_error_unsupported_property():
    """Get the unsupported property error for an audio file stream"""
    return at.kAudioFileStreamError_UnsupportedProperty

def get_audio_file_stream_error_bad_property_size():
    """Get the bad property size error for an audio file stream"""
    return at.kAudioFileStreamError_BadPropertySize

def get_audio_file_stream_error_not_optimized():
    """Get the not optimized error for an audio file stream"""
    return at.kAudioFileStreamError_NotOptimized

def get_audio_file_stream_error_invalid_packet_offset():
    """Get the invalid packet offset error for an audio file stream"""
    return at.kAudioFileStreamError_InvalidPacketOffset

def get_audio_file_stream_error_invalid_file():
    """Get the invalid file error for an audio file stream"""
    return at.kAudioFileStreamError_InvalidFile

def get_audio_file_stream_error_value_unknown():
    """Get the value unknown error for an audio file stream"""
    return at.kAudioFileStreamError_ValueUnknown

def get_audio_file_stream_error_data_unavailable():
    """Get the data unavailable error for an audio file stream"""
    return at.kAudioFileStreamError_DataUnavailable

def get_audio_file_stream_error_illegal_operation():
    """Get the illegal operation error for an audio file stream"""
    return at.kAudioFileStreamError_IllegalOperation

def get_audio_file_stream_error_unspecified_error():
    """Get the unspecified error for an audio file stream"""
    return at.kAudioFileStreamError_UnspecifiedError

def get_audio_file_stream_error_discontinuity_cant_recover():
    """Get the discontinuity cant recover error for an audio file stream"""
    return at.kAudioFileStreamError_DiscontinuityCantRecover


# AudioServices Functions
def audio_services_create_system_sound_id(str file_path):
    """Create a SystemSoundID from an audio file path"""
    cdef at.SystemSoundID sound_id
    cdef cf.CFURLRef url_ref
    cdef bytes path_bytes = file_path.encode('utf-8')

    url_ref = cf.CFURLCreateFromFileSystemRepresentation(
        cf.kCFAllocatorDefault,
        <const cf.UInt8*>path_bytes,
        len(path_bytes),
        False
    )

    if not url_ref:
        raise ValueError("Could not create URL from file path")

    cdef cf.OSStatus status = at.AudioServicesCreateSystemSoundID(
        url_ref,
        &sound_id
    )

    cf.CFRelease(url_ref)

    if status != 0:
        raise RuntimeError(f"AudioServicesCreateSystemSoundID failed with status: {status}")

    return <long>sound_id


def audio_services_dispose_system_sound_id(long sound_id):
    """Dispose a SystemSoundID"""
    cdef at.SystemSoundID system_sound_id = <at.SystemSoundID>sound_id
    cdef cf.OSStatus status = at.AudioServicesDisposeSystemSoundID(system_sound_id)

    if status != 0:
        raise RuntimeError(f"AudioServicesDisposeSystemSoundID failed with status: {status}")

    return status


def audio_services_play_system_sound(long sound_id):
    """Play a system sound (deprecated but widely used)"""
    cdef at.SystemSoundID system_sound_id = <at.SystemSoundID>sound_id
    at.AudioServicesPlaySystemSound(system_sound_id)


def audio_services_play_alert_sound(long sound_id):
    """Play an alert sound (deprecated but widely used)"""
    cdef at.SystemSoundID system_sound_id = <at.SystemSoundID>sound_id
    at.AudioServicesPlayAlertSound(system_sound_id)


def audio_services_get_property(int property_id, long specifier_value=0):
    """Get an AudioServices property"""
    cdef at.AudioServicesPropertyID prop_id = <at.AudioServicesPropertyID>property_id
    cdef cf.UInt32 data_size = 0
    cdef cf.Boolean writable = False
    cdef cf.UInt32 specifier = <cf.UInt32>specifier_value

    # Get property info first
    cdef cf.OSStatus status = at.AudioServicesGetPropertyInfo(
        prop_id,
        sizeof(cf.UInt32) if specifier_value != 0 else 0,
        &specifier if specifier_value != 0 else NULL,
        &data_size,
        &writable
    )

    if status != 0:
        raise RuntimeError(f"AudioServicesGetPropertyInfo failed with status: {status}")

    # Allocate buffer and get property data
    cdef char* buffer = <char*>malloc(data_size)
    cdef cf.UInt32 actual_size = data_size
    if not buffer:
        raise MemoryError("Could not allocate memory for property data")

    try:
        status = at.AudioServicesGetProperty(
            prop_id,
            sizeof(cf.UInt32) if specifier_value != 0 else 0,
            &specifier if specifier_value != 0 else NULL,
            &actual_size,
            buffer
        )

        if status != 0:
            raise RuntimeError(f"AudioServicesGetProperty failed with status: {status}")

        # Return property value based on size
        if data_size == 4:
            return (<cf.UInt32*>buffer)[0]
        else:
            return buffer[:actual_size]

    finally:
        free(buffer)


def audio_services_set_property(int property_id, data, long specifier_value=0):
    """Set an AudioServices property"""
    cdef at.AudioServicesPropertyID prop_id = <at.AudioServicesPropertyID>property_id
    cdef cf.UInt32 specifier = <cf.UInt32>specifier_value
    cdef cf.UInt32 uint_data
    cdef bytes byte_data
    cdef const void* data_ptr
    cdef cf.UInt32 data_size

    # Handle different data types
    if isinstance(data, int):
        uint_data = <cf.UInt32>data
        data_ptr = <const void*>&uint_data
        data_size = sizeof(cf.UInt32)
    elif isinstance(data, bytes):
        byte_data = data
        data_ptr = <const void*>byte_data
        data_size = len(byte_data)
    else:
        raise TypeError("Data must be int or bytes")

    cdef cf.OSStatus status = at.AudioServicesSetProperty(
        prop_id,
        sizeof(cf.UInt32) if specifier_value != 0 else 0,
        &specifier if specifier_value != 0 else NULL,
        data_size,
        data_ptr
    )

    if status != 0:
        raise RuntimeError(f"AudioServicesSetProperty failed with status: {status}")

    return status


# AudioServices constant getter functions
def get_audio_services_no_error():
    """Get the no error for an audio services"""
    return at.kAudioServicesNoError

def get_audio_services_unsupported_property_error():
    """Get the unsupported property error for an audio services"""
    return at.kAudioServicesUnsupportedPropertyError

def get_audio_services_bad_property_size_error():
    """Get the bad property size error for an audio services"""
    return at.kAudioServicesBadPropertySizeError

def get_audio_services_bad_specifier_size_error():
    """Get the bad specifier size error for an audio services"""
    return at.kAudioServicesBadSpecifierSizeError

def get_audio_services_system_sound_unspecified_error():
    """Get the system sound unspecified error for an audio services"""
    return at.kAudioServicesSystemSoundUnspecifiedError

def get_audio_services_system_sound_client_timed_out_error():
    """Get the system sound client timed out error for an audio services"""
    return at.kAudioServicesSystemSoundClientTimedOutError

def get_audio_services_system_sound_exceeded_maximum_duration_error():
    """Get the system sound exceeded maximum duration error for an audio services"""
    return at.kAudioServicesSystemSoundExceededMaximumDurationError

def get_system_sound_id_user_preferred_alert():
    """Get the user preferred alert for an audio services"""
    return at.kSystemSoundID_UserPreferredAlert

def get_system_sound_id_flash_screen():
    """Get the flash screen for an audio services"""
    return at.kSystemSoundID_FlashScreen

def get_system_sound_id_vibrate():
    """Get the vibrate for an audio services"""
    return at.kSystemSoundID_Vibrate

def get_user_preferred_alert():
    """Get the user preferred alert for an audio services"""
    return at.kUserPreferredAlert

def get_audio_services_property_is_ui_sound():
    """Get the is ui sound for an audio services"""
    return at.kAudioServicesPropertyIsUISound

def get_audio_services_property_complete_playback_if_app_dies():
    """Get the complete playback if app dies for an audio services"""
    return at.kAudioServicesPropertyCompletePlaybackIfAppDies


def test_error() -> int:
    """Test function to verify the module works"""
    return ca.kAudio_UnimplementedError


# ===== MusicDevice API =====

def music_device_midi_event(long unit, int status, int data1, int data2, int offset_sample_frame=0):
    """Send a MIDI channel message to a music device audio unit.

    Args:
        unit: The MusicDeviceComponent (AudioComponentInstance)
        status: MIDI status byte (includes channel and command)
        data1: First MIDI data byte (0-127)
        data2: Second MIDI data byte (0-127), or 0 if not needed
        offset_sample_frame: Sample offset for scheduling (default 0)

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If the MIDI event fails
    """
    cdef cf.OSStatus status_result = at.MusicDeviceMIDIEvent(
        <at.MusicDeviceComponent>unit,
        <cf.UInt32>status,
        <cf.UInt32>data1,
        <cf.UInt32>data2,
        <cf.UInt32>offset_sample_frame)

    if status_result != 0:
        raise RuntimeError(f"MusicDeviceMIDIEvent failed with status: {status_result}")
    return status_result

def music_device_sysex(long unit, bytes data):
    """Send a System Exclusive MIDI message to a music device audio unit.

    Args:
        unit: The MusicDeviceComponent (AudioComponentInstance)
        data: Complete MIDI SysEx message including F0 and F7 bytes

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If the SysEx message fails
    """
    cdef const cf.UInt8* data_ptr = <const cf.UInt8*><char*>data
    cdef cf.UInt32 length = len(data)

    cdef cf.OSStatus status = at.MusicDeviceSysEx(
        <at.MusicDeviceComponent>unit,
        data_ptr,
        length)

    if status != 0:
        raise RuntimeError(f"MusicDeviceSysEx failed with status: {status}")
    return status

def music_device_start_note(long unit, int instrument_id, int group_id, float pitch, float velocity, int offset_sample_frame=0, list controls=None):
    """Start a note on a music device audio unit.

    Args:
        unit: The MusicDeviceComponent (AudioComponentInstance)
        instrument_id: Instrument ID (use kMusicNoteEvent_Unused for current patch)
        group_id: Group/channel ID (0-based)
        pitch: MIDI pitch (0-127, can be fractional)
        velocity: MIDI velocity (0-127, can be fractional)
        offset_sample_frame: Sample offset for scheduling (default 0)
        controls: Optional list of (parameter_id, value) tuples for additional controls

    Returns:
        NoteInstanceID token for stopping the note

    Raises:
        RuntimeError: If starting the note fails
    """
    cdef at.NoteInstanceID note_instance_id
    cdef at.MusicDeviceNoteParams* params
    cdef cf.UInt32 arg_count = 2  # pitch + velocity
    cdef int num_controls = 0
    cdef cf.OSStatus status

    if controls:
        num_controls = len(controls)
        arg_count += num_controls

    # Allocate memory for note parameters
    cdef size_t params_size = sizeof(at.MusicDeviceNoteParams) + (num_controls - 1) * sizeof(at.NoteParamsControlValue)
    params = <at.MusicDeviceNoteParams*>malloc(params_size)
    if not params:
        raise MemoryError("Could not allocate memory for note parameters")

    try:
        params.argCount = arg_count
        params.mPitch = pitch
        params.mVelocity = velocity

        # Add control parameters if provided
        if controls:
            for i, (param_id, value) in enumerate(controls):
                params.mControls[i].mID = <at.AudioUnitParameterID>param_id
                params.mControls[i].mValue = <at.AudioUnitParameterValue>value

        status = at.MusicDeviceStartNote(
            <at.MusicDeviceComponent>unit,
            <at.MusicDeviceInstrumentID>instrument_id,
            <at.MusicDeviceGroupID>group_id,
            &note_instance_id,
            <cf.UInt32>offset_sample_frame,
            params)

        if status != 0:
            raise RuntimeError(f"MusicDeviceStartNote failed with status: {status}")

        return <long>note_instance_id

    finally:
        free(params)

def music_device_stop_note(long unit, int group_id, long note_instance_id, int offset_sample_frame=0):
    """Stop a note that was started with music_device_start_note.

    Args:
        unit: The MusicDeviceComponent (AudioComponentInstance)
        group_id: Group/channel ID that the note was started on
        note_instance_id: Token returned by music_device_start_note
        offset_sample_frame: Sample offset for scheduling (default 0)

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If stopping the note fails
    """
    cdef cf.OSStatus status = at.MusicDeviceStopNote(
        <at.MusicDeviceComponent>unit,
        <at.MusicDeviceGroupID>group_id,
        <at.NoteInstanceID>note_instance_id,
        <cf.UInt32>offset_sample_frame)

    if status != 0:
        raise RuntimeError(f"MusicDeviceStopNote failed with status: {status}")
    return status

# Convenience functions for creating standard note parameters

def create_music_device_std_note_params(float pitch, float velocity):
    """Create standard note parameters with just pitch and velocity.

    Args:
        pitch: MIDI pitch (0-127, can be fractional)
        velocity: MIDI velocity (0-127, can be fractional)

    Returns:
        Dictionary with argCount, pitch, and velocity
    """
    return {
        'argCount': 2,
        'pitch': pitch,
        'velocity': velocity
    }

def create_music_device_note_params(float pitch, float velocity, list controls):
    """Create note parameters with pitch, velocity, and additional controls.

    Args:
        pitch: MIDI pitch (0-127, can be fractional)
        velocity: MIDI velocity (0-127, can be fractional)
        controls: List of (parameter_id, value) tuples

    Returns:
        Dictionary with all note parameters
    """
    return {
        'argCount': 2 + len(controls),
        'pitch': pitch,
        'velocity': velocity,
        'controls': controls
    }

# MusicDevice constants

def get_music_note_event_use_group_instrument():
    """Get the constant for using the current patch for a group."""
    return at.kMusicNoteEvent_UseGroupInstrument

def get_music_note_event_unused():
    """Get the constant for unused instrument ID."""
    return at.kMusicNoteEvent_Unused

def get_music_device_range():
    """Get the MusicDevice selector range start."""
    return at.kMusicDeviceRange

def get_music_device_midi_event_select():
    """Get the MusicDevice MIDI event selector."""
    return at.kMusicDeviceMIDIEventSelect

def get_music_device_sysex_select():
    """Get the MusicDevice SysEx selector."""
    return at.kMusicDeviceSysExSelect

def get_music_device_start_note_select():
    """Get the MusicDevice start note selector."""
    return at.kMusicDeviceStartNoteSelect

def get_music_device_stop_note_select():
    """Get the MusicDevice stop note selector."""
    return at.kMusicDeviceStopNoteSelect

def get_music_device_midi_event_list_select():
    """Get the MusicDevice MIDI event list selector."""
    return at.kMusicDeviceMIDIEventListSelect

# Helper functions for MIDI data

def midi_note_on(int channel, int note, int velocity):
    """Create a MIDI Note On status byte.

    Args:
        channel: MIDI channel (0-15)
        note: MIDI note number (0-127)
        velocity: MIDI velocity (0-127)

    Returns:
        Tuple of (status, data1, data2)
    """
    status = 0x90 | (channel & 0x0F)  # Note On + channel
    return (status, note & 0x7F, velocity & 0x7F)

def midi_note_off(int channel, int note, int velocity=0):
    """Create a MIDI Note Off status byte.

    Args:
        channel: MIDI channel (0-15)
        note: MIDI note number (0-127)
        velocity: MIDI velocity (0-127, default 0)

    Returns:
        Tuple of (status, data1, data2)
    """
    status = 0x80 | (channel & 0x0F)  # Note Off + channel
    return (status, note & 0x7F, velocity & 0x7F)

def midi_control_change(int channel, int controller, int value):
    """Create a MIDI Control Change message.

    Args:
        channel: MIDI channel (0-15)
        controller: Controller number (0-127)
        value: Controller value (0-127)

    Returns:
        Tuple of (status, data1, data2)
    """
    status = 0xB0 | (channel & 0x0F)  # Control Change + channel
    return (status, controller & 0x7F, value & 0x7F)

def midi_program_change(int channel, int program):
    """Create a MIDI Program Change message.

    Args:
        channel: MIDI channel (0-15)
        program: Program number (0-127)

    Returns:
        Tuple of (status, data1, data2)
    """
    status = 0xC0 | (channel & 0x0F)  # Program Change + channel
    return (status, program & 0x7F, 0)

def midi_pitch_bend(int channel, int value):
    """Create a MIDI Pitch Bend message.

    Args:
        channel: MIDI channel (0-15)
        value: Pitch bend value (0-16383, 8192 = center)

    Returns:
        Tuple of (status, data1, data2)
    """
    status = 0xE0 | (channel & 0x0F)  # Pitch Bend + channel
    lsb = value & 0x7F
    msb = (value >> 7) & 0x7F
    return (status, lsb, msb)


# ===== MusicPlayer API =====

# MusicPlayer functions

def new_music_player():
    """Create a new music player.

    Returns:
        MusicPlayer handle

    Raises:
        RuntimeError: If player creation fails
    """
    cdef at.MusicPlayer player
    cdef cf.OSStatus status = at.NewMusicPlayer(&player)

    if status != 0:
        raise RuntimeError(f"NewMusicPlayer failed with status: {status}")
    return <long>player

def dispose_music_player(long player):
    """Dispose a music player.

    Args:
        player: The MusicPlayer handle to dispose

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If disposal fails
    """
    cdef cf.OSStatus status = at.DisposeMusicPlayer(<at.MusicPlayer>player)

    if status != 0:
        raise RuntimeError(f"DisposeMusicPlayer failed with status: {status}")
    return status

def music_player_set_sequence(long player, long sequence):
    """Set the sequence for the player to play.

    Args:
        player: The MusicPlayer handle
        sequence: The MusicSequence handle (or 0 for NULL)

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If setting sequence fails
    """
    cdef at.MusicSequence seq = <at.MusicSequence>sequence if sequence != 0 else NULL
    cdef cf.OSStatus status = at.MusicPlayerSetSequence(<at.MusicPlayer>player, seq)

    if status != 0:
        raise RuntimeError(f"MusicPlayerSetSequence failed with status: {status}")
    return status

def music_player_get_sequence(long player):
    """Get the sequence attached to a player.

    Args:
        player: The MusicPlayer handle

    Returns:
        MusicSequence handle

    Raises:
        RuntimeError: If getting sequence fails
    """
    cdef at.MusicSequence sequence
    cdef cf.OSStatus status = at.MusicPlayerGetSequence(<at.MusicPlayer>player, &sequence)

    if status != 0:
        raise RuntimeError(f"MusicPlayerGetSequence failed with status: {status}")
    return <long>sequence

def music_player_set_time(long player, double time):
    """Set the current time on the player.

    Args:
        player: The MusicPlayer handle
        time: The new time value in beats

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If setting time fails
    """
    cdef cf.OSStatus status = at.MusicPlayerSetTime(<at.MusicPlayer>player, <at.MusicTimeStamp>time)

    if status != 0:
        raise RuntimeError(f"MusicPlayerSetTime failed with status: {status}")
    return status

def music_player_get_time(long player):
    """Get the current time of the player.

    Args:
        player: The MusicPlayer handle

    Returns:
        Current time value in beats

    Raises:
        RuntimeError: If getting time fails
    """
    cdef at.MusicTimeStamp time
    cdef cf.OSStatus status = at.MusicPlayerGetTime(<at.MusicPlayer>player, &time)

    if status != 0:
        raise RuntimeError(f"MusicPlayerGetTime failed with status: {status}")
    return <double>time

def music_player_preroll(long player):
    """Prepare the player for playing.

    Args:
        player: The MusicPlayer handle

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If preroll fails
    """
    cdef cf.OSStatus status = at.MusicPlayerPreroll(<at.MusicPlayer>player)

    if status != 0:
        raise RuntimeError(f"MusicPlayerPreroll failed with status: {status}")
    return status

def music_player_start(long player):
    """Start the player.

    Args:
        player: The MusicPlayer handle

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If start fails
    """
    cdef cf.OSStatus status = at.MusicPlayerStart(<at.MusicPlayer>player)

    if status != 0:
        raise RuntimeError(f"MusicPlayerStart failed with status: {status}")
    return status

def music_player_stop(long player):
    """Stop the player.

    Args:
        player: The MusicPlayer handle

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If stop fails
    """
    cdef cf.OSStatus status = at.MusicPlayerStop(<at.MusicPlayer>player)

    if status != 0:
        raise RuntimeError(f"MusicPlayerStop failed with status: {status}")
    return status

def music_player_is_playing(long player):
    """Check if the player is playing.

    Args:
        player: The MusicPlayer handle

    Returns:
        True if playing, False if not

    Raises:
        RuntimeError: If check fails
    """
    cdef cf.Boolean is_playing
    cdef cf.OSStatus status = at.MusicPlayerIsPlaying(<at.MusicPlayer>player, &is_playing)

    if status != 0:
        raise RuntimeError(f"MusicPlayerIsPlaying failed with status: {status}")
    return bool(is_playing)

def music_player_set_play_rate_scalar(long player, double scale_rate):
    """Scale the playback rate of the player.

    Args:
        player: The MusicPlayer handle
        scale_rate: Playback rate scalar (e.g., 2.0 = double speed, 0.5 = half speed)

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If setting rate fails
    """
    if scale_rate <= 0:
        raise ValueError("Scale rate must be greater than zero")

    cdef cf.OSStatus status = at.MusicPlayerSetPlayRateScalar(<at.MusicPlayer>player, <ca.Float64>scale_rate)

    if status != 0:
        raise RuntimeError(f"MusicPlayerSetPlayRateScalar failed with status: {status}")
    return status

def music_player_get_play_rate_scalar(long player):
    """Get the playback rate scalar of the player.

    Args:
        player: The MusicPlayer handle

    Returns:
        Current playback rate scalar

    Raises:
        RuntimeError: If getting rate fails
    """
    cdef ca.Float64 scale_rate
    cdef cf.OSStatus status = at.MusicPlayerGetPlayRateScalar(<at.MusicPlayer>player, &scale_rate)

    if status != 0:
        raise RuntimeError(f"MusicPlayerGetPlayRateScalar failed with status: {status}")
    return <double>scale_rate

# MusicSequence functions

def new_music_sequence():
    """Create a new empty music sequence.

    Returns:
        MusicSequence handle

    Raises:
        RuntimeError: If sequence creation fails
    """
    cdef at.MusicSequence sequence
    cdef cf.OSStatus status = at.NewMusicSequence(&sequence)

    if status != 0:
        raise RuntimeError(f"NewMusicSequence failed with status: {status}")
    return <long>sequence

def dispose_music_sequence(long sequence):
    """Dispose a music sequence.

    Args:
        sequence: The MusicSequence handle to dispose

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If disposal fails
    """
    cdef cf.OSStatus status = at.DisposeMusicSequence(<at.MusicSequence>sequence)

    if status != 0:
        raise RuntimeError(f"DisposeMusicSequence failed with status: {status}")
    return status

def music_sequence_new_track(long sequence):
    """Add a new track to the sequence.

    Args:
        sequence: The MusicSequence handle

    Returns:
        MusicTrack handle

    Raises:
        RuntimeError: If track creation fails
    """
    cdef at.MusicTrack track
    cdef cf.OSStatus status = at.MusicSequenceNewTrack(<at.MusicSequence>sequence, &track)

    if status != 0:
        raise RuntimeError(f"MusicSequenceNewTrack failed with status: {status}")
    return <long>track

def music_sequence_dispose_track(long sequence, long track):
    """Remove and dispose a track from a sequence.

    Args:
        sequence: The MusicSequence handle
        track: The MusicTrack handle to remove

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If track disposal fails
    """
    cdef cf.OSStatus status = at.MusicSequenceDisposeTrack(<at.MusicSequence>sequence, <at.MusicTrack>track)

    if status != 0:
        raise RuntimeError(f"MusicSequenceDisposeTrack failed with status: {status}")
    return status

def music_sequence_get_track_count(long sequence):
    """Get the number of tracks in a sequence.

    Args:
        sequence: The MusicSequence handle

    Returns:
        Number of tracks

    Raises:
        RuntimeError: If getting track count fails
    """
    cdef cf.UInt32 track_count
    cdef cf.OSStatus status = at.MusicSequenceGetTrackCount(<at.MusicSequence>sequence, &track_count)

    if status != 0:
        raise RuntimeError(f"MusicSequenceGetTrackCount failed with status: {status}")
    return track_count

def music_sequence_get_ind_track(long sequence, int track_index):
    """Get a track at the specified index.

    Args:
        sequence: The MusicSequence handle
        track_index: Zero-based track index

    Returns:
        MusicTrack handle

    Raises:
        RuntimeError: If getting track fails
    """
    cdef at.MusicTrack track
    cdef cf.OSStatus status = at.MusicSequenceGetIndTrack(<at.MusicSequence>sequence, <cf.UInt32>track_index, &track)

    if status != 0:
        raise RuntimeError(f"MusicSequenceGetIndTrack failed with status: {status}")
    return <long>track

def music_sequence_get_tempo_track(long sequence):
    """Get the tempo track of the sequence.

    Args:
        sequence: The MusicSequence handle

    Returns:
        MusicTrack handle for the tempo track

    Raises:
        RuntimeError: If getting tempo track fails
    """
    cdef at.MusicTrack track
    cdef cf.OSStatus status = at.MusicSequenceGetTempoTrack(<at.MusicSequence>sequence, &track)

    if status != 0:
        raise RuntimeError(f"MusicSequenceGetTempoTrack failed with status: {status}")
    return <long>track

def music_sequence_set_sequence_type(long sequence, int sequence_type):
    """Set the sequence type.

    Args:
        sequence: The MusicSequence handle
        sequence_type: The sequence type (beats, seconds, or samples)

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If setting sequence type fails
    """
    cdef cf.OSStatus status = at.MusicSequenceSetSequenceType(<at.MusicSequence>sequence, <cf.UInt32>sequence_type)

    if status != 0:
        raise RuntimeError(f"MusicSequenceSetSequenceType failed with status: {status}")
    return status

def music_sequence_get_sequence_type(long sequence):
    """Get the sequence type.

    Args:
        sequence: The MusicSequence handle

    Returns:
        Sequence type constant

    Raises:
        RuntimeError: If getting sequence type fails
    """
    cdef cf.UInt32 sequence_type
    cdef cf.OSStatus status = at.MusicSequenceGetSequenceType(<at.MusicSequence>sequence, &sequence_type)

    if status != 0:
        raise RuntimeError(f"MusicSequenceGetSequenceType failed with status: {status}")
    return sequence_type

def music_sequence_file_load(long sequence, str file_path, int file_type_hint=0, int flags=0):
    """Load a file into the sequence.

    Args:
        sequence: The MusicSequence handle
        file_path: Path to the file to load
        file_type_hint: File type hint (default 0 for auto-detect)
        flags: Load flags (default 0)

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If file loading fails
    """
    cdef bytes path_bytes = file_path.encode('utf-8')
    cdef cf.CFURLRef url_ref = cf.CFURLCreateFromFileSystemRepresentation(
        cf.kCFAllocatorDefault,
        <const cf.UInt8*>path_bytes,
        len(path_bytes),
        False
    )
    cdef cf.OSStatus status

    if not url_ref:
        raise ValueError(f"Could not create URL from file path: {file_path}")

    try:
        status = at.MusicSequenceFileLoad(
            <at.MusicSequence>sequence,
            url_ref,
            <cf.UInt32>file_type_hint,
            <cf.UInt32>flags
        )

        if status != 0:
            raise RuntimeError(f"MusicSequenceFileLoad failed with status: {status}")
        return status

    finally:
        cf.CFRelease(url_ref)

# MusicTrack functions

def music_track_new_midi_note_event(long track, double timestamp, int channel, int note, int velocity, int release_velocity, double duration):
    """Add a MIDI note event to a track.

    Args:
        track: The MusicTrack handle
        timestamp: The time stamp in beats
        channel: MIDI channel (0-15)
        note: MIDI note number (0-127)
        velocity: Note on velocity (0-127)
        release_velocity: Note off velocity (0-127)
        duration: Note duration in beats

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If adding event fails
    """
    cdef at.MIDINoteMessage message
    message.channel = <cf.UInt8>(channel & 0x0F)
    message.note = <cf.UInt8>(note & 0x7F)
    message.velocity = <cf.UInt8>(velocity & 0x7F)
    message.releaseVelocity = <cf.UInt8>(release_velocity & 0x7F)
    message.duration = <ca.Float32>duration

    cdef cf.OSStatus status = at.MusicTrackNewMIDINoteEvent(
        <at.MusicTrack>track,
        <at.MusicTimeStamp>timestamp,
        &message
    )

    if status != 0:
        raise RuntimeError(f"MusicTrackNewMIDINoteEvent failed with status: {status}")
    return status

def music_track_new_midi_channel_event(long track, double timestamp, int status, int data1, int data2):
    """Add a MIDI channel event to a track.

    Args:
        track: The MusicTrack handle
        timestamp: The time stamp in beats
        status: MIDI status byte
        data1: First data byte
        data2: Second data byte

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If adding event fails
    """
    cdef at.MIDIChannelMessage message
    message.status = <cf.UInt8>status
    message.data1 = <cf.UInt8>data1
    message.data2 = <cf.UInt8>data2
    message.reserved = 0

    cdef cf.OSStatus status_result = at.MusicTrackNewMIDIChannelEvent(
        <at.MusicTrack>track,
        <at.MusicTimeStamp>timestamp,
        &message
    )

    if status_result != 0:
        raise RuntimeError(f"MusicTrackNewMIDIChannelEvent failed with status: {status_result}")
    return status_result

def music_track_new_extended_tempo_event(long track, double timestamp, double bpm):
    """Add a tempo event to a track.

    Args:
        track: The MusicTrack handle
        timestamp: The time stamp in beats
        bpm: Beats per minute

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If adding event fails
    """
    if bpm <= 0:
        raise ValueError("BPM must be greater than zero")

    cdef cf.OSStatus status = at.MusicTrackNewExtendedTempoEvent(
        <at.MusicTrack>track,
        <at.MusicTimeStamp>timestamp,
        <ca.Float64>bpm
    )

    if status != 0:
        raise RuntimeError(f"MusicTrackNewExtendedTempoEvent failed with status: {status}")
    return status

# MusicPlayer constants

def get_music_event_type_null():
    """Get the NULL event type constant."""
    return at.kMusicEventType_NULL

def get_music_event_type_extended_note():
    """Get the extended note event type constant."""
    return at.kMusicEventType_ExtendedNote

def get_music_event_type_extended_tempo():
    """Get the extended tempo event type constant."""
    return at.kMusicEventType_ExtendedTempo

def get_music_event_type_user():
    """Get the user event type constant."""
    return at.kMusicEventType_User

def get_music_event_type_meta():
    """Get the meta event type constant."""
    return at.kMusicEventType_Meta

def get_music_event_type_midi_note_message():
    """Get the MIDI note message event type constant."""
    return at.kMusicEventType_MIDINoteMessage

def get_music_event_type_midi_channel_message():
    """Get the MIDI channel message event type constant."""
    return at.kMusicEventType_MIDIChannelMessage

def get_music_event_type_midi_raw_data():
    """Get the MIDI raw data event type constant."""
    return at.kMusicEventType_MIDIRawData

def get_music_event_type_parameter():
    """Get the parameter event type constant."""
    return at.kMusicEventType_Parameter

def get_music_event_type_au_preset():
    """Get the AU preset event type constant."""
    return at.kMusicEventType_AUPreset

def get_music_sequence_type_beats():
    """Get the beats sequence type constant."""
    return at.kMusicSequenceType_Beats

def get_music_sequence_type_seconds():
    """Get the seconds sequence type constant."""
    return at.kMusicSequenceType_Seconds

def get_music_sequence_type_samples():
    """Get the samples sequence type constant."""
    return at.kMusicSequenceType_Samples

def get_music_sequence_file_any_type():
    """Get the any file type constant."""
    return at.kMusicSequenceFile_AnyType

def get_music_sequence_file_midi_type():
    """Get the MIDI file type constant."""
    return at.kMusicSequenceFile_MIDIType

def get_music_sequence_file_imelody_type():
    """Get the iMelody file type constant."""
    return at.kMusicSequenceFile_iMelodyType

def get_sequence_track_property_loop_info():
    """Get the loop info track property constant."""
    return at.kSequenceTrackProperty_LoopInfo

def get_sequence_track_property_offset_time():
    """Get the offset time track property constant."""
    return at.kSequenceTrackProperty_OffsetTime

def get_sequence_track_property_mute_status():
    """Get the mute status track property constant."""
    return at.kSequenceTrackProperty_MuteStatus

def get_sequence_track_property_solo_status():
    """Get the solo status track property constant."""
    return at.kSequenceTrackProperty_SoloStatus

def get_sequence_track_property_automated_parameters():
    """Get the automated parameters track property constant."""
    return at.kSequenceTrackProperty_AutomatedParameters

def get_sequence_track_property_track_length():
    """Get the track length property constant."""
    return at.kSequenceTrackProperty_TrackLength

def get_sequence_track_property_time_resolution():
    """Get the time resolution property constant."""
    return at.kSequenceTrackProperty_TimeResolution

# Helper functions

def create_midi_note_message(int channel, int note, int velocity, int release_velocity=0, double duration=1.0):
    """Create a MIDI note message dictionary.

    Args:
        channel: MIDI channel (0-15)
        note: MIDI note number (0-127)
        velocity: Note on velocity (0-127)
        release_velocity: Note off velocity (0-127, default 0)
        duration: Note duration in beats (default 1.0)

    Returns:
        Dictionary with note message parameters
    """
    return {
        'channel': channel & 0x0F,
        'note': note & 0x7F,
        'velocity': velocity & 0x7F,
        'release_velocity': release_velocity & 0x7F,
        'duration': duration
    }

def create_midi_channel_message(int status, int data1, int data2=0):
    """Create a MIDI channel message dictionary.

    Args:
        status: MIDI status byte
        data1: First data byte
        data2: Second data byte (default 0)

    Returns:
        Dictionary with channel message parameters
    """
    return {
        'status': status & 0xFF,
        'data1': data1 & 0x7F,
        'data2': data2 & 0x7F
    }


# ===== CoreMIDI API =====

# Client functions

def midi_client_create(str name):
    """Create a MIDI client.

    Args:
        name: The client's name

    Returns:
        MIDIClientRef handle

    Raises:
        RuntimeError: If client creation fails
    """
    cdef bytes name_bytes = name.encode('utf-8')
    cdef cf.CFStringRef cf_name = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault, name_bytes, cf.kCFStringEncodingUTF8)
    cdef cm.MIDIClientRef client
    cdef cf.OSStatus status

    if not cf_name:
        raise MemoryError("Could not create CFString from name")

    try:
        status = cm.MIDIClientCreate(cf_name, NULL, NULL, &client)
        if status != 0:
            raise RuntimeError(f"MIDIClientCreate failed with status: {status}")
        return <long>client
    finally:
        cf.CFRelease(cf_name)

def midi_client_dispose(long client):
    """Dispose a MIDI client.

    Args:
        client: The MIDIClientRef to dispose

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If disposal fails
    """
    cdef cf.OSStatus status = cm.MIDIClientDispose(<cm.MIDIClientRef>client)
    if status != 0:
        raise RuntimeError(f"MIDIClientDispose failed with status: {status}")
    return status

# Port functions

def midi_input_port_create(long client, str port_name):
    """Create a MIDI input port.

    Args:
        client: The MIDIClientRef
        port_name: Name for the input port

    Returns:
        MIDIPortRef handle

    Raises:
        RuntimeError: If port creation fails
    """
    cdef bytes port_name_bytes = port_name.encode('utf-8')
    cdef cf.CFStringRef cf_name = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault, port_name_bytes, cf.kCFStringEncodingUTF8)
    cdef cm.MIDIPortRef port
    cdef cf.OSStatus status

    if not cf_name:
        raise MemoryError("Could not create CFString from port name")

    try:
        status = cm.MIDIInputPortCreate(
            <cm.MIDIClientRef>client, cf_name, NULL, NULL, &port)
        if status != 0:
            raise RuntimeError(f"MIDIInputPortCreate failed with status: {status}")
        return <long>port
    finally:
        cf.CFRelease(cf_name)

def midi_output_port_create(long client, str port_name):
    """Create a MIDI output port.

    Args:
        client: The MIDIClientRef
        port_name: Name for the output port

    Returns:
        MIDIPortRef handle

    Raises:
        RuntimeError: If port creation fails
    """
    cdef bytes port_name_bytes = port_name.encode('utf-8')
    cdef cf.CFStringRef cf_name = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault, port_name_bytes, cf.kCFStringEncodingUTF8)
    cdef cm.MIDIPortRef port
    cdef cf.OSStatus status

    if not cf_name:
        raise MemoryError("Could not create CFString from port name")

    try:
        status = cm.MIDIOutputPortCreate(
            <cm.MIDIClientRef>client, cf_name, &port)
        if status != 0:
            raise RuntimeError(f"MIDIOutputPortCreate failed with status: {status}")
        return <long>port
    finally:
        cf.CFRelease(cf_name)

def midi_port_dispose(long port):
    """Dispose a MIDI port.

    Args:
        port: The MIDIPortRef to dispose

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If disposal fails
    """
    cdef cf.OSStatus status = cm.MIDIPortDispose(<cm.MIDIPortRef>port)
    if status != 0:
        raise RuntimeError(f"MIDIPortDispose failed with status: {status}")
    return status

def midi_port_connect_source(long port, long source):
    """Connect a source to an input port.

    Args:
        port: The MIDIPortRef (input port)
        source: The MIDIEndpointRef (source endpoint)

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If connection fails
    """
    cdef cf.OSStatus status = cm.MIDIPortConnectSource(
        <cm.MIDIPortRef>port, <cm.MIDIEndpointRef>source, NULL)
    if status != 0:
        raise RuntimeError(f"MIDIPortConnectSource failed with status: {status}")
    return status

def midi_port_disconnect_source(long port, long source):
    """Disconnect a source from an input port.

    Args:
        port: The MIDIPortRef (input port)
        source: The MIDIEndpointRef (source endpoint)

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If disconnection fails
    """
    cdef cf.OSStatus status = cm.MIDIPortDisconnectSource(
        <cm.MIDIPortRef>port, <cm.MIDIEndpointRef>source)
    if status != 0:
        raise RuntimeError(f"MIDIPortDisconnectSource failed with status: {status}")
    return status

# Device and endpoint discovery

def midi_get_number_of_devices():
    """Get the number of MIDI devices in the system.

    Returns:
        Number of devices
    """
    return cm.MIDIGetNumberOfDevices()

def midi_get_device(int device_index) -> int:
    """Get a device by index.

    Args:
        device_index: Zero-based device index

    Returns:
        MIDIDeviceRef handle

    Raises:
        ValueError: If index is out of range
    """
    cdef int num_devices = cm.MIDIGetNumberOfDevices()
    if device_index < 0 or device_index >= num_devices:
        raise ValueError(f"Device index {device_index} out of range (0-{num_devices-1})")

    cdef cm.MIDIDeviceRef device = cm.MIDIGetDevice(<cf.UInt32>device_index)
    return <long>device

def midi_device_get_number_of_entities(long device):
    """Get the number of entities in a device.

    Args:
        device: The MIDIDeviceRef

    Returns:
        Number of entities
    """
    return cm.MIDIDeviceGetNumberOfEntities(<cm.MIDIDeviceRef>device)

def midi_device_get_entity(long device, int entity_index):
    """Get an entity from a device by index.

    Args:
        device: The MIDIDeviceRef
        entity_index: Zero-based entity index

    Returns:
        MIDIEntityRef handle

    Raises:
        ValueError: If index is out of range
    """
    cdef int num_entities = cm.MIDIDeviceGetNumberOfEntities(<cm.MIDIDeviceRef>device)
    if entity_index < 0 or entity_index >= num_entities:
        raise ValueError(f"Entity index {entity_index} out of range (0-{num_entities-1})")

    cdef cm.MIDIEntityRef entity = cm.MIDIDeviceGetEntity(
        <cm.MIDIDeviceRef>device, <cf.UInt32>entity_index)
    return <long>entity

def midi_entity_get_number_of_sources(long entity):
    """Get the number of sources in an entity.

    Args:
        entity: The MIDIEntityRef

    Returns:
        Number of sources
    """
    return cm.MIDIEntityGetNumberOfSources(<cm.MIDIEntityRef>entity)

def midi_entity_get_source(long entity, int source_index):
    """Get a source from an entity by index.

    Args:
        entity: The MIDIEntityRef
        source_index: Zero-based source index

    Returns:
        MIDIEndpointRef handle

    Raises:
        ValueError: If index is out of range
    """
    cdef int num_sources = cm.MIDIEntityGetNumberOfSources(<cm.MIDIEntityRef>entity)
    if source_index < 0 or source_index >= num_sources:
        raise ValueError(f"Source index {source_index} out of range (0-{num_sources-1})")

    cdef cm.MIDIEndpointRef source = cm.MIDIEntityGetSource(
        <cm.MIDIEntityRef>entity, <cf.UInt32>source_index)
    return <long>source

def midi_entity_get_number_of_destinations(long entity):
    """Get the number of destinations in an entity.

    Args:
        entity: The MIDIEntityRef

    Returns:
        Number of destinations
    """
    return cm.MIDIEntityGetNumberOfDestinations(<cm.MIDIEntityRef>entity)

def midi_entity_get_destination(long entity, int dest_index):
    """Get a destination from an entity by index.

    Args:
        entity: The MIDIEntityRef
        dest_index: Zero-based destination index

    Returns:
        MIDIEndpointRef handle

    Raises:
        ValueError: If index is out of range
    """
    cdef int num_dests = cm.MIDIEntityGetNumberOfDestinations(<cm.MIDIEntityRef>entity)
    if dest_index < 0 or dest_index >= num_dests:
        raise ValueError(f"Destination index {dest_index} out of range (0-{num_dests-1})")

    cdef cm.MIDIEndpointRef dest = cm.MIDIEntityGetDestination(
        <cm.MIDIEntityRef>entity, <cf.UInt32>dest_index)
    return <long>dest

def midi_get_number_of_sources():
    """Get the total number of MIDI sources in the system.

    Returns:
        Number of sources
    """
    return cm.MIDIGetNumberOfSources()

def midi_get_source(int source_index) -> int:
    """Get a source by system-wide index.

    Args:
        source_index: Zero-based source index

    Returns:
        MIDIEndpointRef handle

    Raises:
        ValueError: If index is out of range
    """
    cdef int num_sources = cm.MIDIGetNumberOfSources()
    if source_index < 0 or source_index >= num_sources:
        raise ValueError(f"Source index {source_index} out of range (0-{num_sources-1})")

    cdef cm.MIDIEndpointRef source = cm.MIDIGetSource(<cf.UInt32>source_index)
    return <long>source

def midi_get_number_of_destinations():
    """Get the total number of MIDI destinations in the system.

    Returns:
        Number of destinations
    """
    return cm.MIDIGetNumberOfDestinations()

def midi_get_destination(int dest_index):
    """Get a destination by system-wide index.

    Args:
        dest_index: Zero-based destination index

    Returns:
        MIDIEndpointRef handle

    Raises:
        ValueError: If index is out of range
    """
    cdef int num_dests = cm.MIDIGetNumberOfDestinations()
    if dest_index < 0 or dest_index >= num_dests:
        raise ValueError(f"Destination index {dest_index} out of range (0-{num_dests-1})")

    cdef cm.MIDIEndpointRef dest = cm.MIDIGetDestination(<cf.UInt32>dest_index)
    return <long>dest

# Virtual endpoint functions

def midi_source_create(long client, str name):
    """Create a virtual MIDI source.

    Args:
        client: The MIDIClientRef
        name: Name for the virtual source

    Returns:
        MIDIEndpointRef handle

    Raises:
        RuntimeError: If source creation fails
    """
    cdef bytes name_bytes = name.encode('utf-8')
    cdef cf.CFStringRef cf_name = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault, name_bytes, cf.kCFStringEncodingUTF8)
    cdef cm.MIDIEndpointRef source
    cdef cf.OSStatus status

    if not cf_name:
        raise MemoryError("Could not create CFString from name")

    try:
        status = cm.MIDISourceCreate(<cm.MIDIClientRef>client, cf_name, &source)
        if status != 0:
            raise RuntimeError(f"MIDISourceCreate failed with status: {status}")
        return <long>source
    finally:
        cf.CFRelease(cf_name)

def midi_destination_create(long client, str name):
    """Create a virtual MIDI destination.

    Args:
        client: The MIDIClientRef
        name: Name for the virtual destination

    Returns:
        MIDIEndpointRef handle

    Raises:
        RuntimeError: If destination creation fails
    """
    cdef bytes name_bytes = name.encode('utf-8')
    cdef cf.CFStringRef cf_name = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault, name_bytes, cf.kCFStringEncodingUTF8)
    cdef cm.MIDIEndpointRef dest
    cdef cf.OSStatus status

    if not cf_name:
        raise MemoryError("Could not create CFString from name")

    try:
        status = cm.MIDIDestinationCreate(
            <cm.MIDIClientRef>client, cf_name, NULL, NULL, &dest)
        if status != 0:
            raise RuntimeError(f"MIDIDestinationCreate failed with status: {status}")
        return <long>dest
    finally:
        cf.CFRelease(cf_name)

def midi_endpoint_dispose(long endpoint):
    """Dispose a virtual MIDI endpoint.

    Args:
        endpoint: The MIDIEndpointRef to dispose

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If disposal fails
    """
    cdef cf.OSStatus status = cm.MIDIEndpointDispose(<cm.MIDIEndpointRef>endpoint)
    if status != 0:
        raise RuntimeError(f"MIDIEndpointDispose failed with status: {status}")
    return status

# Property functions

def midi_object_get_string_property(long obj, str property_name):
    """Get a string property from a MIDI object.

    Args:
        obj: The MIDIObjectRef
        property_name: Name of the property to get

    Returns:
        String value of the property

    Raises:
        RuntimeError: If getting property fails
        ValueError: If property name conversion fails
    """
    cdef cf.CFStringRef cf_prop_name
    cdef cf.CFStringRef cf_value
    cdef cf.OSStatus status
    cdef char* c_str
    cdef cf.CFIndex length
    cdef cf.CFIndex max_size
    cdef char* buffer

    cf_prop_name = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault, property_name.encode('utf-8'), cf.kCFStringEncodingUTF8)

    if not cf_prop_name:
        raise ValueError("Could not create CFString from property name")

    try:
        status = cm.MIDIObjectGetStringProperty(
            <cm.MIDIObjectRef>obj, cf_prop_name, &cf_value)
        if status != 0:
            raise RuntimeError(f"MIDIObjectGetStringProperty failed with status: {status}")

        # Convert CFString to Python string
        c_str = <char*>cf.CFStringGetCStringPtr(cf_value, cf.kCFStringEncodingUTF8)
        if c_str:
            result = c_str.decode('utf-8')
        else:
            # Fallback for when direct pointer isn't available
            length = cf.CFStringGetLength(cf_value)
            max_size = cf.CFStringGetMaximumSizeForEncoding(length, cf.kCFStringEncodingUTF8) + 1
            buffer = <char*>malloc(max_size)
            if not buffer:
                cf.CFRelease(cf_value)
                raise MemoryError("Could not allocate buffer for string conversion")
            try:
                if cf.CFStringGetCString(cf_value, buffer, max_size, cf.kCFStringEncodingUTF8):
                    result = buffer.decode('utf-8')
                else:
                    cf.CFRelease(cf_value)
                    raise RuntimeError("Could not convert CFString to C string")
            finally:
                free(buffer)

        cf.CFRelease(cf_value)
        return result
    finally:
        cf.CFRelease(cf_prop_name)

def midi_object_get_integer_property(long obj, str property_name):
    """Get an integer property from a MIDI object.

    Args:
        obj: The MIDIObjectRef
        property_name: Name of the property to get

    Returns:
        Integer value of the property

    Raises:
        RuntimeError: If getting property fails
        ValueError: If property name conversion fails
    """
    cdef bytes prop_bytes = property_name.encode('utf-8')
    cdef cf.CFStringRef cf_prop_name = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault, prop_bytes, cf.kCFStringEncodingUTF8)
    cdef ca.SInt32 value
    cdef cf.OSStatus status

    if not cf_prop_name:
        raise ValueError("Could not create CFString from property name")

    try:
        status = cm.MIDIObjectGetIntegerProperty(
            <cm.MIDIObjectRef>obj, cf_prop_name, &value)
        if status != 0:
            raise RuntimeError(f"MIDIObjectGetIntegerProperty failed with status: {status}")
        return value
    finally:
        cf.CFRelease(cf_prop_name)

def midi_object_set_string_property(long obj, str property_name, str value):
    """Set a string property on a MIDI object.

    Args:
        obj: The MIDIObjectRef
        property_name: Name of the property to set
        value: String value to set

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If setting property fails
        ValueError: If string conversion fails
    """
    cdef bytes prop_bytes = property_name.encode('utf-8')
    cdef cf.CFStringRef cf_prop_name = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault, prop_bytes, cf.kCFStringEncodingUTF8)
    cdef bytes value_bytes = value.encode('utf-8')
    cdef cf.CFStringRef cf_value = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault, value_bytes, cf.kCFStringEncodingUTF8)
    cdef cf.OSStatus status

    if not cf_prop_name:
        raise ValueError("Could not create CFString from property name")
    if not cf_value:
        cf.CFRelease(cf_prop_name)
        raise ValueError("Could not create CFString from value")

    try:
        status = cm.MIDIObjectSetStringProperty(
            <cm.MIDIObjectRef>obj, cf_prop_name, cf_value)
        if status != 0:
            raise RuntimeError(f"MIDIObjectSetStringProperty failed with status: {status}")
        return status
    finally:
        cf.CFRelease(cf_prop_name)
        cf.CFRelease(cf_value)

def midi_object_set_integer_property(long obj, str property_name, int value):
    """Set an integer property on a MIDI object.

    Args:
        obj: The MIDIObjectRef
        property_name: Name of the property to set
        value: Integer value to set

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If setting property fails
        ValueError: If property name conversion fails
    """
    cdef bytes prop_bytes = property_name.encode('utf-8')
    cdef cf.CFStringRef cf_prop_name = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault, prop_bytes, cf.kCFStringEncodingUTF8)
    cdef cf.OSStatus status

    if not cf_prop_name:
        raise ValueError("Could not create CFString from property name")

    try:
        status = cm.MIDIObjectSetIntegerProperty(
            <cm.MIDIObjectRef>obj, cf_prop_name, <ca.SInt32>value)
        if status != 0:
            raise RuntimeError(f"MIDIObjectSetIntegerProperty failed with status: {status}")
        return status
    finally:
        cf.CFRelease(cf_prop_name)

# Convenience functions for getting common properties

def midi_object_get_name(long obj):
    """Get the name of a MIDI object (device, entity, or endpoint).

    Args:
        obj: The MIDIObjectRef (device, entity, or endpoint)

    Returns:
        String name of the object, or None if no name is set

    Raises:
        RuntimeError: If getting the name fails
    """
    cdef cf.CFStringRef cf_value
    cdef cf.OSStatus status
    cdef char* c_str
    cdef cf.CFIndex length
    cdef cf.CFIndex max_size
    cdef char* buffer

    status = cm.MIDIObjectGetStringProperty(
        <cm.MIDIObjectRef>obj, cm.kMIDIPropertyName, &cf_value)
    if status != 0:
        raise RuntimeError(f"MIDIObjectGetStringProperty failed with status: {status}")

    try:
        # Convert CFString to Python string
        c_str = <char*>cf.CFStringGetCStringPtr(cf_value, cf.kCFStringEncodingUTF8)
        if c_str:
            result = c_str.decode('utf-8')
        else:
            # Fallback for when direct pointer isn't available
            length = cf.CFStringGetLength(cf_value)
            max_size = cf.CFStringGetMaximumSizeForEncoding(length, cf.kCFStringEncodingUTF8) + 1
            buffer = <char*>malloc(max_size)
            if not buffer:
                raise MemoryError("Could not allocate buffer for string conversion")
            try:
                if cf.CFStringGetCString(cf_value, buffer, max_size, cf.kCFStringEncodingUTF8):
                    result = buffer.decode('utf-8')
                else:
                    raise RuntimeError("Could not convert CFString to C string")
            finally:
                free(buffer)

        return result
    finally:
        cf.CFRelease(cf_value)

def midi_device_get_name(long device):
    """Get the name of a MIDI device.

    Args:
        device: The MIDIDeviceRef

    Returns:
        String name of the device, or None if no name is set

    Raises:
        RuntimeError: If getting the device name fails
    """
    return midi_object_get_name(device)

def midi_endpoint_get_name(long endpoint):
    """Get the name of a MIDI endpoint (source or destination).

    Args:
        endpoint: The MIDIEndpointRef

    Returns:
        String name of the endpoint, or None if no name is set

    Raises:
        RuntimeError: If getting the endpoint name fails
    """
    return midi_object_get_name(endpoint)

def midi_entity_get_name(long entity):
    """Get the name of a MIDI entity.

    Args:
        entity: The MIDIEntityRef

    Returns:
        String name of the entity, or None if no name is set

    Raises:
        RuntimeError: If getting the entity name fails
    """
    return midi_object_get_name(entity)

def midi_object_get_manufacturer(long obj):
    """Get the manufacturer of a MIDI object.

    Args:
        obj: The MIDIObjectRef (device or endpoint)

    Returns:
        String manufacturer name, or None if not set

    Raises:
        RuntimeError: If getting the manufacturer fails
    """
    cdef cf.CFStringRef cf_value
    cdef cf.OSStatus status
    cdef char* c_str
    cdef cf.CFIndex length
    cdef cf.CFIndex max_size
    cdef char* buffer

    status = cm.MIDIObjectGetStringProperty(
        <cm.MIDIObjectRef>obj, cm.kMIDIPropertyManufacturer, &cf_value)
    if status != 0:
        raise RuntimeError(f"MIDIObjectGetStringProperty failed with status: {status}")

    try:
        # Convert CFString to Python string
        c_str = <char*>cf.CFStringGetCStringPtr(cf_value, cf.kCFStringEncodingUTF8)
        if c_str:
            result = c_str.decode('utf-8')
        else:
            # Fallback for when direct pointer isn't available
            length = cf.CFStringGetLength(cf_value)
            max_size = cf.CFStringGetMaximumSizeForEncoding(length, cf.kCFStringEncodingUTF8) + 1
            buffer = <char*>malloc(max_size)
            if not buffer:
                raise MemoryError("Could not allocate buffer for string conversion")
            try:
                if cf.CFStringGetCString(cf_value, buffer, max_size, cf.kCFStringEncodingUTF8):
                    result = buffer.decode('utf-8')
                else:
                    raise RuntimeError("Could not convert CFString to C string")
            finally:
                free(buffer)

        return result
    finally:
        cf.CFRelease(cf_value)

def midi_object_get_model(long obj):
    """Get the model of a MIDI object.

    Args:
        obj: The MIDIObjectRef (device or endpoint)

    Returns:
        String model name, or None if not set

    Raises:
        RuntimeError: If getting the model fails
    """
    cdef cf.CFStringRef cf_value
    cdef cf.OSStatus status
    cdef char* c_str
    cdef cf.CFIndex length
    cdef cf.CFIndex max_size
    cdef char* buffer

    status = cm.MIDIObjectGetStringProperty(
        <cm.MIDIObjectRef>obj, cm.kMIDIPropertyModel, &cf_value)
    if status != 0:
        raise RuntimeError(f"MIDIObjectGetStringProperty failed with status: {status}")

    try:
        # Convert CFString to Python string
        c_str = <char*>cf.CFStringGetCStringPtr(cf_value, cf.kCFStringEncodingUTF8)
        if c_str:
            result = c_str.decode('utf-8')
        else:
            # Fallback for when direct pointer isn't available
            length = cf.CFStringGetLength(cf_value)
            max_size = cf.CFStringGetMaximumSizeForEncoding(length, cf.kCFStringEncodingUTF8) + 1
            buffer = <char*>malloc(max_size)
            if not buffer:
                raise MemoryError("Could not allocate buffer for string conversion")
            try:
                if cf.CFStringGetCString(cf_value, buffer, max_size, cf.kCFStringEncodingUTF8):
                    result = buffer.decode('utf-8')
                else:
                    raise RuntimeError("Could not convert CFString to C string")
            finally:
                free(buffer)

        return result
    finally:
        cf.CFRelease(cf_value)

# Send functions with simplified packet creation

def midi_send_data(long port, long destination, bytes data, int timestamp=0):
    """Send MIDI data to a destination.

    Args:
        port: The MIDIPortRef (output port)
        destination: The MIDIEndpointRef (destination endpoint)
        data: MIDI data bytes to send
        timestamp: MIDI timestamp (default 0 for immediate)

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If sending fails
        ValueError: If data is too large
    """
    cdef size_t pktlist_size
    cdef cm.MIDIPacketList* pktlist
    cdef cm.MIDIPacket* packet
    cdef cf.OSStatus status

    if len(data) > 256:
        raise ValueError("MIDI data too large (max 256 bytes)")

    # Create a packet list with one packet
    pktlist_size = sizeof(cm.MIDIPacketList) + len(data)
    pktlist = <cm.MIDIPacketList*>malloc(pktlist_size)
    if not pktlist:
        raise MemoryError("Could not allocate packet list")

    try:
        # Initialize packet list
        packet = cm.MIDIPacketListInit(pktlist)

        # Add our data to the packet
        packet = cm.MIDIPacketListAdd(
            pktlist, pktlist_size, packet,
            <cm.MIDITimeStamp>timestamp,
            <cf.UInt32>len(data),
            <const cf.UInt8*><char*>data)

        if not packet:
            raise RuntimeError("Could not add packet to packet list")

        # Send the packet list
        status = cm.MIDISend(
            <cm.MIDIPortRef>port,
            <cm.MIDIEndpointRef>destination,
            pktlist)

        if status != 0:
            raise RuntimeError(f"MIDISend failed with status: {status}")
        return status

    finally:
        free(pktlist)

# Constants and helpers

def get_midi_error_invalid_client():
    """Get the kMIDIInvalidClient error constant."""
    return cm.kMIDIInvalidClient

def get_midi_error_invalid_port():
    """Get the kMIDIInvalidPort error constant."""
    return cm.kMIDIInvalidPort

def get_midi_error_wrong_endpoint_type():
    """Get the kMIDIWrongEndpointType error constant."""
    return cm.kMIDIWrongEndpointType

def get_midi_error_no_connection():
    """Get the kMIDINoConnection error constant."""
    return cm.kMIDINoConnection

def get_midi_error_unknown_endpoint():
    """Get the kMIDIUnknownEndpoint error constant."""
    return cm.kMIDIUnknownEndpoint

def get_midi_error_unknown_property():
    """Get the kMIDIUnknownProperty error constant."""
    return cm.kMIDIUnknownProperty

def get_midi_error_wrong_property_type():
    """Get the kMIDIWrongPropertyType error constant."""
    return cm.kMIDIWrongPropertyType

def get_midi_error_no_current_setup():
    """Get the kMIDINoCurrentSetup error constant."""
    return cm.kMIDINoCurrentSetup

def get_midi_error_message_send_err():
    """Get the kMIDIMessageSendErr error constant."""
    return cm.kMIDIMessageSendErr

def get_midi_error_server_start_err():
    """Get the kMIDIServerStartErr error constant."""
    return cm.kMIDIServerStartErr

def get_midi_error_setup_format_err():
    """Get the kMIDISetupFormatErr error constant."""
    return cm.kMIDISetupFormatErr

def get_midi_error_wrong_thread():
    """Get the kMIDIWrongThread error constant."""
    return cm.kMIDIWrongThread

def get_midi_error_object_not_found():
    """Get the kMIDIObjectNotFound error constant."""
    return cm.kMIDIObjectNotFound

def get_midi_error_id_not_unique():
    """Get the kMIDIIDNotUnique error constant."""
    return cm.kMIDIIDNotUnique

def get_midi_error_not_permitted():
    """Get the kMIDINotPermitted error constant."""
    return cm.kMIDINotPermitted

def get_midi_error_unknown_error():
    """Get the kMIDIUnknownError error constant."""
    return cm.kMIDIUnknownError

def get_midi_object_type_other():
    """Get the kMIDIObjectType_Other constant."""
    return cm.kMIDIObjectType_Other

def get_midi_object_type_device():
    """Get the kMIDIObjectType_Device constant."""
    return cm.kMIDIObjectType_Device

def get_midi_object_type_entity():
    """Get the kMIDIObjectType_Entity constant."""
    return cm.kMIDIObjectType_Entity

def get_midi_object_type_source():
    """Get the kMIDIObjectType_Source constant."""
    return cm.kMIDIObjectType_Source

def get_midi_object_type_destination():
    """Get the kMIDIObjectType_Destination constant."""
    return cm.kMIDIObjectType_Destination

def get_midi_object_type_external_device():
    """Get the kMIDIObjectType_ExternalDevice constant."""
    return cm.kMIDIObjectType_ExternalDevice

def get_midi_object_type_external_entity():
    """Get the kMIDIObjectType_ExternalEntity constant."""
    return cm.kMIDIObjectType_ExternalEntity

def get_midi_object_type_external_source():
    """Get the kMIDIObjectType_ExternalSource constant."""
    return cm.kMIDIObjectType_ExternalSource

def get_midi_object_type_external_destination():
    """Get the kMIDIObjectType_ExternalDestination constant."""
    return cm.kMIDIObjectType_ExternalDestination

def get_midi_protocol_1_0():
    """Get the kMIDIProtocol_1_0 constant."""
    return cm.kMIDIProtocol_1_0

def get_midi_protocol_2_0():
    """Get the kMIDIProtocol_2_0 constant."""
    return cm.kMIDIProtocol_2_0

# Common property name helpers

def get_midi_property_name():
    """Get the 'name' property key."""
    return "name"

def get_midi_property_manufacturer():
    """Get the 'manufacturer' property key."""
    return "manufacturer"

def get_midi_property_model():
    """Get the 'model' property key."""
    return "model"

def get_midi_property_unique_id():
    """Get the 'uniqueID' property key."""
    return "uniqueID"

def get_midi_property_device_id():
    """Get the 'deviceID' property key."""
    return "deviceID"

def get_midi_property_receive_channels():
    """Get the 'receiveChannels' property key."""
    return "receiveChannels"

def get_midi_property_transmit_channels():
    """Get the 'transmitChannels' property key."""
    return "transmitChannels"

def get_midi_property_offline():
    """Get the 'offline' property key."""
    return "offline"

def get_midi_property_private():
    """Get the 'private' property key."""
    return "private"

def get_midi_property_driver_owner():
    """Get the 'driverOwner' property key."""
    return "driverOwner"

def get_midi_property_display_name():
    """Get the 'displayName' property key."""
    return "displayName"


# MIDI Messages (Universal MIDI Packet) Functions

def midi_message_type_for_up_word(int word):
    """Get the message type from a Universal MIDI Packet word.

    Args:
        word: 32-bit Universal MIDI Packet word

    Returns:
        MIDIMessageType enum value
    """
    return cm.MIDIMessageTypeForUPWord(<cf.UInt32>word)

# MIDI 1.0 Universal MIDI Packet Functions

def midi1_up_channel_voice_message(int group, int status, int channel, int data1, int data2):
    """Create a MIDI 1.0 Universal Packet channel voice message.

    Args:
        group: MIDI group (0-15)
        status: MIDI status nibble
        channel: MIDI channel (0-15)
        data1: First data byte
        data2: Second data byte

    Returns:
        32-bit MIDI message
    """
    return cm.MIDI1UPChannelVoiceMessage(<cf.UInt8>group, <cf.UInt8>status,
                                           <cf.UInt8>channel, <cf.UInt8>data1, <cf.UInt8>data2)

def midi1_up_note_off(int group, int channel, int note_number, int velocity):
    """Create a MIDI 1.0 Universal Packet Note Off message.

    Args:
        group: MIDI group (0-15)
        channel: MIDI channel (0-15)
        note_number: Note number (0-127)
        velocity: Note velocity (0-127)

    Returns:
        32-bit MIDI message
    """
    return cm.MIDI1UPNoteOff(<cf.UInt8>group, <cf.UInt8>channel,
                               <cf.UInt8>note_number, <cf.UInt8>velocity)

def midi1_up_note_on(int group, int channel, int note_number, int velocity):
    """Create a MIDI 1.0 Universal Packet Note On message.

    Args:
        group: MIDI group (0-15)
        channel: MIDI channel (0-15)
        note_number: Note number (0-127)
        velocity: Note velocity (0-127)

    Returns:
        32-bit MIDI message
    """
    return cm.MIDI1UPNoteOn(<cf.UInt8>group, <cf.UInt8>channel,
                              <cf.UInt8>note_number, <cf.UInt8>velocity)

def midi1_up_control_change(int group, int channel, int index, int data):
    """Create a MIDI 1.0 Universal Packet Control Change message.

    Args:
        group: MIDI group (0-15)
        channel: MIDI channel (0-15)
        index: Controller number (0-127)
        data: Controller value (0-127)

    Returns:
        32-bit MIDI message
    """
    return cm.MIDI1UPControlChange(<cf.UInt8>group, <cf.UInt8>channel,
                                     <cf.UInt8>index, <cf.UInt8>data)

def midi1_up_pitch_bend(int group, int channel, int lsb, int msb):
    """Create a MIDI 1.0 Universal Packet Pitch Bend message.

    Args:
        group: MIDI group (0-15)
        channel: MIDI channel (0-15)
        lsb: Pitch bend LSB (0-127)
        msb: Pitch bend MSB (0-127)

    Returns:
        32-bit MIDI message
    """
    return cm.MIDI1UPPitchBend(<cf.UInt8>group, <cf.UInt8>channel,
                                 <cf.UInt8>lsb, <cf.UInt8>msb)

def midi1_up_system_common(int group, int status, int byte1, int byte2):
    """Create a MIDI 1.0 Universal Packet System Common message.

    Args:
        group: MIDI group (0-15)
        status: System status byte
        byte1: First data byte
        byte2: Second data byte

    Returns:
        32-bit MIDI message
    """
    return cm.MIDI1UPSystemCommon(<cf.UInt8>group, <cf.UInt8>status,
                                    <cf.UInt8>byte1, <cf.UInt8>byte2)

def midi1_up_sysex(int group, int status, int bytes_used, int byte1, int byte2, int byte3, int byte4, int byte5, int byte6):
    """Create a MIDI 1.0 Universal Packet SysEx message.

    Args:
        group: MIDI group (0-15)
        status: SysEx status nibble
        bytes_used: Number of data bytes used (0-6)
        byte1-byte6: SysEx data bytes

    Returns:
        Tuple of (word0, word1) for 64-bit MIDI message
    """
    cdef cm.MIDIMessage_64 msg = cm.MIDI1UPSysEx(<cf.UInt8>group, <cf.UInt8>status, <cf.UInt8>bytes_used,
                                                      <cf.UInt8>byte1, <cf.UInt8>byte2, <cf.UInt8>byte3,
                                                      <cf.UInt8>byte4, <cf.UInt8>byte5, <cf.UInt8>byte6)
    return (msg.word0, msg.word1)

# MIDI 2.0 Channel Voice Message Functions

def midi2_channel_voice_message(int group, int status, int channel, int index, long value):
    """Create a MIDI 2.0 Channel Voice message.

    Args:
        group: MIDI group (0-15)
        status: MIDI status nibble
        channel: MIDI channel (0-15)
        index: 16-bit index value
        value: 32-bit data value

    Returns:
        Tuple of (word0, word1) for 64-bit MIDI message
    """
    cdef cm.MIDIMessage_64 msg = cm.MIDI2ChannelVoiceMessage(<cf.UInt8>group, <cf.UInt8>status,
                                                                  <cf.UInt8>channel, <ca.UInt16>index, <cf.UInt32>value)
    return (msg.word0, msg.word1)

def midi2_note_on(int group, int channel, int note_number, int attribute_type, int attribute_data, int velocity):
    """Create a MIDI 2.0 Note On message.

    Args:
        group: MIDI group (0-15)
        channel: MIDI channel (0-15)
        note_number: Note number (0-127)
        attribute_type: Note attribute type
        attribute_data: Note attribute data (16-bit)
        velocity: Note velocity (16-bit)

    Returns:
        Tuple of (word0, word1) for 64-bit MIDI message
    """
    cdef cm.MIDIMessage_64 msg = cm.MIDI2NoteOn(<cf.UInt8>group, <cf.UInt8>channel, <cf.UInt8>note_number,
                                                     <cf.UInt8>attribute_type, <ca.UInt16>attribute_data, <ca.UInt16>velocity)
    return (msg.word0, msg.word1)

def midi2_note_off(int group, int channel, int note_number, int attribute_type, int attribute_data, int velocity):
    """Create a MIDI 2.0 Note Off message.

    Args:
        group: MIDI group (0-15)
        channel: MIDI channel (0-15)
        note_number: Note number (0-127)
        attribute_type: Note attribute type
        attribute_data: Note attribute data (16-bit)
        velocity: Note velocity (16-bit)

    Returns:
        Tuple of (word0, word1) for 64-bit MIDI message
    """
    cdef cm.MIDIMessage_64 msg = cm.MIDI2NoteOff(<cf.UInt8>group, <cf.UInt8>channel, <cf.UInt8>note_number,
                                                      <cf.UInt8>attribute_type, <ca.UInt16>attribute_data, <ca.UInt16>velocity)
    return (msg.word0, msg.word1)

def midi2_control_change(int group, int channel, int index, long value):
    """Create a MIDI 2.0 Control Change message.

    Args:
        group: MIDI group (0-15)
        channel: MIDI channel (0-15)
        index: Controller number
        value: Controller value (32-bit)

    Returns:
        Tuple of (word0, word1) for 64-bit MIDI message
    """
    cdef cm.MIDIMessage_64 msg = cm.MIDI2ControlChange(<cf.UInt8>group, <cf.UInt8>channel,
                                                            <cf.UInt8>index, <cf.UInt32>value)
    return (msg.word0, msg.word1)

def midi2_program_change(int group, int channel, bint bank_is_valid, int program, int bank_msb, int bank_lsb):
    """Create a MIDI 2.0 Program Change message.

    Args:
        group: MIDI group (0-15)
        channel: MIDI channel (0-15)
        bank_is_valid: Whether bank change is included
        program: Program number
        bank_msb: Bank MSB
        bank_lsb: Bank LSB

    Returns:
        Tuple of (word0, word1) for 64-bit MIDI message
    """
    cdef cm.MIDIMessage_64 msg = cm.MIDI2ProgramChange(<cf.UInt8>group, <cf.UInt8>channel, bank_is_valid,
                                                            <cf.UInt8>program, <cf.UInt8>bank_msb, <cf.UInt8>bank_lsb)
    return (msg.word0, msg.word1)

def midi2_pitch_bend(int group, int channel, long value):
    """Create a MIDI 2.0 Pitch Bend message.

    Args:
        group: MIDI group (0-15)
        channel: MIDI channel (0-15)
        value: Pitch bend value (32-bit)

    Returns:
        Tuple of (word0, word1) for 64-bit MIDI message
    """
    cdef cm.MIDIMessage_64 msg = cm.MIDI2PitchBend(<cf.UInt8>group, <cf.UInt8>channel, <cf.UInt32>value)
    return (msg.word0, msg.word1)

# MIDI Message Type Constants

def get_midi_message_type_utility():
    """Get the Utility message type constant."""
    return cm.kMIDIMessageTypeUtility

def get_midi_message_type_system():
    """Get the System message type constant."""
    return cm.kMIDIMessageTypeSystem

def get_midi_message_type_channel_voice1():
    """Get the Channel Voice 1 (MIDI 1.0) message type constant."""
    return cm.kMIDIMessageTypeChannelVoice1

def get_midi_message_type_sysex():
    """Get the SysEx message type constant."""
    return cm.kMIDIMessageTypeSysEx

def get_midi_message_type_channel_voice2():
    """Get the Channel Voice 2 (MIDI 2.0) message type constant."""
    return cm.kMIDIMessageTypeChannelVoice2

def get_midi_message_type_data128():
    """Get the Data128 message type constant."""
    return cm.kMIDIMessageTypeData128

# MIDI CV Status Constants

def get_midi_cv_status_note_off():
    """Get the Note Off status constant."""
    return cm.kMIDICVStatusNoteOff

def get_midi_cv_status_note_on():
    """Get the Note On status constant."""
    return cm.kMIDICVStatusNoteOn

def get_midi_cv_status_poly_pressure():
    """Get the Poly Pressure status constant."""
    return cm.kMIDICVStatusPolyPressure

def get_midi_cv_status_control_change():
    """Get the Control Change status constant."""
    return cm.kMIDICVStatusControlChange

def get_midi_cv_status_program_change():
    """Get the Program Change status constant."""
    return cm.kMIDICVStatusProgramChange

def get_midi_cv_status_channel_pressure():
    """Get the Channel Pressure status constant."""
    return cm.kMIDICVStatusChannelPressure

def get_midi_cv_status_pitch_bend():
    """Get the Pitch Bend status constant."""
    return cm.kMIDICVStatusPitchBend


# MIDI Setup (Device and Entity Management) Functions

def midi_device_new_entity(long device, str name, int protocol, bint embedded, int num_source_endpoints, int num_destination_endpoints):
    """Create a new entity for a MIDI device (macOS 11.0+, iOS 14.0+).

    Args:
        device: The MIDIDeviceRef to add an entity to
        name: Name of the new entity
        protocol: MIDI protocol ID (1 for MIDI 1.0, 2 for MIDI 2.0)
        embedded: True if entity is inside device, False if external connectors
        num_source_endpoints: Number of source endpoints for the entity
        num_destination_endpoints: Number of destination endpoints for the entity

    Returns:
        The new MIDIEntityRef

    Raises:
        RuntimeError: If entity creation fails
    """
    cdef cm.MIDIEntityRef entity
    cdef cf.CFStringRef cf_name
    cdef bytes name_bytes = name.encode('utf-8')

    cf_name = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault,
        name_bytes,
        cf.kCFStringEncodingUTF8
    )

    cdef cf.OSStatus status
    try:
        status = cm.MIDIDeviceNewEntity(
            <cm.MIDIDeviceRef>device,
            cf_name,
            <cm.MIDIProtocolID>protocol,
            <cf.Boolean>embedded,
            <cm.ItemCount>num_source_endpoints,
            <cm.ItemCount>num_destination_endpoints,
            &entity
        )

        if status != 0:
            raise RuntimeError(f"MIDIDeviceNewEntity failed with status: {status}")

        return entity

    finally:
        if cf_name:
            cf.CFRelease(cf_name)

def midi_device_add_entity(long device, str name, bint embedded, int num_source_endpoints, int num_destination_endpoints):
    """Add an entity to a MIDI device (deprecated, use midi_device_new_entity).

    Args:
        device: The MIDIDeviceRef to add an entity to
        name: Name of the new entity
        embedded: True if entity is inside device, False if external connectors
        num_source_endpoints: Number of source endpoints for the entity
        num_destination_endpoints: Number of destination endpoints for the entity

    Returns:
        The new MIDIEntityRef

    Raises:
        RuntimeError: If entity creation fails
    """
    cdef cm.MIDIEntityRef entity
    cdef cf.CFStringRef cf_name
    cdef bytes name_bytes = name.encode('utf-8')

    cf_name = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault,
        name_bytes,
        cf.kCFStringEncodingUTF8
    )

    cdef cf.OSStatus status
    try:
        status = cm.MIDIDeviceAddEntity(
            <cm.MIDIDeviceRef>device,
            cf_name,
            <cf.Boolean>embedded,
            <cm.ItemCount>num_source_endpoints,
            <cm.ItemCount>num_destination_endpoints,
            &entity
        )

        if status != 0:
            raise RuntimeError(f"MIDIDeviceAddEntity failed with status: {status}")

        return entity

    finally:
        if cf_name:
            cf.CFRelease(cf_name)

def midi_device_remove_entity(long device, long entity):
    """Remove an entity from a MIDI device.

    Args:
        device: The MIDIDeviceRef to remove entity from
        entity: The MIDIEntityRef to remove

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If entity removal fails
    """
    cdef cf.OSStatus status = cm.MIDIDeviceRemoveEntity(
        <cm.MIDIDeviceRef>device,
        <cm.MIDIEntityRef>entity
    )

    if status != 0:
        raise RuntimeError(f"MIDIDeviceRemoveEntity failed with status: {status}")

    return status

def midi_entity_add_or_remove_endpoints(long entity, int num_source_endpoints, int num_destination_endpoints):
    """Add or remove endpoints from a MIDI entity.

    Args:
        entity: The MIDIEntityRef to modify
        num_source_endpoints: Desired number of source endpoints
        num_destination_endpoints: Desired number of destination endpoints

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If endpoint modification fails
    """
    cdef cf.OSStatus status = cm.MIDIEntityAddOrRemoveEndpoints(
        <cm.MIDIEntityRef>entity,
        <cm.ItemCount>num_source_endpoints,
        <cm.ItemCount>num_destination_endpoints
    )

    if status != 0:
        raise RuntimeError(f"MIDIEntityAddOrRemoveEndpoints failed with status: {status}")

    return status

def midi_setup_add_device(long device):
    """Add a driver-owned MIDI device to the current setup.

    Args:
        device: The MIDIDeviceRef to add

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If device addition fails
    """
    cdef cf.OSStatus status = cm.MIDISetupAddDevice(<cm.MIDIDeviceRef>device)

    if status != 0:
        raise RuntimeError(f"MIDISetupAddDevice failed with status: {status}")

    return status

def midi_setup_remove_device(long device):
    """Remove a driver-owned MIDI device from the current setup.

    Args:
        device: The MIDIDeviceRef to remove

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If device removal fails
    """
    cdef cf.OSStatus status = cm.MIDISetupRemoveDevice(<cm.MIDIDeviceRef>device)

    if status != 0:
        raise RuntimeError(f"MIDISetupRemoveDevice failed with status: {status}")

    return status

def midi_setup_add_external_device(long device):
    """Add an external MIDI device to the current setup.

    Args:
        device: The MIDIDeviceRef to add

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If external device addition fails
    """
    cdef cf.OSStatus status = cm.MIDISetupAddExternalDevice(<cm.MIDIDeviceRef>device)

    if status != 0:
        raise RuntimeError(f"MIDISetupAddExternalDevice failed with status: {status}")

    return status

def midi_setup_remove_external_device(long device):
    """Remove an external MIDI device from the current setup.

    Args:
        device: The MIDIDeviceRef to remove

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If external device removal fails
    """
    cdef cf.OSStatus status = cm.MIDISetupRemoveExternalDevice(<cm.MIDIDeviceRef>device)

    if status != 0:
        raise RuntimeError(f"MIDISetupRemoveExternalDevice failed with status: {status}")

    return status

def midi_external_device_create(str name, str manufacturer, str model):
    """Create a new external MIDI device.

    Args:
        name: Name of the device
        manufacturer: Manufacturer name
        model: Model name

    Returns:
        The new MIDIDeviceRef

    Raises:
        RuntimeError: If device creation fails
    """
    cdef cm.MIDIDeviceRef device
    cdef cf.CFStringRef cf_name, cf_manufacturer, cf_model
    cdef bytes name_bytes = name.encode('utf-8')
    cdef bytes manufacturer_bytes = manufacturer.encode('utf-8')
    cdef bytes model_bytes = model.encode('utf-8')

    cf_name = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault, name_bytes, cf.kCFStringEncodingUTF8
    )
    cf_manufacturer = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault, manufacturer_bytes, cf.kCFStringEncodingUTF8
    )
    cf_model = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault, model_bytes, cf.kCFStringEncodingUTF8
    )

    cdef cf.OSStatus status
    try:
        status = cm.MIDIExternalDeviceCreate(
            cf_name, cf_manufacturer, cf_model, &device
        )

        if status != 0:
            raise RuntimeError(f"MIDIExternalDeviceCreate failed with status: {status}")

        return device

    finally:
        if cf_name:
            cf.CFRelease(cf_name)
        if cf_manufacturer:
            cf.CFRelease(cf_manufacturer)
        if cf_model:
            cf.CFRelease(cf_model)


# MIDI Driver Functions

def midi_device_create(str name, str manufacturer, str model):
    """Create a new MIDI device (available to non-drivers).

    Args:
        name: Name of the device
        manufacturer: Manufacturer name
        model: Model name

    Returns:
        The new MIDIDeviceRef

    Raises:
        RuntimeError: If device creation fails
    """
    cdef cm.MIDIDeviceRef device
    cdef cf.CFStringRef cf_name, cf_manufacturer, cf_model
    cdef bytes name_bytes = name.encode('utf-8')
    cdef bytes manufacturer_bytes = manufacturer.encode('utf-8')
    cdef bytes model_bytes = model.encode('utf-8')

    cf_name = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault, name_bytes, cf.kCFStringEncodingUTF8
    )
    cf_manufacturer = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault, manufacturer_bytes, cf.kCFStringEncodingUTF8
    )
    cf_model = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault, model_bytes, cf.kCFStringEncodingUTF8
    )

    cdef cf.OSStatus status
    try:
        # NULL owner indicates non-driver creation
        status = cm.MIDIDeviceCreate(
            <cm.MIDIDriverRef>NULL,
            cf_name, cf_manufacturer, cf_model, &device
        )

        if status != 0:
            raise RuntimeError(f"MIDIDeviceCreate failed with status: {status}")

        return device

    finally:
        if cf_name:
            cf.CFRelease(cf_name)
        if cf_manufacturer:
            cf.CFRelease(cf_manufacturer)
        if cf_model:
            cf.CFRelease(cf_model)

def midi_device_dispose(long device):
    """Dispose a MIDI device that hasn't been added to the system.

    Args:
        device: The MIDIDeviceRef to dispose

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If device disposal fails
    """
    cdef cf.OSStatus status = cm.MIDIDeviceDispose(<cm.MIDIDeviceRef>device)

    if status != 0:
        raise RuntimeError(f"MIDIDeviceDispose failed with status: {status}")

    return status

def midi_device_list_get_number_of_devices(long dev_list):
    """Get the number of devices in a device list.

    Args:
        dev_list: The MIDIDeviceListRef

    Returns:
        Number of devices in the list
    """
    return cm.MIDIDeviceListGetNumberOfDevices(<cm.MIDIDeviceListRef>dev_list)

def midi_device_list_get_device(long dev_list, int index):
    """Get a device from a device list.

    Args:
        dev_list: The MIDIDeviceListRef
        index: Index of the device to retrieve (0-based)

    Returns:
        The MIDIDeviceRef at the specified index

    Raises:
        IndexError: If index is out of bounds
    """
    cdef cm.ItemCount num_devices = cm.MIDIDeviceListGetNumberOfDevices(<cm.MIDIDeviceListRef>dev_list)

    if index < 0 or index >= num_devices:
        raise IndexError(f"Device index {index} out of bounds (0-{num_devices-1})")

    cdef cm.MIDIDeviceRef device = cm.MIDIDeviceListGetDevice(
        <cm.MIDIDeviceListRef>dev_list, <cm.ItemCount>index
    )

    if device == 0:
        raise RuntimeError(f"Failed to get device at index {index}")

    return device

def midi_device_list_add_device(long dev_list, long device):
    """Add a device to a device list.

    Args:
        dev_list: The MIDIDeviceListRef
        device: The MIDIDeviceRef to add

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If adding device fails
    """
    cdef cf.OSStatus status = cm.MIDIDeviceListAddDevice(
        <cm.MIDIDeviceListRef>dev_list, <cm.MIDIDeviceRef>device
    )

    if status != 0:
        raise RuntimeError(f"MIDIDeviceListAddDevice failed with status: {status}")

    return status

def midi_device_list_dispose(long dev_list):
    """Dispose a device list (but not the contained devices).

    Args:
        dev_list: The MIDIDeviceListRef to dispose

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If disposal fails
    """
    cdef cf.OSStatus status = cm.MIDIDeviceListDispose(<cm.MIDIDeviceListRef>dev_list)

    if status != 0:
        raise RuntimeError(f"MIDIDeviceListDispose failed with status: {status}")

    return status

def midi_endpoint_set_ref_cons(long endpoint, long ref1=0, long ref2=0):
    """Set reference constants for a MIDI endpoint.

    Args:
        endpoint: The MIDIEndpointRef
        ref1: First reference constant (optional)
        ref2: Second reference constant (optional)

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If setting refCons fails
    """
    cdef cf.OSStatus status = cm.MIDIEndpointSetRefCons(
        <cm.MIDIEndpointRef>endpoint,
        <void*>ref1,
        <void*>ref2
    )

    if status != 0:
        raise RuntimeError(f"MIDIEndpointSetRefCons failed with status: {status}")

    return status

def midi_endpoint_get_ref_cons(long endpoint):
    """Get reference constants for a MIDI endpoint.

    Args:
        endpoint: The MIDIEndpointRef

    Returns:
        Tuple of (ref1, ref2) as integers

    Raises:
        RuntimeError: If getting refCons fails
    """
    cdef void* ref1
    cdef void* ref2
    cdef cf.OSStatus status = cm.MIDIEndpointGetRefCons(
        <cm.MIDIEndpointRef>endpoint, &ref1, &ref2
    )

    if status != 0:
        raise RuntimeError(f"MIDIEndpointGetRefCons failed with status: {status}")

    return (<long>ref1, <long>ref2)

def midi_get_driver_io_runloop():
    """Get the driver I/O run loop.

    Returns:
        CFRunLoopRef as an integer (for advanced use)

    Note:
        This is primarily used by MIDI drivers for high-priority I/O operations.
    """
    cdef ca.CFRunLoopRef runloop = cm.MIDIGetDriverIORunLoop()
    return <long>runloop

def midi_get_driver_device_list(long driver):
    """Get the device list for a specific driver.

    Args:
        driver: The MIDIDriverRef

    Returns:
        MIDIDeviceListRef containing devices owned by the driver

    Note:
        The returned device list should be disposed using midi_device_list_dispose().
        This function is primarily useful for driver development.
    """
    cdef cm.MIDIDeviceListRef dev_list = cm.MIDIGetDriverDeviceList(<cm.MIDIDriverRef>driver)
    return dev_list

def midi_driver_enable_monitoring(long driver, bint enabled):
    """Enable or disable MIDI monitoring for a driver.

    Args:
        driver: The MIDIDriverRef
        enabled: True to enable monitoring, False to disable

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If enabling/disabling monitoring fails

    Note:
        This allows a driver to monitor all outgoing MIDI packets in the system.
        Primarily used for specialized drivers like MIDI monitor displays.
    """
    cdef cf.OSStatus status = cm.MIDIDriverEnableMonitoring(
        <cm.MIDIDriverRef>driver, <cf.Boolean>enabled
    )

    if status != 0:
        raise RuntimeError(f"MIDIDriverEnableMonitoring failed with status: {status}")

    return status


# MIDI Thru Connection Functions

def midi_thru_connection_params_initialize():
    """Initialize a MIDIThruConnectionParams structure with default values.

    Returns:
        Dictionary containing the default thru connection parameters

    Note:
        This creates a basic structure with no endpoints and no transformations.
        You can then modify the returned dictionary and use it with other functions.
    """
    cdef cm.MIDIThruConnectionParams params
    cm.MIDIThruConnectionParamsInitialize(&params)

    # Convert to Python dictionary for easier manipulation
    result = {
        'version': params.version,
        'sources': [],
        'destinations': [],
        'channelMap': [params.channelMap[i] for i in range(16)],
        'lowVelocity': params.lowVelocity,
        'highVelocity': params.highVelocity,
        'lowNote': params.lowNote,
        'highNote': params.highNote,
        'noteNumber': {'transform': params.noteNumber.transform, 'param': params.noteNumber.param},
        'velocity': {'transform': params.velocity.transform, 'param': params.velocity.param},
        'keyPressure': {'transform': params.keyPressure.transform, 'param': params.keyPressure.param},
        'channelPressure': {'transform': params.channelPressure.transform, 'param': params.channelPressure.param},
        'programChange': {'transform': params.programChange.transform, 'param': params.programChange.param},
        'pitchBend': {'transform': params.pitchBend.transform, 'param': params.pitchBend.param},
        'filterOutSysEx': params.filterOutSysEx,
        'filterOutMTC': params.filterOutMTC,
        'filterOutBeatClock': params.filterOutBeatClock,
        'filterOutTuneRequest': params.filterOutTuneRequest,
        'filterOutAllControls': params.filterOutAllControls,
        'controlTransforms': [],
        'valueMaps': []
    }

    return result

def midi_thru_connection_create(str persistent_owner_id=None, dict connection_params=None):
    """Create a MIDI thru connection.

    Args:
        persistent_owner_id: If provided, connection persists; if None, owned by client
        connection_params: Dictionary with connection parameters (use midi_thru_connection_params_initialize())

    Returns:
        MIDIThruConnectionRef

    Raises:
        RuntimeError: If connection creation fails
    """
    if connection_params is None:
        connection_params = midi_thru_connection_params_initialize()

    # Convert Python dict back to C structure
    cdef cm.MIDIThruConnectionParams params
    cdef int note_transform
    cdef int velocity_transform
    cm.MIDIThruConnectionParamsInitialize(&params)

    # Fill in the structure from the dictionary
    params.version = connection_params.get('version', 0)

    # Sources
    sources = connection_params.get('sources', [])
    params.numSources = min(len(sources), 8)
    for i in range(params.numSources):
        if isinstance(sources[i], dict):
            params.sources[i].endpointRef = <cm.MIDIEndpointRef>sources[i].get('endpointRef', 0)
            params.sources[i].uniqueID = <cm.MIDIUniqueID>sources[i].get('uniqueID', 0)

    # Destinations
    destinations = connection_params.get('destinations', [])
    params.numDestinations = min(len(destinations), 8)
    for i in range(params.numDestinations):
        if isinstance(destinations[i], dict):
            params.destinations[i].endpointRef = <cm.MIDIEndpointRef>destinations[i].get('endpointRef', 0)
            params.destinations[i].uniqueID = <cm.MIDIUniqueID>destinations[i].get('uniqueID', 0)

    # Channel map
    channel_map = connection_params.get('channelMap', list(range(16)))
    for i in range(16):
        params.channelMap[i] = <cf.UInt8>channel_map[i] if i < len(channel_map) else <cf.UInt8>i

    # Velocity and note filtering
    params.lowVelocity = <cf.UInt8>connection_params.get('lowVelocity', 0)
    params.highVelocity = <cf.UInt8>connection_params.get('highVelocity', 0)
    params.lowNote = <cf.UInt8>connection_params.get('lowNote', 0)
    params.highNote = <cf.UInt8>connection_params.get('highNote', 127)

    # Transform settings
    note_number = connection_params.get('noteNumber', {'transform': 0, 'param': 0})
    note_transform = note_number.get('transform', 0)
    params.noteNumber.transform = <cm.MIDITransformType>note_transform
    params.noteNumber.param = <ca.SInt16>note_number.get('param', 0)

    velocity = connection_params.get('velocity', {'transform': 0, 'param': 0})
    velocity_transform = velocity.get('transform', 0)
    params.velocity.transform = <cm.MIDITransformType>velocity_transform
    params.velocity.param = <ca.SInt16>velocity.get('param', 0)

    # Filter settings
    params.filterOutSysEx = <cf.UInt8>connection_params.get('filterOutSysEx', 0)
    params.filterOutMTC = <cf.UInt8>connection_params.get('filterOutMTC', 0)
    params.filterOutBeatClock = <cf.UInt8>connection_params.get('filterOutBeatClock', 0)
    params.filterOutTuneRequest = <cf.UInt8>connection_params.get('filterOutTuneRequest', 0)
    params.filterOutAllControls = <cf.UInt8>connection_params.get('filterOutAllControls', 0)

    # Note: For simplicity, we're not implementing the variable-length portions
    # (control transforms and value maps) in this basic wrapper
    params.numControlTransforms = 0
    params.numMaps = 0

    # Create CFData from the structure
    cdef cf.CFDataRef cf_params = ca.CFDataCreate(
        cf.kCFAllocatorDefault,
        <cf.UInt8*>&params,
        sizeof(cm.MIDIThruConnectionParams)
    )

    cdef cf.CFStringRef cf_owner_id = NULL
    cdef bytes owner_id_bytes
    if persistent_owner_id is not None:
        owner_id_bytes = persistent_owner_id.encode('utf-8')
        cf_owner_id = cf.CFStringCreateWithCString(
            cf.kCFAllocatorDefault, owner_id_bytes, cf.kCFStringEncodingUTF8
        )

    cdef cm.MIDIThruConnectionRef connection
    cdef cf.OSStatus status

    try:
        status = cm.MIDIThruConnectionCreate(cf_owner_id, cf_params, &connection)

        if status != 0:
            raise RuntimeError(f"MIDIThruConnectionCreate failed with status: {status}")

        return connection

    finally:
        if cf_params:
            cf.CFRelease(cf_params)
        if cf_owner_id:
            cf.CFRelease(cf_owner_id)

def midi_thru_connection_dispose(long connection):
    """Dispose a MIDI thru connection.

    Args:
        connection: The MIDIThruConnectionRef to dispose

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If disposal fails
    """
    cdef cf.OSStatus status = cm.MIDIThruConnectionDispose(<cm.MIDIThruConnectionRef>connection)

    if status != 0:
        raise RuntimeError(f"MIDIThruConnectionDispose failed with status: {status}")

    return status

def midi_thru_connection_get_params(long connection):
    """Get the parameters of a MIDI thru connection.

    Args:
        connection: The MIDIThruConnectionRef

    Returns:
        Dictionary containing the connection parameters

    Raises:
        RuntimeError: If getting parameters fails
    """
    cdef cf.CFDataRef cf_params
    cdef cf.OSStatus status = cm.MIDIThruConnectionGetParams(
        <cm.MIDIThruConnectionRef>connection, &cf_params
    )

    if status != 0:
        raise RuntimeError(f"MIDIThruConnectionGetParams failed with status: {status}")

    cdef cf.CFIndex data_length
    cdef cf.UInt8* data_ptr
    cdef cm.MIDIThruConnectionParams* params

    try:
        # Extract the data
        data_length = ca.CFDataGetLength(cf_params)
        data_ptr = <cf.UInt8*>ca.CFDataGetBytePtr(cf_params)

        if data_length < sizeof(cm.MIDIThruConnectionParams):
            raise RuntimeError("Invalid connection parameters data")

        params = <cm.MIDIThruConnectionParams*>data_ptr

        # Convert to Python dictionary
        result = {
            'version': params.version,
            'sources': [],
            'destinations': [],
            'channelMap': [params.channelMap[i] for i in range(16)],
            'lowVelocity': params.lowVelocity,
            'highVelocity': params.highVelocity,
            'lowNote': params.lowNote,
            'highNote': params.highNote,
            'noteNumber': {'transform': params.noteNumber.transform, 'param': params.noteNumber.param},
            'velocity': {'transform': params.velocity.transform, 'param': params.velocity.param},
            'keyPressure': {'transform': params.keyPressure.transform, 'param': params.keyPressure.param},
            'channelPressure': {'transform': params.channelPressure.transform, 'param': params.channelPressure.param},
            'programChange': {'transform': params.programChange.transform, 'param': params.programChange.param},
            'pitchBend': {'transform': params.pitchBend.transform, 'param': params.pitchBend.param},
            'filterOutSysEx': params.filterOutSysEx,
            'filterOutMTC': params.filterOutMTC,
            'filterOutBeatClock': params.filterOutBeatClock,
            'filterOutTuneRequest': params.filterOutTuneRequest,
            'filterOutAllControls': params.filterOutAllControls,
            'numControlTransforms': params.numControlTransforms,
            'numMaps': params.numMaps
        }

        # Add sources
        for i in range(params.numSources):
            source = {
                'endpointRef': params.sources[i].endpointRef,
                'uniqueID': params.sources[i].uniqueID
            }
            result['sources'].append(source)

        # Add destinations
        for i in range(params.numDestinations):
            dest = {
                'endpointRef': params.destinations[i].endpointRef,
                'uniqueID': params.destinations[i].uniqueID
            }
            result['destinations'].append(dest)

        return result

    finally:
        if cf_params:
            cf.CFRelease(cf_params)

def midi_thru_connection_set_params(long connection, dict connection_params):
    """Set the parameters of a MIDI thru connection.

    Args:
        connection: The MIDIThruConnectionRef
        connection_params: Dictionary with new connection parameters

    Returns:
      f OSStatus result code

    Raises:
        RuntimeError: If setting parameters fails
    """
    # Convert Python dict to C structure (similar to create function)
    cdef cm.MIDIThruConnectionParams params
    cm.MIDIThruConnectionParamsInitialize(&params)

    # Fill in the structure from the dictionary (abbreviated version)
    params.version = connection_params.get('version', 0)

    # Sources
    sources = connection_params.get('sources', [])
    params.numSources = min(len(sources), 8)
    for i in range(params.numSources):
        if isinstance(sources[i], dict):
            params.sources[i].endpointRef = <cm.MIDIEndpointRef>sources[i].get('endpointRef', 0)
            params.sources[i].uniqueID = <cm.MIDIUniqueID>sources[i].get('uniqueID', 0)

    # Destinations
    destinations = connection_params.get('destinations', [])
    params.numDestinations = min(len(destinations), 8)
    for i in range(params.numDestinations):
        if isinstance(destinations[i], dict):
            params.destinations[i].endpointRef = <cm.MIDIEndpointRef>destinations[i].get('endpointRef', 0)
            params.destinations[i].uniqueID = <cm.MIDIUniqueID>destinations[i].get('uniqueID', 0)

    # Basic parameters
    params.filterOutSysEx = <cf.UInt8>connection_params.get('filterOutSysEx', 0)
    params.filterOutMTC = <cf.UInt8>connection_params.get('filterOutMTC', 0)
    params.filterOutBeatClock = <cf.UInt8>connection_params.get('filterOutBeatClock', 0)
    params.filterOutAllControls = <cf.UInt8>connection_params.get('filterOutAllControls', 0)

    # Create CFData from the structure
    cdef cf.CFDataRef cf_params = ca.CFDataCreate(
        cf.kCFAllocatorDefault,
        <cf.UInt8*>&params,
        sizeof(cm.MIDIThruConnectionParams)
    )

    cdef cf.OSStatus status
    try:
        status = cm.MIDIThruConnectionSetParams(
            <cm.MIDIThruConnectionRef>connection, cf_params
        )

        if status != 0:
            raise RuntimeError(f"MIDIThruConnectionSetParams failed with status: {status}")

        return status

    finally:
        if cf_params:
            cf.CFRelease(cf_params)

def midi_thru_connection_find(str persistent_owner_id):
    """Find all thru connections created by a specific owner.

    Args:
        persistent_owner_id: The ID of the owner whose connections to find

    Returns:
        List of MIDIThruConnectionRef values

    Raises:
        RuntimeError: If finding connections fails
    """
    cdef bytes owner_id_bytes = persistent_owner_id.encode('utf-8')
    cdef cf.CFStringRef cf_owner_id = cf.CFStringCreateWithCString(
        cf.kCFAllocatorDefault, owner_id_bytes, cf.kCFStringEncodingUTF8
    )

    cdef cf.CFDataRef cf_connection_list
    cdef cf.OSStatus status
    cdef cf.CFIndex data_length
    cdef cf.UInt8* data_ptr
    cdef cf.CFIndex num_connections
    cdef cm.MIDIThruConnectionRef* connections

    try:
        status = cm.MIDIThruConnectionFind(cf_owner_id, &cf_connection_list)

        if status != 0:
            raise RuntimeError(f"MIDIThruConnectionFind failed with status: {status}")

        # Extract the connection list
        data_length = ca.CFDataGetLength(cf_connection_list)
        data_ptr = <cf.UInt8*>ca.CFDataGetBytePtr(cf_connection_list)

        # Each connection is a MIDIThruConnectionRef (which is a MIDIObjectRef)
        num_connections = data_length // sizeof(cm.MIDIThruConnectionRef)
        connections = <cm.MIDIThruConnectionRef*>data_ptr

        result = []
        for i in range(num_connections):
            result.append(connections[i])

        return result

    finally:
        if cf_owner_id:
            cf.CFRelease(cf_owner_id)
        if cf_connection_list:
            cf.CFRelease(cf_connection_list)

# MIDI Thru Connection Constants

def get_midi_transform_none():
    """Get the 'None' transform type constant."""
    return cm.kMIDITransform_None

def get_midi_transform_filter_out():
    """Get the 'FilterOut' transform type constant."""
    return cm.kMIDITransform_FilterOut

def get_midi_transform_map_control():
    """Get the 'MapControl' transform type constant."""
    return cm.kMIDITransform_MapControl

def get_midi_transform_add():
    """Get the 'Add' transform type constant."""
    return cm.kMIDITransform_Add

def get_midi_transform_scale():
    """Get the 'Scale' transform type constant."""
    return cm.kMIDITransform_Scale

def get_midi_transform_min_value():
    """Get the 'MinValue' transform type constant."""
    return cm.kMIDITransform_MinValue

def get_midi_transform_max_value():
    """Get the 'MaxValue' transform type constant."""
    return cm.kMIDITransform_MaxValue

def get_midi_transform_map_value():
    """Get the 'MapValue' transform type constant."""
    return cm.kMIDITransform_MapValue

def get_midi_control_type_7bit():
    """Get the '7Bit' control type constant."""
    return cm.kMIDIControlType_7Bit

def get_midi_control_type_14bit():
    """Get the '14Bit' control type constant."""
    return cm.kMIDIControlType_14Bit

def get_midi_control_type_7bit_rpn():
    """Get the '7BitRPN' control type constant."""
    return cm.kMIDIControlType_7BitRPN

def get_midi_control_type_14bit_rpn():
    """Get the '14BitRPN' control type constant."""
    return cm.kMIDIControlType_14BitRPN

def get_midi_control_type_7bit_nrpn():
    """Get the '7BitNRPN' control type constant."""
    return cm.kMIDIControlType_7BitNRPN

def get_midi_control_type_14bit_nrpn():
    """Get the '14BitNRPN' control type constant."""
    return cm.kMIDIControlType_14BitNRPN

def get_midi_thru_connection_max_endpoints():
    """Get the maximum number of endpoints for a thru connection."""
    return 8  # kMIDIThruConnection_MaxEndpoints


# ============================================================================
# Cython Extension Classes
# ============================================================================

# Cython extension class for automatic resource management in coremusic.

# The CoreAudioObject base class has automatic resource cleanup via __dealloc__.
# All other classes are implemented as pure Python classes in the objects module.

cdef class CoreAudioObject:
    """Base class for all CoreAudio objects with automatic resource management

    This is the only Cython extension class, providing __dealloc__ for automatic
    cleanup. All other classes are implemented as pure Python classes.
    """

    cdef long _object_id
    cdef bint _is_disposed

    def __cinit__(self):
        self._object_id = 0
        self._is_disposed = False

    def __dealloc__(self):
        """Automatic cleanup when object is garbage collected"""
        if not self._is_disposed:
            self._dispose_internal()

    @property
    def is_disposed(self) -> bool:
        """Check if the object has been disposed"""
        return self._is_disposed

    def dispose(self) -> None:
        """Explicitly dispose of the object's resources"""
        if not self._is_disposed:
            self._dispose_internal()

    cdef void _dispose_internal(self):
        """Internal disposal implementation"""
        self._is_disposed = True
        self._object_id = 0

    def _ensure_not_disposed(self) -> None:
        """Ensure the object has not been disposed"""
        if self._is_disposed:
            raise RuntimeError(f"{self.__class__.__name__} has been disposed")

    @property
    def object_id(self) -> int:
        """Expose object ID for testing purposes"""
        return self._object_id

    def _set_object_id(self, object_id: int) -> None:
        """Set object ID (for internal use)"""
        self._object_id = object_id


# ============================================================================
# AUGraph API
# ============================================================================

def au_graph_new() -> int:
    """Create a new AUGraph

    Returns:
        AUGraph ID

    Raises:
        RuntimeError: If graph creation fails
    """
    cdef at.AUGraph graph
    cdef cf.OSStatus status

    status = at.NewAUGraph(&graph)
    if status != 0:
        raise RuntimeError(f"NewAUGraph failed with status: {status}")

    return <long>graph


def au_graph_dispose(long graph_id):
    """Dispose an AUGraph

    Args:
        graph_id: AUGraph ID

    Raises:
        RuntimeError: If disposal fails
    """
    cdef cf.OSStatus status = at.DisposeAUGraph(<at.AUGraph>graph_id)
    if status != 0:
        raise RuntimeError(f"DisposeAUGraph failed with status: {status}")


def au_graph_open(long graph_id):
    """Open an AUGraph (opens AudioUnits but doesn't initialize)

    Args:
        graph_id: AUGraph ID

    Raises:
        RuntimeError: If open fails
    """
    cdef cf.OSStatus status = at.AUGraphOpen(<at.AUGraph>graph_id)
    if status != 0:
        raise RuntimeError(f"AUGraphOpen failed with status: {status}")


def au_graph_close(long graph_id):
    """Close an AUGraph

    Args:
        graph_id: AUGraph ID

    Raises:
        RuntimeError: If close fails
    """
    cdef cf.OSStatus status = at.AUGraphClose(<at.AUGraph>graph_id)
    if status != 0:
        raise RuntimeError(f"AUGraphClose failed with status: {status}")


def au_graph_initialize(long graph_id):
    """Initialize an AUGraph (prepares for rendering)

    Args:
        graph_id: AUGraph ID

    Raises:
        RuntimeError: If initialization fails
    """
    cdef cf.OSStatus status = at.AUGraphInitialize(<at.AUGraph>graph_id)
    if status != 0:
        raise RuntimeError(f"AUGraphInitialize failed with status: {status}")


def au_graph_uninitialize(long graph_id):
    """Uninitialize an AUGraph

    Args:
        graph_id: AUGraph ID

    Raises:
        RuntimeError: If uninitialization fails
    """
    cdef cf.OSStatus status = at.AUGraphUninitialize(<at.AUGraph>graph_id)
    if status != 0:
        raise RuntimeError(f"AUGraphUninitialize failed with status: {status}")


def au_graph_start(long graph_id):
    """Start an AUGraph (begins rendering)

    Args:
        graph_id: AUGraph ID

    Raises:
        RuntimeError: If start fails
    """
    cdef cf.OSStatus status = at.AUGraphStart(<at.AUGraph>graph_id)
    if status != 0:
        raise RuntimeError(f"AUGraphStart failed with status: {status}")


def au_graph_stop(long graph_id):
    """Stop an AUGraph (stops rendering)

    Args:
        graph_id: AUGraph ID

    Raises:
        RuntimeError: If stop fails
    """
    cdef cf.OSStatus status = at.AUGraphStop(<at.AUGraph>graph_id)
    if status != 0:
        raise RuntimeError(f"AUGraphStop failed with status: {status}")


def au_graph_is_open(long graph_id) -> bool:
    """Check if AUGraph is open

    Args:
        graph_id: AUGraph ID

    Returns:
        True if open, False otherwise

    Raises:
        RuntimeError: If query fails
    """
    cdef cf.Boolean is_open
    cdef cf.OSStatus status = at.AUGraphIsOpen(<at.AUGraph>graph_id, &is_open)
    if status != 0:
        raise RuntimeError(f"AUGraphIsOpen failed with status: {status}")
    return bool(is_open)


def au_graph_is_initialized(long graph_id) -> bool:
    """Check if AUGraph is initialized

    Args:
        graph_id: AUGraph ID

    Returns:
        True if initialized, False otherwise

    Raises:
        RuntimeError: If query fails
    """
    cdef cf.Boolean is_initialized
    cdef cf.OSStatus status = at.AUGraphIsInitialized(<at.AUGraph>graph_id, &is_initialized)
    if status != 0:
        raise RuntimeError(f"AUGraphIsInitialized failed with status: {status}")
    return bool(is_initialized)


def au_graph_is_running(long graph_id) -> bool:
    """Check if AUGraph is running

    Args:
        graph_id: AUGraph ID

    Returns:
        True if running, False otherwise

    Raises:
        RuntimeError: If query fails
    """
    cdef cf.Boolean is_running
    cdef cf.OSStatus status = at.AUGraphIsRunning(<at.AUGraph>graph_id, &is_running)
    if status != 0:
        raise RuntimeError(f"AUGraphIsRunning failed with status: {status}")
    return bool(is_running)


def au_graph_add_node(long graph_id, dict description) -> int:
    """Add a node to an AUGraph

    Args:
        graph_id: AUGraph ID
        description: AudioComponentDescription as dict with keys:
                    'type', 'subtype', 'manufacturer', 'flags', 'flags_mask'

    Returns:
        Node ID

    Raises:
        RuntimeError: If adding node fails
    """
    cdef at.AudioComponentDescription desc
    cdef at.AUNode node
    cdef cf.OSStatus status

    # Convert dict to AudioComponentDescription
    desc.componentType = description['type']
    desc.componentSubType = description['subtype']
    desc.componentManufacturer = description['manufacturer']
    desc.componentFlags = description.get('flags', 0)
    desc.componentFlagsMask = description.get('flags_mask', 0)

    status = at.AUGraphAddNode(<at.AUGraph>graph_id, &desc, &node)
    if status != 0:
        raise RuntimeError(f"AUGraphAddNode failed with status: {status}")

    return node


def au_graph_remove_node(long graph_id, int node_id):
    """Remove a node from an AUGraph

    Args:
        graph_id: AUGraph ID
        node_id: Node ID to remove

    Raises:
        RuntimeError: If removing node fails
    """
    cdef cf.OSStatus status = at.AUGraphRemoveNode(<at.AUGraph>graph_id, <at.AUNode>node_id)
    if status != 0:
        raise RuntimeError(f"AUGraphRemoveNode failed with status: {status}")


def au_graph_get_node_count(long graph_id) -> int:
    """Get the number of nodes in an AUGraph

    Args:
        graph_id: AUGraph ID

    Returns:
        Number of nodes

    Raises:
        RuntimeError: If query fails
    """
    cdef cf.UInt32 count
    cdef cf.OSStatus status = at.AUGraphGetNodeCount(<at.AUGraph>graph_id, &count)
    if status != 0:
        raise RuntimeError(f"AUGraphGetNodeCount failed with status: {status}")
    return count


def au_graph_get_ind_node(long graph_id, int index) -> int:
    """Get node at index

    Args:
        graph_id: AUGraph ID
        index: Node index

    Returns:
        Node ID

    Raises:
        RuntimeError: If query fails
    """
    cdef at.AUNode node
    cdef cf.OSStatus status = at.AUGraphGetIndNode(<at.AUGraph>graph_id, index, &node)
    if status != 0:
        raise RuntimeError(f"AUGraphGetIndNode failed with status: {status}")
    return node


def au_graph_node_info(long graph_id, int node_id) -> tuple:
    """Get information about a node

    Args:
        graph_id: AUGraph ID
        node_id: Node ID

    Returns:
        Tuple of (description_dict, audio_unit_id)

    Raises:
        RuntimeError: If query fails
    """
    cdef at.AudioComponentDescription desc
    cdef at.AudioUnit audio_unit
    cdef cf.OSStatus status

    status = at.AUGraphNodeInfo(<at.AUGraph>graph_id, <at.AUNode>node_id, &desc, &audio_unit)
    if status != 0:
        raise RuntimeError(f"AUGraphNodeInfo failed with status: {status}")

    desc_dict = {
        'type': desc.componentType,
        'subtype': desc.componentSubType,
        'manufacturer': desc.componentManufacturer,
        'flags': desc.componentFlags,
        'flags_mask': desc.componentFlagsMask
    }

    return (desc_dict, <long>audio_unit)


def au_graph_connect_node_input(long graph_id, int source_node, int source_output,
                                  int dest_node, int dest_input):
    """Connect two nodes

    Args:
        graph_id: AUGraph ID
        source_node: Source node ID
        source_output: Source output bus number
        dest_node: Destination node ID
        dest_input: Destination input bus number

    Raises:
        RuntimeError: If connection fails
    """
    cdef cf.OSStatus status

    status = at.AUGraphConnectNodeInput(
        <at.AUGraph>graph_id,
        <at.AUNode>source_node,
        source_output,
        <at.AUNode>dest_node,
        dest_input
    )
    if status != 0:
        raise RuntimeError(f"AUGraphConnectNodeInput failed with status: {status}")


def au_graph_disconnect_node_input(long graph_id, int dest_node, int dest_input):
    """Disconnect a node's input

    Args:
        graph_id: AUGraph ID
        dest_node: Destination node ID
        dest_input: Destination input bus number

    Raises:
        RuntimeError: If disconnection fails
    """
    cdef cf.OSStatus status

    status = at.AUGraphDisconnectNodeInput(
        <at.AUGraph>graph_id,
        <at.AUNode>dest_node,
        dest_input
    )
    if status != 0:
        raise RuntimeError(f"AUGraphDisconnectNodeInput failed with status: {status}")


def au_graph_clear_connections(long graph_id):
    """Clear all connections in a graph

    Args:
        graph_id: AUGraph ID

    Raises:
        RuntimeError: If clearing connections fails
    """
    cdef cf.OSStatus status = at.AUGraphClearConnections(<at.AUGraph>graph_id)
    if status != 0:
        raise RuntimeError(f"AUGraphClearConnections failed with status: {status}")


def au_graph_update(long graph_id) -> bool:
    """Update the graph after making changes

    Args:
        graph_id: AUGraph ID

    Returns:
        True if update completed, False if pending

    Raises:
        RuntimeError: If update fails
    """
    cdef cf.Boolean is_updated
    cdef cf.OSStatus status = at.AUGraphUpdate(<at.AUGraph>graph_id, &is_updated)
    if status != 0:
        raise RuntimeError(f"AUGraphUpdate failed with status: {status}")
    return bool(is_updated)


def au_graph_get_cpu_load(long graph_id) -> float:
    """Get current CPU load of the graph

    Args:
        graph_id: AUGraph ID

    Returns:
        CPU load as percentage (0.0-1.0)

    Raises:
        RuntimeError: If query fails
    """
    cdef cf.Float32 cpu_load
    cdef cf.OSStatus status = at.AUGraphGetCPULoad(<at.AUGraph>graph_id, &cpu_load)
    if status != 0:
        raise RuntimeError(f"AUGraphGetCPULoad failed with status: {status}")
    return cpu_load


def au_graph_get_max_cpu_load(long graph_id) -> float:
    """Get maximum CPU load of the graph since last query

    Args:
        graph_id: AUGraph ID

    Returns:
        Maximum CPU load as percentage (0.0-1.0)

    Raises:
        RuntimeError: If query fails
    """
    cdef cf.Float32 max_load
    cdef cf.OSStatus status = at.AUGraphGetMaxCPULoad(<at.AUGraph>graph_id, &max_load)
    if status != 0:
        raise RuntimeError(f"AUGraphGetMaxCPULoad failed with status: {status}")
    return max_load


# AUGraph error code getters
def get_au_graph_err_node_not_found() -> int:
    """Get error code for node not found"""
    return -10860

def get_au_graph_err_invalid_connection() -> int:
    """Get error code for invalid connection"""
    return -10861

def get_au_graph_err_output_node_err() -> int:
    """Get error code for output node error"""
    return -10862

def get_au_graph_err_cannot_do_in_current_context() -> int:
    """Get error code for cannot do in current context"""
    return -10863

def get_au_graph_err_invalid_audio_unit() -> int:
    """Get error code for invalid audio unit"""
    return -10864


# ============================================================================
# CoreAudioClock - Audio/MIDI Synchronization and Timing
# ============================================================================

def ca_clock_new() -> int:
    """Create a new CoreAudio clock

    Returns:
        Clock ID as integer

    Raises:
        RuntimeError: If clock creation fails
    """
    cdef at.CAClockRef clock_ref
    cdef cf.OSStatus status = at.CAClockNew(0, &clock_ref)
    if status != 0:
        raise RuntimeError(f"CAClockNew failed with status: {status}")
    return <long>clock_ref


def ca_clock_dispose(long clock_id):
    """Dispose a CoreAudio clock

    Args:
        clock_id: Clock ID to dispose

    Raises:
        RuntimeError: If disposal fails
    """
    cdef cf.OSStatus status = at.CAClockDispose(<at.CAClockRef>clock_id)
    if status != 0:
        raise RuntimeError(f"CAClockDispose failed with status: {status}")


def ca_clock_get_current_time(long clock_id, int time_format) -> dict:
    """Get current time from clock

    Args:
        clock_id: Clock ID
        time_format: Time format (use get_ca_clock_time_format_* functions)

    Returns:
        Dictionary with time information

    Raises:
        RuntimeError: If getting time fails
    """
    cdef at.CAClockTime clock_time
    cdef cf.OSStatus status = at.CAClockGetCurrentTime(
        <at.CAClockRef>clock_id,
        <at.CAClockTimeFormat>time_format,
        &clock_time
    )
    if status != 0:
        raise RuntimeError(f"CAClockGetCurrentTime failed with status: {status}")

    result = {"format": clock_time.format}

    # Extract value based on format
    if clock_time.format == at.kCAClockTimeFormat_HostTime:
        result["value"] = clock_time.time.hostTime
    elif clock_time.format == at.kCAClockTimeFormat_Samples:
        result["value"] = clock_time.time.samples
    elif clock_time.format == at.kCAClockTimeFormat_Beats:
        result["value"] = clock_time.time.beats
    elif clock_time.format in (at.kCAClockTimeFormat_Seconds,
                                at.kCAClockTimeFormat_SMPTESeconds,
                                at.kCAClockTimeFormat_AbsoluteSeconds):
        result["value"] = clock_time.time.seconds
    elif clock_time.format == at.kCAClockTimeFormat_SMPTETime:
        result["value"] = {
            "hours": clock_time.time.smpte.mHours,
            "minutes": clock_time.time.smpte.mMinutes,
            "seconds": clock_time.time.smpte.mSeconds,
            "frames": clock_time.time.smpte.mFrames,
            "subframes": clock_time.time.smpte.mSubframes,
            "subframe_divisor": clock_time.time.smpte.mSubframeDivisor,
            "type": clock_time.time.smpte.mType,
            "flags": clock_time.time.smpte.mFlags
        }

    return result


def ca_clock_start(long clock_id):
    """Start the clock

    Args:
        clock_id: Clock ID

    Raises:
        RuntimeError: If starting clock fails
    """
    cdef cf.OSStatus status = at.CAClockStart(<at.CAClockRef>clock_id)
    if status != 0:
        raise RuntimeError(f"CAClockStart failed with status: {status}")


def ca_clock_stop(long clock_id):
    """Stop the clock

    Args:
        clock_id: Clock ID

    Raises:
        RuntimeError: If stopping clock fails
    """
    cdef cf.OSStatus status = at.CAClockStop(<at.CAClockRef>clock_id)
    if status != 0:
        raise RuntimeError(f"CAClockStop failed with status: {status}")


def ca_clock_set_play_rate(long clock_id, double rate):
    """Set playback rate

    Args:
        clock_id: Clock ID
        rate: Playback rate (1.0 = normal speed)

    Raises:
        RuntimeError: If setting rate fails
    """
    cdef cf.OSStatus status = at.CAClockSetPlayRate(<at.CAClockRef>clock_id, rate)
    if status != 0:
        raise RuntimeError(f"CAClockSetPlayRate failed with status: {status}")


def ca_clock_get_play_rate(long clock_id) -> float:
    """Get current playback rate

    Args:
        clock_id: Clock ID

    Returns:
        Current playback rate

    Raises:
        RuntimeError: If getting rate fails
    """
    cdef double rate
    cdef cf.OSStatus status = at.CAClockGetPlayRate(<at.CAClockRef>clock_id, &rate)
    if status != 0:
        raise RuntimeError(f"CAClockGetPlayRate failed with status: {status}")
    return rate


def get_ca_clock_time_format_host_time() -> int:
    """Get time format for host time"""
    return at.kCAClockTimeFormat_HostTime

def get_ca_clock_time_format_samples() -> int:
    """Get time format for audio samples"""
    return at.kCAClockTimeFormat_Samples

def get_ca_clock_time_format_beats() -> int:
    """Get time format for musical beats"""
    return at.kCAClockTimeFormat_Beats

def get_ca_clock_time_format_seconds() -> int:
    """Get time format for seconds"""
    return at.kCAClockTimeFormat_Seconds

def get_ca_clock_time_format_smpte_time() -> int:
    """Get time format for SMPTE time structure"""
    return at.kCAClockTimeFormat_SMPTETime
