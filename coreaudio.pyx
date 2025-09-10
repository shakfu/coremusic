cimport coreaudio as ca
from libc.stdlib cimport malloc, free

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
    ca.AudioObjectShow(object_id)

def audio_hardware_unload() -> int:
    return ca.AudioHardwareUnload()


def audio_hardware_destroy_aggregate_device(int in_device_id) -> int:
    return ca.AudioHardwareDestroyAggregateDevice(in_device_id)


# Audio File Functions
def audio_file_open_url(str file_path, int permissions=1, int file_type_hint=0):
    """Open an audio file at the given path"""
    cdef ca.AudioFileID audio_file
    cdef ca.CFURLRef url_ref
    cdef bytes path_bytes = file_path.encode('utf-8')
    
    url_ref = ca.CFURLCreateFromFileSystemRepresentation(
        ca.kCFAllocatorDefault, 
        <const ca.UInt8*>path_bytes, 
        len(path_bytes), 
        False
    )
    
    if not url_ref:
        raise ValueError("Could not create URL from file path")
    
    cdef ca.OSStatus status = ca.AudioFileOpenURL(
        url_ref, 
        <ca.AudioFilePermissions>permissions, 
        <ca.AudioFileTypeID>file_type_hint, 
        &audio_file
    )
    
    ca.CFRelease(url_ref)
    
    if status != 0:
        raise RuntimeError(f"AudioFileOpenURL failed with status: {status}")
    
    return <long>audio_file


def audio_file_close(long audio_file_id):
    """Close an audio file"""
    cdef ca.AudioFileID audio_file = <ca.AudioFileID>audio_file_id
    cdef ca.OSStatus status = ca.AudioFileClose(audio_file)
    if status != 0:
        raise RuntimeError(f"AudioFileClose failed with status: {status}")
    return status


def audio_file_get_property(long audio_file_id, int property_id):
    """Get a property from an audio file"""
    cdef ca.AudioFileID audio_file = <ca.AudioFileID>audio_file_id
    cdef ca.UInt32 data_size = 0
    cdef ca.UInt32 writable = 0
    
    # Get the size of the property data
    cdef ca.OSStatus status = ca.AudioFileGetPropertyInfo(
        audio_file, 
        <ca.AudioFilePropertyID>property_id, 
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
        status = ca.AudioFileGetProperty(
            audio_file, 
            <ca.AudioFilePropertyID>property_id, 
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
    cdef ca.AudioFileID audio_file = <ca.AudioFileID>audio_file_id
    cdef ca.UInt32 num_bytes = 0
    cdef ca.UInt32 packet_count = <ca.UInt32>num_packets
    
    # First get the maximum packet size to determine buffer size
    cdef ca.UInt32 max_packet_size = 0
    cdef ca.UInt32 prop_size = sizeof(ca.UInt32)
    
    cdef ca.OSStatus status = ca.AudioFileGetProperty(
        audio_file,
        ca.kAudioFilePropertyMaximumPacketSize,
        &prop_size,
        &max_packet_size
    )
    
    if status != 0:
        raise RuntimeError(f"Could not get maximum packet size: {status}")
    
    # Allocate buffer
    cdef ca.UInt32 buffer_size = max_packet_size * packet_count
    cdef char* buffer = <char*>malloc(buffer_size)
    if not buffer:
        raise MemoryError("Could not allocate buffer for packet data")
    
    try:
        num_bytes = buffer_size
        status = ca.AudioFileReadPackets(
            audio_file,
            False,  # don't use cache
            &num_bytes,
            NULL,   # no packet descriptions
            <ca.SInt64>start_packet,
            &packet_count,
            buffer
        )
        
        if status != 0:
            raise RuntimeError(f"AudioFileReadPackets failed with status: {status}")
        
        return buffer[:num_bytes], packet_count
        
    finally:
        free(buffer)


# Audio Queue Functions  
cdef void audio_queue_output_callback(void* user_data, ca.AudioQueueRef queue, ca.AudioQueueBufferRef buffer) noexcept:
    """C callback function for audio queue output"""
    # This will be called by CoreAudio when it needs more audio data
    # For now, we'll just enqueue the buffer again to keep playing
    cdef ca.OSStatus status = ca.AudioQueueEnqueueBuffer(queue, buffer, 0, NULL)


def audio_queue_new_output(audio_format):
    """Create a new audio output queue"""
    cdef ca.AudioStreamBasicDescription format
    cdef ca.AudioQueueRef queue
    
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
    
    cdef ca.OSStatus status = ca.AudioQueueNewOutput(
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
    cdef ca.AudioQueueRef queue = <ca.AudioQueueRef>queue_id
    cdef ca.AudioQueueBufferRef buffer
    
    cdef ca.OSStatus status = ca.AudioQueueAllocateBuffer(
        queue, 
        <ca.UInt32>buffer_size, 
        &buffer
    )
    
    if status != 0:
        raise RuntimeError(f"AudioQueueAllocateBuffer failed with status: {status}")
    
    return <long>buffer


def audio_queue_enqueue_buffer(long queue_id, long buffer_id):
    """Enqueue a buffer to an audio queue"""
    cdef ca.AudioQueueRef queue = <ca.AudioQueueRef>queue_id
    cdef ca.AudioQueueBufferRef buffer = <ca.AudioQueueBufferRef>buffer_id
    
    cdef ca.OSStatus status = ca.AudioQueueEnqueueBuffer(queue, buffer, 0, NULL)
    
    if status != 0:
        raise RuntimeError(f"AudioQueueEnqueueBuffer failed with status: {status}")
    
    return status


def audio_queue_start(long queue_id):
    """Start an audio queue"""
    cdef ca.AudioQueueRef queue = <ca.AudioQueueRef>queue_id
    
    cdef ca.OSStatus status = ca.AudioQueueStart(queue, NULL)
    
    if status != 0:
        raise RuntimeError(f"AudioQueueStart failed with status: {status}")
    
    return status


def audio_queue_stop(long queue_id, bint immediate=True):
    """Stop an audio queue"""
    cdef ca.AudioQueueRef queue = <ca.AudioQueueRef>queue_id
    
    cdef ca.OSStatus status = ca.AudioQueueStop(queue, immediate)
    
    if status != 0:
        raise RuntimeError(f"AudioQueueStop failed with status: {status}")
    
    return status


def audio_queue_dispose(long queue_id, bint immediate=True):
    """Dispose of an audio queue"""
    cdef ca.AudioQueueRef queue = <ca.AudioQueueRef>queue_id
    
    cdef ca.OSStatus status = ca.AudioQueueDispose(queue, immediate)
    
    if status != 0:
        raise RuntimeError(f"AudioQueueDispose failed with status: {status}")
    
    return status


# Constant getter functions
def get_audio_format_linear_pcm():
    return ca.kAudioFormatLinearPCM

def get_linear_pcm_format_flag_is_signed_integer():
    return ca.kLinearPCMFormatFlagIsSignedInteger

def get_linear_pcm_format_flag_is_packed():
    return ca.kLinearPCMFormatFlagIsPacked

def get_audio_file_wave_type():
    return ca.kAudioFileWAVEType

def get_audio_file_read_permission():
    return ca.kAudioFileReadPermission

def get_audio_file_property_data_format():
    return ca.kAudioFilePropertyDataFormat

def get_audio_file_property_maximum_packet_size():
    return ca.kAudioFilePropertyMaximumPacketSize

def test_error() -> int:
    """Test function to verify the module works"""
    return ca.kAudio_UnimplementedError
