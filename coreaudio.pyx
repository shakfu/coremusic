cimport coreaudio as ca

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


def audio_hardware_unload() -> int:
    return ca.AudioHardwareUnload()


def audio_hardware_destroy_aggregate_device(int in_device_id) -> int:
    return ca.AudioHardwareDestroyAggregateDevice(in_device_id)
