cimport coreaudio

from libc.stdio cimport printf, fprintf, stderr, FILE
from posix.unistd cimport sleep

# from libc.string cimport strcpy, strlen
# from libc.stdlib cimport malloc


# def test():
#     printf("code: %c%c%c%c", FOURCC_ARGS(778924083))

def test():
    # print(coreaudio.AudioObjectPropertySelector.kAudioDevicePropertyPlugIn)
    print(coreaudio.Dummy.Plug)

# def test_error():
#     return coreaudio.kAudio_UnimplementedError
