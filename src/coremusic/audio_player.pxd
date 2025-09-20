# audio_player.pxd
# Cython declarations for audio_player.h

from . cimport corefoundation as cf
from . cimport audiotoolbox as at
from . cimport coreaudio as ca

# Function declarations - use ctypedef to match C struct definitions exactly
cdef extern from "audio_player.h":
    # Audio player data structure (defined in C header)
    ctypedef struct AudioPlayerData:
        ca.AudioBufferList *bufferList
        cf.UInt32 totalFrames
        cf.UInt32 currentFrame
        cf.Boolean playing
        cf.Boolean loop

    # Audio output structure (defined in C header)
    ctypedef struct AudioOutput:
        at.AudioUnit outputUnit
        AudioPlayerData playerData

    # File loading and disposal
    cf.OSStatus LoadAudioFile(cf.CFURLRef url, AudioPlayerData *playerData)
    void DisposeAudioPlayer(AudioPlayerData *playerData)

    # Audio output management
    cf.OSStatus SetupAudioOutput(AudioOutput *output)
    cf.OSStatus StartAudioOutput(AudioOutput *output)
    cf.OSStatus StopAudioOutput(AudioOutput *output)
    void DisposeAudioOutput(AudioOutput *output)

    # Control functions
    void SetLooping(AudioPlayerData *playerData, cf.Boolean loop)
    void ResetPlayback(AudioPlayerData *playerData)
    cf.Boolean IsPlaying(AudioPlayerData *playerData)
    cf.Float32 GetPlaybackProgress(AudioPlayerData *playerData)