# audio_player.pxd
# Cython declarations for audio_player.h

from coreaudio cimport *

# Function declarations - use ctypedef to match C struct definitions exactly
cdef extern from "audio_player.h":
    # Audio player data structure (defined in C header)
    ctypedef struct AudioPlayerData:
        AudioBufferList *bufferList
        UInt32 totalFrames
        UInt32 currentFrame
        Boolean playing
        Boolean loop

    # Audio output structure (defined in C header)
    ctypedef struct AudioOutput:
        AudioUnit outputUnit
        AudioPlayerData playerData
    
    # File loading and disposal
    OSStatus LoadAudioFile(CFURLRef url, AudioPlayerData *playerData)
    void DisposeAudioPlayer(AudioPlayerData *playerData)
    
    # Audio output management
    OSStatus SetupAudioOutput(AudioOutput *output)
    OSStatus StartAudioOutput(AudioOutput *output)
    OSStatus StopAudioOutput(AudioOutput *output)
    void DisposeAudioOutput(AudioOutput *output)
    
    # Control functions
    void SetLooping(AudioPlayerData *playerData, Boolean loop)
    void ResetPlayback(AudioPlayerData *playerData)
    Boolean IsPlaying(AudioPlayerData *playerData)
    Float32 GetPlaybackProgress(AudioPlayerData *playerData)