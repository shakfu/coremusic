//
//  audio_player.h
//  cycoreaudio - Simple Audio Player Implementation
//
//  Based on iOS CoreAudio player by James Alvarez
//  Adapted for cycoreaudio Python wrapper
//

#ifndef audio_player_h
#define audio_player_h

#include <stdio.h>
#include <AudioToolbox/AudioToolbox.h>
#include <AudioUnit/AudioUnit.h>

// Audio player data structure
typedef struct {
    AudioBufferList *bufferList;
    UInt32 totalFrames;
    UInt32 currentFrame;
    Boolean playing;
    Boolean loop;
} AudioPlayerData;

// Audio output structure
typedef struct {
    AudioUnit outputUnit;
    AudioPlayerData playerData;
} AudioOutput;

// Function declarations
OSStatus LoadAudioFile(CFURLRef url, AudioPlayerData *playerData);
void DisposeAudioPlayer(AudioPlayerData *playerData);

OSStatus SetupAudioOutput(AudioOutput *output);
OSStatus StartAudioOutput(AudioOutput *output);
OSStatus StopAudioOutput(AudioOutput *output);
void DisposeAudioOutput(AudioOutput *output);

// Control functions
void SetLooping(AudioPlayerData *playerData, Boolean loop);
void ResetPlayback(AudioPlayerData *playerData);
Boolean IsPlaying(AudioPlayerData *playerData);
Float32 GetPlaybackProgress(AudioPlayerData *playerData);

#endif /* audio_player_h */