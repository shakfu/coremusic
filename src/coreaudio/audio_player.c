//
//  audio_player.c
//  cycoreaudio - Simple Audio Player Implementation
//
//  Based on iOS CoreAudio player by James Alvarez
//  Adapted for cycoreaudio Python wrapper
//

#include "audio_player.h"
#include <math.h>

#define SAMPLE_RATE 44100
#define CHANNELS 2
#define SAMPLE_SIZE sizeof(Float32)

// Standard audio format for our player
static const AudioStreamBasicDescription standardFormat = {
    .mSampleRate        = SAMPLE_RATE,
    .mFormatID          = kAudioFormatLinearPCM,
    .mFormatFlags       = kAudioFormatFlagIsFloat,
    .mBytesPerPacket    = SAMPLE_SIZE * CHANNELS,
    .mFramesPerPacket   = 1,
    .mBytesPerFrame     = CHANNELS * SAMPLE_SIZE,
    .mChannelsPerFrame  = CHANNELS,
    .mBitsPerChannel    = 8 * SAMPLE_SIZE,
    .mReserved          = 0
};

// Error checking helper
static void CheckError(OSStatus error, const char *operation) {
    if (error == noErr) return;
    
    char str[20];
    // Check if it's a 4-char-code
    *(UInt32 *)(str + 1) = CFSwapInt32HostToBig(error);
    if (isprint(str[1]) && isprint(str[2]) && isprint(str[3]) && isprint(str[4])) {
        str[0] = str[5] = '\'';
        str[6] = '\0';
    } else {
        sprintf(str, "%d", (int)error);
    }
    
    fprintf(stderr, "Error: %s (%s)\n", operation, str);
}

// Render callback function
static OSStatus RenderCallback(void *inRefCon,
                              AudioUnitRenderActionFlags *ioActionFlags,
                              const AudioTimeStamp *inTimeStamp,
                              UInt32 inBusNumber,
                              UInt32 inNumberFrames,
                              AudioBufferList *ioData) {
    
    AudioOutput *audioOutput = (AudioOutput*)inRefCon;
    AudioPlayerData *playerData = &audioOutput->playerData;
    
    // Clear output buffers
    for (UInt32 buffer = 0; buffer < ioData->mNumberBuffers; buffer++) {
        memset(ioData->mBuffers[buffer].mData, 0, ioData->mBuffers[buffer].mDataByteSize);
    }
    
    // Check if we have data and are playing
    if (!playerData->playing || !playerData->bufferList || playerData->totalFrames == 0) {
        return noErr; // Return silence
    }
    
    UInt32 currentFrame = playerData->currentFrame;
    UInt32 maxFrames = playerData->totalFrames;
    
    Float32 *outputData = (Float32*)ioData->mBuffers[0].mData;
    Float32 *inputData = (Float32*)playerData->bufferList->mBuffers[0].mData;
    
    // Copy audio data frame by frame
    for (UInt32 frame = 0; frame < inNumberFrames; ++frame) {
        if (currentFrame >= maxFrames) {
            if (playerData->loop) {
                currentFrame = 0;  // Loop back to start
            } else {
                playerData->playing = false;
                break; // Stop playing
            }
        }
        
        // Copy stereo frame (2 channels)
        UInt32 outSample = frame * 2;
        UInt32 inSample = currentFrame * 2;
        
        if (currentFrame < maxFrames) {
            outputData[outSample] = inputData[inSample];         // Left channel
            outputData[outSample + 1] = inputData[inSample + 1]; // Right channel
        }
        
        currentFrame++;
    }
    
    playerData->currentFrame = currentFrame;
    
    return noErr;
}

// Load audio file using ExtAudioFile
OSStatus LoadAudioFile(CFURLRef url, AudioPlayerData *playerData) {
    ExtAudioFileRef audioFile;
    OSStatus status = noErr;
    
    // Initialize player data
    playerData->bufferList = NULL;
    playerData->totalFrames = 0;
    playerData->currentFrame = 0;
    playerData->playing = false;
    playerData->loop = false;
    
    // Open audio file
    status = ExtAudioFileOpenURL(url, &audioFile);
    if (status != noErr) {
        CheckError(status, "Could not open audio file");
        return status;
    }
    
    // Get file format
    AudioStreamBasicDescription fileFormat;
    UInt32 size = sizeof(fileFormat);
    status = ExtAudioFileGetProperty(audioFile,
                                    kExtAudioFileProperty_FileDataFormat,
                                    &size,
                                    &fileFormat);
    if (status != noErr) {
        CheckError(status, "Could not get file format");
        ExtAudioFileDispose(audioFile);
        return status;
    }
    
    // Set client format (what we want the data converted to)
    status = ExtAudioFileSetProperty(audioFile,
                                    kExtAudioFileProperty_ClientDataFormat,
                                    sizeof(standardFormat),
                                    &standardFormat);
    if (status != noErr) {
        CheckError(status, "Could not set client format");
        ExtAudioFileDispose(audioFile);
        return status;
    }
    
    // Get file length in frames
    UInt64 fileLengthInFrames;
    size = sizeof(fileLengthInFrames);
    status = ExtAudioFileGetProperty(audioFile,
                                    kExtAudioFileProperty_FileLengthFrames,
                                    &size,
                                    &fileLengthInFrames);
    if (status != noErr) {
        CheckError(status, "Could not get file length");
        ExtAudioFileDispose(audioFile);
        return status;
    }
    
    // Calculate true length accounting for sample rate conversion
    fileLengthInFrames = ceil(fileLengthInFrames * (standardFormat.mSampleRate / fileFormat.mSampleRate));
    
    // Prepare AudioBufferList
    int numberOfBuffers = 1; // Always use interleaved for simplicity
    int bytesPerBuffer = standardFormat.mBytesPerFrame * (int)fileLengthInFrames;
    
    AudioBufferList *bufferList = malloc(sizeof(AudioBufferList) + (numberOfBuffers-1)*sizeof(AudioBuffer));
    if (!bufferList) {
        ExtAudioFileDispose(audioFile);
        return -1;
    }
    
    bufferList->mNumberBuffers = numberOfBuffers;
    bufferList->mBuffers[0].mData = calloc(bytesPerBuffer, 1);
    if (!bufferList->mBuffers[0].mData) {
        free(bufferList);
        ExtAudioFileDispose(audioFile);
        return -1;
    }
    bufferList->mBuffers[0].mDataByteSize = bytesPerBuffer;
    bufferList->mBuffers[0].mNumberChannels = standardFormat.mChannelsPerFrame;
    
    // Read audio data in chunks
    UInt32 readFrames = 0;
    while (readFrames < fileLengthInFrames) {
        UInt32 framesToRead = (UInt32)fileLengthInFrames - readFrames;
        if (framesToRead > 16384) {
            framesToRead = 16384; // Read in chunks to avoid crashes
        }
        
        // Create temporary buffer list for this chunk
        AudioBufferList tempBufferList;
        tempBufferList.mNumberBuffers = 1;
        tempBufferList.mBuffers[0].mNumberChannels = standardFormat.mChannelsPerFrame;
        tempBufferList.mBuffers[0].mData = (char*)bufferList->mBuffers[0].mData + (readFrames * standardFormat.mBytesPerFrame);
        tempBufferList.mBuffers[0].mDataByteSize = framesToRead * standardFormat.mBytesPerFrame;
        
        status = ExtAudioFileRead(audioFile, &framesToRead, &tempBufferList);
        if (framesToRead == 0) break;
        
        readFrames += framesToRead;
    }
    
    ExtAudioFileDispose(audioFile);
    
    // Set up player data
    playerData->bufferList = bufferList;
    playerData->totalFrames = readFrames;
    playerData->currentFrame = 0;
    playerData->playing = false;
    playerData->loop = false;
    
    return noErr;
}

void DisposeAudioPlayer(AudioPlayerData *playerData) {
    if (playerData && playerData->bufferList) {
        for (UInt32 i = 0; i < playerData->bufferList->mNumberBuffers; i++) {
            if (playerData->bufferList->mBuffers[i].mData) {
                free(playerData->bufferList->mBuffers[i].mData);
            }
        }
        free(playerData->bufferList);
        playerData->bufferList = NULL;
    }
    playerData->totalFrames = 0;
    playerData->currentFrame = 0;
    playerData->playing = false;
}

OSStatus SetupAudioOutput(AudioOutput *output) {
    OSStatus status = noErr;
    
    // Find default output AudioComponent
    AudioComponentDescription outputDesc = {
        .componentType = kAudioUnitType_Output,
        .componentSubType = kAudioUnitSubType_DefaultOutput,
        .componentManufacturer = kAudioUnitManufacturer_Apple,
        .componentFlags = 0,
        .componentFlagsMask = 0
    };
    
    AudioComponent comp = AudioComponentFindNext(NULL, &outputDesc);
    if (comp == NULL) {
        fprintf(stderr, "Cannot find default output AudioComponent\n");
        return -1;
    }
    
    // Create AudioUnit instance
    status = AudioComponentInstanceNew(comp, &output->outputUnit);
    if (status != noErr) {
        CheckError(status, "Could not create AudioUnit instance");
        return status;
    }
    
    // Set stream format
    status = AudioUnitSetProperty(output->outputUnit,
                                 kAudioUnitProperty_StreamFormat,
                                 kAudioUnitScope_Input,
                                 0,
                                 &standardFormat,
                                 sizeof(standardFormat));
    if (status != noErr) {
        CheckError(status, "Could not set stream format");
        return status;
    }
    
    // Set render callback
    AURenderCallbackStruct callbackStruct = {
        .inputProc = RenderCallback,
        .inputProcRefCon = output
    };
    
    status = AudioUnitSetProperty(output->outputUnit,
                                 kAudioUnitProperty_SetRenderCallback,
                                 kAudioUnitScope_Global,
                                 0,
                                 &callbackStruct,
                                 sizeof(callbackStruct));
    if (status != noErr) {
        CheckError(status, "Could not set render callback");
        return status;
    }
    
    // Initialize AudioUnit
    status = AudioUnitInitialize(output->outputUnit);
    if (status != noErr) {
        CheckError(status, "Could not initialize AudioUnit");
        return status;
    }
    
    return noErr;
}

OSStatus StartAudioOutput(AudioOutput *output) {
    output->playerData.playing = true;
    output->playerData.currentFrame = 0;
    
    OSStatus status = AudioOutputUnitStart(output->outputUnit);
    if (status != noErr) {
        CheckError(status, "Could not start AudioUnit");
        output->playerData.playing = false;
    }
    return status;
}

OSStatus StopAudioOutput(AudioOutput *output) {
    output->playerData.playing = false;
    
    OSStatus status = AudioOutputUnitStop(output->outputUnit);
    if (status != noErr) {
        CheckError(status, "Could not stop AudioUnit");
    }
    return status;
}

void DisposeAudioOutput(AudioOutput *output) {
    if (output->outputUnit) {
        AudioOutputUnitStop(output->outputUnit);
        AudioUnitUninitialize(output->outputUnit);
        AudioComponentInstanceDispose(output->outputUnit);
        output->outputUnit = NULL;
    }
}

// Control functions
void SetLooping(AudioPlayerData *playerData, Boolean loop) {
    playerData->loop = loop;
}

void ResetPlayback(AudioPlayerData *playerData) {
    playerData->currentFrame = 0;
}

Boolean IsPlaying(AudioPlayerData *playerData) {
    return playerData->playing;
}

Float32 GetPlaybackProgress(AudioPlayerData *playerData) {
    if (playerData->totalFrames == 0) return 0.0f;
    return (Float32)playerData->currentFrame / (Float32)playerData->totalFrames;
}