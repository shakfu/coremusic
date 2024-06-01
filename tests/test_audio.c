// from: https://gist.github.com/framkant/6daed7f6145af958f0bc
// -------------------------------------------------------------------------------
// CoreAudio continuous play test
// (c) 2014 by Arthur Langereis (@zenmumbler)
// created: 2014-12-07
// 
// As part of my efforts for stardazed and to create a Mac OS X version of
// Handmade Hero.
//
// compile with:
// clang++ -std=c++11 -stdlib=libc++ -framework AudioToolbox catest.cpp -o catest
// then run:
// ./catest
//
// converted to pure c99 by Filip Wänström (@filipwanstrom)
// compile with:
// clang -std=c99 -lc -framework AudioToolbox CoreAudioTest.c -o CoreAudioTestC
// the run
// ./CoreAudioTestC
//
// -------------------------------------------------------------------------------

#include <string.h>
#include <math.h>
#include <unistd.h>

#include <AudioToolbox/AudioToolbox.h>

typedef struct SoundState {
    float toneFreq, volume;
    float sampleRate, frameOffset;
    float squareWaveSign;
}SoundState;


void auCallback(void *inUserData, AudioQueueRef queue, AudioQueueBufferRef buffer) {
  SoundState *soundState = (SoundState*)(inUserData);
    
    // we're just filling the entire buffer here
    // In a real game we might only fill part of the buffer and set the mAudioDataBytes
    // accordingly.
    uint32_t framesToGen = buffer->mAudioDataBytesCapacity / 4;
    buffer->mAudioDataByteSize = framesToGen * 4;

    // calc the samples per up/down portion of each square wave (with 50% period)
    float framesPerTransition = soundState->sampleRate / soundState->toneFreq;

    // sample to output at current state
    int16_t sample = 32767.f * soundState->squareWaveSign * soundState->volume;
    
    int16_t *bufferPos =(int16_t*)(buffer->mAudioData);
    float frameOffset = soundState->frameOffset;

    while (framesToGen) {
        // calc rounded frames to generate and accumulate fractional error
        uint32_t frames;
        uint32_t needFrames = (uint32_t)(round(framesPerTransition - frameOffset));
        frameOffset -= framesPerTransition - needFrames;

        // we may be at the end of the buffer, if so, place offset at location in wave and clip
        if (needFrames > framesToGen) {
            frameOffset += framesToGen;
            frames = framesToGen;
        }
        else {
            frames = needFrames;
        }
        framesToGen -= frames;

        // simply put the samples in
        for (int x = 0; x < frames; ++x) {
            *bufferPos++ = sample;
            *bufferPos++ = sample;
        }

        // flip sign of wave unless we were cut off prematurely
        if (needFrames == frames)
            sample = -sample;
    }

    // save square wave state for next callback
    if (sample > 0)
        soundState->squareWaveSign = 1;
    else
        soundState->squareWaveSign = -1;
    soundState->frameOffset = frameOffset;

    AudioQueueEnqueueBuffer(queue, buffer, 0, 0);
}


int main(int argc, const char * argv[]) {
    // stereo 16-bit interleaved linear PCM audio data at 48kHz in SNORM format
    AudioStreamBasicDescription auDesc =  {};
    auDesc.mSampleRate = 48000.0f;
    auDesc.mFormatID = kAudioFormatLinearPCM;
    auDesc.mFormatFlags = kLinearPCMFormatFlagIsBigEndian | kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
    auDesc.mBytesPerPacket = 4;
    auDesc.mFramesPerPacket = 1;
    auDesc.mBytesPerFrame = 4;
    auDesc.mChannelsPerFrame = 2;
    auDesc.mBitsPerChannel = 16;
    
    AudioQueueRef auQueue = 0;
    AudioQueueBufferRef auBuffers[2] ={};
    
    // our persistent state for sound playback
    SoundState soundState=  {};
    soundState.toneFreq = 261.6 * 2; // 261.6 ~= Middle C frequency
    soundState.volume = 0.1; // don't crank this up and expect your ears to still function
    soundState.sampleRate = auDesc.mSampleRate;
    soundState.squareWaveSign = 1; // sign of the current part of the square wave
    
    OSStatus err;

    // most of the 0 and nullptr params here are for compressed sound formats etc.
    err = AudioQueueNewOutput(&auDesc, &auCallback, &soundState, 0, 0, 0, &auQueue);
    
    if (! err) {
        // generate buffers holding at most 1/16th of a second of data
        uint32_t bufferSize = auDesc.mBytesPerFrame * (auDesc.mSampleRate / 16);
        err = AudioQueueAllocateBuffer(auQueue, bufferSize, &(auBuffers[0]));

        if (! err) {
            err = AudioQueueAllocateBuffer(auQueue, bufferSize, &(auBuffers[1]));

            if (! err) {
                // prime the buffers
                auCallback(&soundState, auQueue, auBuffers[0]);
                auCallback(&soundState, auQueue, auBuffers[1]);

                // enqueue for playing
                AudioQueueEnqueueBuffer(auQueue, auBuffers[0], 0, 0);
                AudioQueueEnqueueBuffer(auQueue, auBuffers[1], 0, 0);

                // go!
                AudioQueueStart(auQueue, 0);
            }
        }
    }

    // Our AudioQueue creation options put the CA handling on its own thread
    // so this is a quick hack to allow us to hear some sound.
    //std::this_thread::sleep_for(std::chrono::seconds{2});
    usleep(2000000);
    

    // be nice even it doesn't really matter at this point
    if (auQueue)
        AudioQueueDispose(auQueue, true);
}

