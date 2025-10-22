// from: https://gist.github.com/gcatlin/0dd61f19d40804173d015c01a80461b8
// To run:
//   clang core-audio-sine-wave.c -framework AudioUnit && ./a.out
#include <AudioUnit/AudioUnit.h>

#define SAMPLE_RATE 48000
#define TONE_FREQUENCY 440
#define M_TAU 2.0 * M_PI

OSStatus RenderSineWave(
        void *inRefCon,
        AudioUnitRenderActionFlags *ioActionFlags,
        const AudioTimeStamp *inTimeStamp,
        UInt32 inBusNumber,
        UInt32 inNumberFrames,
        AudioBufferList *ioData)
{
    // static float theta;

    // SInt16 *left = (SInt16 *)ioData->mBuffers[0].mData;
    // for (UInt32 frame = 0; frame < inNumberFrames; ++frame) {
    //     left[frame] = (SInt16)(sin(theta) * 32767.0f);
    //     theta += M_TAU * TONE_FREQUENCY / SAMPLE_RATE;
    //     if (theta > M_TAU) {
    //         theta -= M_TAU;
    //     }
    // }

    static float theta = 0.01;
    static float amplitude = 0.25; // SignalLevel - Volume [0; 1]
    static float thetaIncrement = M_TAU * TONE_FREQUENCY / SAMPLE_RATE;

    float *left = (float *)ioData->mBuffers[0].mData;

    for (UInt32 frame = 0; frame < inNumberFrames; ++frame) {
        left[frame] = (float)(sin(theta) * amplitude);
        theta += thetaIncrement;
    }

    // Copy left channel to right channel
    memcpy(ioData->mBuffers[1].mData, left, ioData->mBuffers[1].mDataByteSize);

    return noErr;
}

int main() {
    OSErr err;

    AudioComponentDescription acd = {
        .componentType = kAudioUnitType_Output,
        .componentSubType = kAudioUnitSubType_DefaultOutput,
        .componentManufacturer = kAudioUnitManufacturer_Apple,
    };

    AudioComponent output = AudioComponentFindNext(NULL, &acd);
    if (!output) printf("Can't find default output\n");

    AudioUnit toneUnit;
    err = AudioComponentInstanceNew(output, &toneUnit);
    if (err) fprintf(stderr, "Error creating unit: %d\n", err);

    AURenderCallbackStruct input = { .inputProc = RenderSineWave };
    err = AudioUnitSetProperty(toneUnit, kAudioUnitProperty_SetRenderCallback,
            kAudioUnitScope_Input, 0, &input, sizeof(input));
    if (err) printf("Error setting callback: %d\n", err);

    AudioStreamBasicDescription asbd = {
        .mFormatID = kAudioFormatLinearPCM,
        .mFormatFlags = 0
            | kAudioFormatFlagIsSignedInteger
            | kAudioFormatFlagIsPacked
            | kAudioFormatFlagIsNonInterleaved,
        .mSampleRate = 48000,
        .mBitsPerChannel = 16,
        .mChannelsPerFrame = 2,
        .mFramesPerPacket = 1,
        .mBytesPerFrame = 2,
        .mBytesPerPacket = 2,
    };

    err = AudioUnitSetProperty(toneUnit, kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Input, 0, &asbd, sizeof(asbd));
    if (err) printf("Error setting stream format: %d\n", err);

    err = AudioUnitInitialize(toneUnit);
    if (err) printf("Error initializing unit: %d\n", err);

    err = AudioOutputUnitStart(toneUnit);
    if (err) printf("Error starting unit: %d\n", err);

    usleep(500000);

    AudioOutputUnitStop(toneUnit);
    AudioUnitUninitialize(toneUnit);
    AudioComponentInstanceDispose(toneUnit);
}

