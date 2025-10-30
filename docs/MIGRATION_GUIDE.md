# CoreMusic Migration Guide

**Version:** 0.1.8
**Last Updated:** October 2025

This guide helps you migrate from other Python audio libraries to CoreMusic, as well as port existing CoreAudio C/Objective-C code to Python.

---

## Table of Contents

1. [From pydub](#1-from-pydub)
2. [From soundfile / libsndfile](#2-from-soundfile--libsndfile)
3. [From wave / audioread](#3-from-wave--audioread)
4. [From mido (MIDI)](#4-from-mido-midi)
5. [From CoreAudio C/Objective-C](#5-from-coreaudio-cobjective-c)
6. [From AudioKit (Swift)](#6-from-audiokit-swift)
7. [Feature Comparison Matrix](#7-feature-comparison-matrix)

---

## 1. From pydub

**pydub** is a high-level audio library focused on simplicity. CoreMusic provides similar ease-of-use with native performance.

### 1.1 Loading Audio Files

**pydub:**
```python
from pydub import AudioSegment

# Load audio
audio = AudioSegment.from_wav("audio.wav")
audio = AudioSegment.from_mp3("audio.mp3")

# Get properties
duration = len(audio)  # milliseconds
sample_rate = audio.frame_rate
channels = audio.channels
```

**CoreMusic:**
```python
import coremusic as cm

# Load audio (supports WAV, MP3, AAC, AIFF, etc.)
with cm.AudioFile("audio.wav") as audio:
    # Get properties
    duration = audio.duration  # seconds
    sample_rate = audio.format.sample_rate
    channels = audio.format.channels_per_frame

# Or for any format with automatic conversion
with cm.ExtendedAudioFile("audio.mp3") as audio:
    format = audio.file_format
```

### 1.2 Basic Operations

**pydub:**
```python
from pydub import AudioSegment

# Load
audio = AudioSegment.from_wav("input.wav")

# Volume adjustment
louder = audio + 10  # Increase by 10dB
quieter = audio - 5  # Decrease by 5dB

# Slicing
first_10_seconds = audio[:10000]  # milliseconds

# Concatenation
combined = audio1 + audio2

# Export
audio.export("output.mp3", format="mp3")
```

**CoreMusic:**
```python
import coremusic as cm
import numpy as np

# Load
with cm.AudioFile("input.wav") as audio:
    data, count = audio.read(audio.frame_count)
    samples = np.frombuffer(data, dtype=np.float32)

    # Volume adjustment (in place)
    samples *= 1.26  # +10dB ≈ 3.16x, +5dB ≈ 1.78x
    samples *= 0.56  # -5dB ≈ 0.56x

    # Slicing (use audio slicer module)
    from coremusic.audio import AudioSlicer
    slicer = AudioSlicer("input.wav")
    first_10_seconds = slicer.slice_time_range(0.0, 10.0)

    # Export
    with cm.ExtendedAudioFile.create(
        "output.wav",
        cm.capi.fourchar_to_int('WAVE'),
        audio.format
    ) as output:
        output.write(count, samples.tobytes())
```

### 1.3 Effects

**pydub:**
```python
from pydub import AudioSegment

audio = AudioSegment.from_wav("input.wav")

# Fade in/out
audio = audio.fade_in(2000).fade_out(2000)

# Reverse
reversed = audio.reverse()

# Speed up (changes pitch)
faster = audio.speedup(playback_speed=1.5)
```

**CoreMusic:**
```python
import coremusic as cm
import numpy as np

with cm.AudioFile("input.wav") as audio:
    data, count = audio.read(audio.frame_count)
    samples = np.frombuffer(data, dtype=np.float32)

    # Fade in/out
    fade_len = int(2.0 * audio.format.sample_rate)
    fade_in = np.linspace(0, 1, fade_len)
    samples[:fade_len] *= fade_in

    # Reverse
    samples = samples[::-1]

    # For time-stretching/pitch-shifting, use AudioConverter
    # or AudioUnit time-stretching effects
```

### 1.4 Key Differences

| Feature | pydub | CoreMusic |
|---------|-------|-----------|
| **Dependencies** | ffmpeg (external) | None (uses macOS CoreAudio) |
| **Performance** | Slow (shell calls) | Fast (native C) |
| **Format Support** | Via ffmpeg | Native macOS formats |
| **Audio Processing** | Limited | Full CoreAudio access |
| **Real-time** | No | Yes (AudioQueue, AudioUnit) |
| **Platform** | Cross-platform | macOS only |

---

## 2. From soundfile / libsndfile

**soundfile** provides NumPy-friendly audio I/O. CoreMusic offers similar functionality with better macOS integration.

### 2.1 Reading Audio

**soundfile:**
```python
import soundfile as sf

# Read entire file
data, samplerate = sf.read("audio.wav")

# Read specific frames
data, samplerate = sf.read("audio.wav", start=1000, stop=5000)

# Read file info
info = sf.info("audio.wav")
print(info.samplerate, info.channels, info.duration)
```

**CoreMusic:**
```python
import coremusic as cm
import numpy as np

# Read entire file
with cm.AudioFile("audio.wav") as audio:
    data_bytes, count = audio.read(audio.frame_count)
    data = np.frombuffer(data_bytes, dtype=np.float32)
    samplerate = audio.format.sample_rate

# Read specific frames
with cm.AudioFile("audio.wav") as audio:
    # Seek to frame 1000
    data_bytes, count = audio.read_packets(start=1000, count=4000)

# Read file info
with cm.AudioFile("audio.wav") as audio:
    print(audio.format.sample_rate)
    print(audio.format.channels_per_frame)
    print(audio.duration)
```

### 2.2 Writing Audio

**soundfile:**
```python
import soundfile as sf
import numpy as np

# Generate audio
samplerate = 44100
data = np.random.randn(44100, 2)  # 1 second stereo

# Write file
sf.write("output.wav", data, samplerate)
```

**CoreMusic:**
```python
import coremusic as cm
import numpy as np

# Generate audio
samplerate = 44100
data = np.random.randn(44100 * 2).astype(np.float32)  # Interleaved stereo

# Create format
format = cm.AudioFormat(
    sample_rate=samplerate,
    format_id='lpcm',
    channels_per_frame=2,
    bits_per_channel=32,
    is_float=True
)

# Write file
with cm.ExtendedAudioFile.create(
    "output.wav",
    cm.capi.fourchar_to_int('WAVE'),
    format
) as output:
    output.write(num_frames=44100, audio_data=data.tobytes())
```

### 2.3 Format Conversion

**soundfile:**
```python
import soundfile as sf

# Read in one format
data, samplerate = sf.read("input.flac")

# Write in another format
sf.write("output.wav", data, samplerate, subtype='PCM_16')
```

**CoreMusic:**
```python
import coremusic as cm

# Automatic format conversion
with cm.ExtendedAudioFile("input.aiff") as input_file:
    # Set desired output format
    output_format = cm.AudioFormat(
        sample_rate=44100.0,
        format_id='lpcm',
        channels_per_frame=2,
        bits_per_channel=16
    )

    # Create output file
    with cm.ExtendedAudioFile.create(
        "output.wav",
        cm.capi.fourchar_to_int('WAVE'),
        output_format
    ) as output_file:
        # Copy with automatic conversion
        while True:
            data, count = input_file.read(4096)
            if count == 0:
                break
            output_file.write(count, data)
```

---

## 3. From wave / audioread

### 3.1 Basic File Reading

**wave:**
```python
import wave

with wave.open("audio.wav", "rb") as wav:
    channels = wav.getnchannels()
    sample_width = wav.getsampwidth()
    framerate = wav.getframerate()
    n_frames = wav.getnframes()

    # Read audio
    audio_data = wav.readframes(n_frames)
```

**CoreMusic:**
```python
import coremusic as cm

with cm.AudioFile("audio.wav") as audio:
    channels = audio.format.channels_per_frame
    sample_width = audio.format.bits_per_channel // 8
    framerate = audio.format.sample_rate
    n_frames = audio.frame_count

    # Read audio
    audio_data, count = audio.read(n_frames)
```

### 3.2 Writing WAV Files

**wave:**
```python
import wave
import struct

with wave.open("output.wav", "wb") as wav:
    wav.setnchannels(2)  # Stereo
    wav.setsampwidth(2)  # 16-bit
    wav.setframerate(44100)

    # Write audio
    wav.writeframes(audio_data)
```

**CoreMusic:**
```python
import coremusic as cm

format = cm.AudioFormat(
    sample_rate=44100.0,
    format_id='lpcm',
    channels_per_frame=2,
    bits_per_channel=16
)

with cm.ExtendedAudioFile.create(
    "output.wav",
    cm.capi.fourchar_to_int('WAVE'),
    format
) as audio:
    # Write audio
    audio.write(num_frames=len(audio_data)//4, audio_data=audio_data)
```

---

## 4. From mido (MIDI)

**mido** is a popular MIDI library. CoreMusic provides full CoreMIDI and MusicPlayer integration.

### 4.1 MIDI File Reading

**mido:**
```python
import mido

# Load MIDI file
mid = mido.MidiFile("song.mid")

print(f"Tempo: {mid.tempo}")
print(f"Tracks: {len(mid.tracks)}")

for track in mid.tracks:
    for msg in track:
        if msg.type == "note_on":
            print(f"Note: {msg.note}, Velocity: {msg.velocity}")
```

**CoreMusic:**
```python
import coremusic as cm
from coremusic.midi import load_midi_file

# Load MIDI file
sequence = load_midi_file("song.mid")

print(f"Tempo: {sequence.tempo} BPM")
print(f"Duration: {sequence.duration:.2f}s")
print(f"Tracks: {len(sequence.tracks)}")

for track in sequence.tracks:
    print(f"{track.name}: {len(track.notes)} notes")
    for note in track.notes[:10]:  # First 10 notes
        print(f"  Note: {note.pitch}, Velocity: {note.velocity}")
```

### 4.2 MIDI Playback

**mido:**
```python
import mido
from mido import MidiFile
import time

mid = MidiFile("song.mid")

# Requires external MIDI output setup
port = mido.open_output()

for msg in mid.play():
    port.send(msg)
```

**CoreMusic:**
```python
import coremusic as cm

# Simple playback using MusicPlayer
player = cm.MusicPlayer()
sequence = cm.MusicSequence()

# Load MIDI file
sequence.load_from_file("song.mid")

# Play through default MIDI device
player.sequence = sequence
player.preroll()
player.start()

# Wait for playback
import time
time.sleep(sequence.duration)

player.stop()
player.dispose()
sequence.dispose()
```

### 4.3 Creating MIDI Sequences

**mido:**
```python
from mido import MidiFile, MidiTrack, Message

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

track.append(Message('program_change', program=0))
track.append(Message('note_on', note=60, velocity=100, time=0))
track.append(Message('note_off', note=60, velocity=100, time=480))

mid.save("output.mid")
```

**CoreMusic:**
```python
import coremusic as cm

# Create sequence
sequence = cm.MusicSequence()
track = sequence.new_track()

# Set tempo
sequence.tempo_track.add_tempo_event(0.0, bpm=120.0)

# Add program change
track.add_midi_channel_event(0.0, status=0xC0, data1=0)  # Piano

# Add note
track.add_midi_note(
    time=0.0,
    channel=0,
    note=60,
    velocity=100,
    duration=1.0  # 1 beat
)

# Save (requires additional implementation)
# Or play directly
player = cm.MusicPlayer()
player.sequence = sequence
player.start()
```

---

## 5. From CoreAudio C/Objective-C

### 5.1 Audio File Operations

**C/Objective-C:**
```c
#include <AudioToolbox/AudioToolbox.h>

AudioFileID audioFile;
CFURLRef fileURL = CFURLCreateFromFileSystemRepresentation(
    NULL, (UInt8*)"audio.wav", strlen("audio.wav"), false
);

OSStatus status = AudioFileOpenURL(
    fileURL, kAudioFileReadPermission, 0, &audioFile
);

// Get format
AudioStreamBasicDescription asbd;
UInt32 size = sizeof(asbd);
AudioFileGetProperty(
    audioFile, kAudioFilePropertyDataFormat, &size, &asbd
);

// Read packets
UInt32 numPackets = 1024;
AudioBufferList bufferList;
AudioFileReadPackets(
    audioFile, false, &numBytes, NULL, 0, &numPackets, bufferList.mBuffers[0].mData
);

AudioFileClose(audioFile);
```

**CoreMusic:**
```python
import coremusic as cm

# Much simpler!
with cm.AudioFile("audio.wav") as audio:
    # Get format
    format = audio.format
    print(f"Sample rate: {format.sample_rate}")
    print(f"Channels: {format.channels_per_frame}")

    # Read packets
    data, num_packets = audio.read_packets(start=0, count=1024)

# Automatic cleanup
```

### 5.2 Audio Queue

**C/Objective-C:**
```c
#include <AudioToolbox/AudioQueue.h>

AudioQueueRef queue;
AudioStreamBasicDescription format;
// ... setup format ...

// Create queue
AudioQueueNewOutput(&format, audioCallback, userData, NULL, NULL, 0, &queue);

// Allocate buffers
AudioQueueBufferRef buffers[3];
for (int i = 0; i < 3; i++) {
    AudioQueueAllocateBuffer(queue, bufferSize, &buffers[i]);
}

// Start
AudioQueueStart(queue, NULL);

// ... later ...
AudioQueueStop(queue, true);
AudioQueueDispose(queue, true);
```

**CoreMusic:**
```python
import coremusic as cm

# Create format
format = cm.AudioFormat(
    sample_rate=44100.0,
    format_id='lpcm',
    channels_per_frame=2,
    bits_per_channel=16
)

# Create queue (with context manager)
with cm.AudioQueue.create_output(format) as queue:
    # Allocate buffers
    buffers = [queue.allocate_buffer(4096) for _ in range(3)]

    # Start
    queue.start()

    # ... use queue ...

    # Stop
    queue.stop()

# Automatic disposal
```

### 5.3 MusicPlayer

**C/Objective-C:**
```c
#include <AudioToolbox/MusicPlayer.h>

MusicPlayer player;
MusicSequence sequence;
MusicTrack track;

NewMusicPlayer(&player);
NewMusicSequence(&sequence);
MusicSequenceNewTrack(sequence, &track);

// Add note
MusicTimeStamp time = 0;
MIDINoteMessage note;
note.channel = 0;
note.note = 60;
note.velocity = 100;
note.releaseVelocity = 64;
note.duration = 1.0;

MusicTrackNewMIDINoteEvent(track, time, &note);

MusicPlayerSetSequence(player, sequence);
MusicPlayerPreroll(player);
MusicPlayerStart(player);

// ... cleanup ...
DisposeMusicPlayer(player);
DisposeMusicSequence(sequence);
```

**CoreMusic:**
```python
import coremusic as cm

# Much cleaner!
with cm.MusicPlayer() as player:
    sequence = cm.MusicSequence()
    track = sequence.new_track()

    # Add note
    track.add_midi_note(
        time=0.0,
        channel=0,
        note=60,
        velocity=100,
        release_velocity=64,
        duration=1.0
    )

    # Play
    player.sequence = sequence
    player.preroll()
    player.start()

    # Automatic cleanup
```

---

## 6. From AudioKit (Swift)

**AudioKit** is a popular Swift audio framework. CoreMusic provides Python bindings to the same underlying CoreAudio APIs.

### 6.1 Audio Player

**AudioKit (Swift):**
```swift
import AudioKit

let player = AudioPlayer()
do {
    try player.load(url: URL(fileURLWithPath: "audio.wav"))
    player.play()
} catch {
    print("Error loading audio")
}
```

**CoreMusic:**
```python
import coremusic as cm

# Using AudioPlayer utility class
player = cm.AudioPlayer("audio.wav")
player.play()

# Or using AudioQueue for more control
with cm.AudioFile("audio.wav") as audio:
    format = audio.format
    queue = cm.AudioQueue.create_output(format)
    # ... set up playback ...
```

### 6.2 MIDI Sequencing

**AudioKit (Swift):**
```swift
import AudioKit

let sequencer = AppleSequencer()
let track = sequencer.newTrack()

track.add(noteNumber: 60, velocity: 100, position: 0, duration: 1)
try? sequencer.play()
```

**CoreMusic:**
```python
import coremusic as cm

# Very similar API!
player = cm.MusicPlayer()
sequence = cm.MusicSequence()
track = sequence.new_track()

track.add_midi_note(time=0.0, channel=0, note=60, velocity=100, duration=1.0)

player.sequence = sequence
player.start()
```

---

## 7. Feature Comparison Matrix

| Feature | pydub | soundfile | wave | mido | CoreAudio C | AudioKit | **CoreMusic** |
|---------|-------|-----------|------|------|-------------|----------|---------------|
| **File Formats** | ⭐⭐⭐ (via ffmpeg) | ⭐⭐ | ⭐ | MIDI only | ⭐⭐⭐ | ⭐⭐⭐ | **⭐⭐⭐** |
| **Performance** | ⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | **⭐⭐⭐** |
| **Ease of Use** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ | **⭐⭐⭐** |
| **NumPy Integration** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | **✅** |
| **Real-time Audio** | ❌ | ❌ | ❌ | ⚠️ | ✅ | ✅ | **✅** |
| **MIDI Support** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | **✅** |
| **AudioUnit Support** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | **✅** |
| **Platform** | Cross | Cross | Cross | Cross | macOS | iOS/macOS | **macOS** |
| **Dependencies** | ffmpeg | libsndfile | stdlib | None | None | None | **None** |
| **License** | MIT | BSD | PSF | MIT | Apple | MIT | **MIT** |

**Legend:**
- ⭐⭐⭐ = Excellent
- ⭐⭐ = Good
- ⭐ = Basic
- ✅ = Supported
- ⚠️ = Partial support
- ❌ = Not supported

---

## 8. Migration Checklist

When migrating to CoreMusic:

- [ ] Identify which operations you need (file I/O, real-time, MIDI, etc.)
- [ ] Choose appropriate API level (OO for convenience, functional for performance)
- [ ] Update import statements
- [ ] Convert file paths to strings (CoreMusic accepts Path objects too)
- [ ] Update audio data handling (CoreMusic returns bytes, convert to NumPy if needed)
- [ ] Use context managers for automatic resource cleanup
- [ ] Test on target macOS version
- [ ] Profile performance if critical
- [ ] Update error handling (CoreMusic uses specific exception types)
- [ ] Check format compatibility (CoreMusic uses native macOS formats)

---

## 9. Common Pitfalls

### 9.1 Audio Data Format

**Other libraries** often return NumPy arrays directly:
```python
# soundfile
data, sr = sf.read("audio.wav")  # Returns NumPy array
```

**CoreMusic** returns bytes (for performance):
```python
# CoreMusic
with cm.AudioFile("audio.wav") as audio:
    data_bytes, count = audio.read(1024)
    # Convert to NumPy if needed
    data = np.frombuffer(data_bytes, dtype=np.float32)
```

### 9.2 Sample vs Frame Counting

**Frames** = samples across all channels at one time point
**Samples** = individual channel values

```python
# Example: Stereo audio, 1024 frames
# = 2048 samples (1024 frames * 2 channels)

with cm.AudioFile("stereo.wav") as audio:
    # Request 1024 FRAMES
    data_bytes, frame_count = audio.read(1024)

    # Get 2048 SAMPLES
    samples = np.frombuffer(data_bytes, dtype=np.float32)
    assert len(samples) == frame_count * audio.format.channels_per_frame
```

### 9.3 Resource Cleanup

**Always use context managers** or explicit disposal:

```python
# GOOD
with cm.AudioFile("audio.wav") as audio:
    data = audio.read(1024)
# Automatic cleanup

# ALSO GOOD
audio = cm.AudioFile("audio.wav")
audio.open()
try:
    data = audio.read(1024)
finally:
    audio.dispose()

# BAD - May leak resources
audio = cm.AudioFile("audio.wav")
audio.open()
data = audio.read(1024)
# Forgot to close!
```

---

## 10. Getting Help

**Resources:**
- CoreMusic Documentation: `docs/`
- API Reference: Use `help(cm.AudioFile)` in Python
- Examples: `tests/demos/`
- Issue Tracker: https://github.com/anthropics/coremusic/issues

**Common Questions:**
- Performance issues? See `docs/PERFORMANCE_GUIDE.md`
- API usage? See `docs/COOKBOOK.md`
- Core concepts? See `CLAUDE.md` in repository

---

**Ready to migrate?** Start with simple file I/O operations and gradually adopt more advanced features!
