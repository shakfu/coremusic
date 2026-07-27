# Migration Guide

Guide for migrating from other Python audio libraries to CoreMusic, porting
CoreAudio C/Objective-C code to Python, and updating code written against older
CoreMusic releases.

## From CoreMusic 0.2.2 and Earlier

0.2.3 dissolved the `coremusic.objects` package into domain subpackages, and
there is no compatibility shim: `import coremusic.objects` raises
`ModuleNotFoundError`. The top-level package exports only `__version__`, so
`coremusic.AudioFile` and similar flat names do not resolve either.

| Old import | New import |
| --- | --- |
| `from coremusic.objects import AudioFile, AudioFormat` | `from coremusic.audio import AudioFile, AudioFormat` |
| `from coremusic.objects import AudioQueue, AudioUnit, AUGraph` | `from coremusic.audio import AudioQueue, AudioUnit, AUGraph` |
| `from coremusic.objects import AudioDevice, AudioDeviceManager` | `from coremusic.audio import AudioDevice, AudioDeviceManager` |
| `from coremusic.objects import AudioClock, ClockTimeFormat` | `from coremusic.audio import AudioClock, ClockTimeFormat` |
| `from coremusic.objects import MIDIClient, MIDIPort` | `from coremusic.midi import MIDIClient, MIDIPort` |
| `from coremusic.objects import MusicPlayer, MusicSequence, MusicTrack` | `from coremusic.midi import MusicPlayer, MusicSequence, MusicTrack` |
| `from coremusic.objects import AudioFileError, MIDIError` | `from coremusic.exceptions import AudioFileError, MIDIError` |
| `from coremusic.objects import CoreAudioObject, AudioPlayer, NUMPY_AVAILABLE` | `from coremusic.base import CoreAudioObject, AudioPlayer, NUMPY_AVAILABLE` |
| `import coremusic as cm; cm.AudioFile(...)` | `from coremusic.audio import AudioFile` |

The MIDI CLI was restructured at the same time - `midi device list`,
`midi input monitor`, and `midi output send` became `midi list`,
`midi monitor`, and `midi send`. See the [CLI Guide](cli.md).

The [Import Guide](imports.md) has the full package map.

## From pydub

**pydub** is a high-level audio library focused on simplicity. CoreMusic provides similar ease-of-use with native performance.

### Loading Audio Files

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
--8<-- "examples/guides/migration/from_pydub.py:load"
```

### Basic Operations

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
--8<-- "examples/guides/migration/from_pydub.py:operations"
```

### Key Differences

| Feature | pydub | CoreMusic |
|---------|-------|-----------|
| Performance | Relies on ffmpeg | Native CoreAudio |
| Memory Usage | High (loads all) | Low (streaming) |
| Platform | Cross-platform | macOS only |
| Real-time Audio | No | Yes (AudioUnit) |
| MIDI Support | No | Yes (CoreMIDI) |
| Dependencies | ffmpeg required | No external deps |
| Type | Immutable segments | Mutable buffers |

## From soundfile / libsndfile

**soundfile** provides NumPy-based audio I/O. CoreMusic offers similar functionality with deeper macOS integration.

### Reading Audio

**soundfile:**

```python
import soundfile as sf

# Read entire file
data, sample_rate = sf.read("audio.wav")

# Read with specific dtype
data, sample_rate = sf.read("audio.wav", dtype='float32')

# Get info without reading
info = sf.info("audio.wav")
print(f"Duration: {info.duration}s")
print(f"Channels: {info.channels}")
```

**CoreMusic:**

```python
--8<-- "examples/guides/migration/from_soundfile.py:read"
```

### Writing Audio

**soundfile:**

```python
import soundfile as sf
import numpy as np

# Generate audio
data = np.random.randn(44100 * 2)  # 2 seconds

# Write
sf.write("output.wav", data, 44100)
```

**CoreMusic:**

```python
--8<-- "examples/guides/migration/from_soundfile.py:write"
```

### Streaming

**soundfile:**

```python
import soundfile as sf

# Read in blocks
with sf.SoundFile("audio.wav") as file:
    while True:
        data = file.read(1024)
        if len(data) == 0:
            break
        # Process block
```

**CoreMusic:**

```python
--8<-- "examples/guides/migration/from_soundfile.py:stream"
```

## From wave / audioread

**wave** is Python's built-in WAV module. CoreMusic provides more features and better performance.

### Reading WAV

**wave:**

```python
import wave

with wave.open("audio.wav", 'rb') as wav:
    # Get parameters
    channels = wav.getnchannels()
    sample_width = wav.getsampwidth()
    framerate = wav.getframerate()
    n_frames = wav.getnframes()

    # Read frames
    frames = wav.readframes(n_frames)
```

**CoreMusic:**

```python
--8<-- "examples/guides/migration/from_soundfile.py:wave-read"
```

### Writing WAV

**wave:**

```python
import wave
import numpy as np

data = np.random.randint(-32768, 32767, 44100, dtype=np.int16)

with wave.open("output.wav", 'wb') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(44100)
    wav.writeframes(data.tobytes())
```

**CoreMusic:**

```python
--8<-- "examples/guides/migration/from_soundfile.py:wave-write"
```

## From mido (MIDI)

**mido** is a popular MIDI library. CoreMusic provides CoreMIDI access for macOS.

### Opening MIDI Ports

**mido:**

```python
import mido

# List ports
print(mido.get_output_names())

# Open output port
with mido.open_output('IAC Driver Bus 1') as port:
    msg = mido.Message('note_on', note=60, velocity=100)
    port.send(msg)
```

**CoreMusic:**

```python
--8<-- "examples/guides/migration/from_mido.py:ports"
```

### MIDI Files

**mido:**

```python
import mido

# Load MIDI file
mid = mido.MidiFile("song.mid")

# Iterate through messages
for track in mid.tracks:
    for msg in track:
        print(msg)

# Create new file
mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)

track.append(mido.Message('note_on', note=60, time=0))
track.append(mido.Message('note_off', note=60, time=480))

mid.save("output.mid")
```

**CoreMusic:**

```python
--8<-- "examples/guides/migration/from_mido.py:files"
```

## From CoreAudio C/Objective-C

Migrating existing CoreAudio code to Python with CoreMusic.

### AudioFile Operations

**C/Objective-C:**

```c
// Open audio file
AudioFileID fileID;
CFURLRef fileURL = CFURLCreateFromFileSystemRepresentation(
    NULL, (const UInt8 *)"/path/to/audio.wav", strlen("/path/to/audio.wav"), false
);
OSStatus status = AudioFileOpenURL(fileURL, kAudioFileReadPermission, 0, &fileID);

// Get format
AudioStreamBasicDescription format;
UInt32 size = sizeof(format);
AudioFileGetProperty(fileID, kAudioFilePropertyDataFormat, &size, &format);

// Read packets
UInt32 numPackets = 1024;
void *buffer = malloc(numPackets * format.mBytesPerPacket);
AudioFileReadPacketData(fileID, false, &size, NULL, 0, &numPackets, buffer);

// Cleanup
AudioFileClose(fileID);
free(buffer);
```

**CoreMusic:**

```python
--8<-- "examples/guides/migration/from_c_api.py:audiofile-oo"
```

Or using functional API for closer C mapping:

```python
--8<-- "examples/guides/migration/from_c_api.py:audiofile-functional"
```

### AudioUnit Operations

**C/Objective-C:**

```c
// Find output unit
AudioComponentDescription desc;
desc.componentType = kAudioUnitType_Output;
desc.componentSubType = kAudioUnitSubType_DefaultOutput;
desc.componentManufacturer = kAudioUnitManufacturer_Apple;

AudioComponent comp = AudioComponentFindNext(NULL, &desc);
AudioUnit unit;
AudioComponentInstanceNew(comp, &unit);

// Initialize and start
AudioUnitInitialize(unit);
AudioOutputUnitStart(unit);
```

**CoreMusic:**

```python
--8<-- "examples/guides/migration/from_c_api.py:audiounit"
```

### MIDI Operations

**C/Objective-C:**

```objc
// Create MIDI client
MIDIClientRef client;
MIDIClientCreate(CFSTR("MyClient"), NULL, NULL, &client);

// Create output port
MIDIPortRef outputPort;
MIDIOutputPortCreate(client, CFSTR("Output"), &outputPort);

// Get destination
MIDIEndpointRef dest = MIDIGetDestination(0);

// Send note
Byte packet[3] = {0x90, 60, 100};  // Note on
MIDISend(outputPort, dest, packet, 3);
```

**CoreMusic:**

```python
--8<-- "examples/guides/migration/from_c_api.py:midi"
```

## From AudioKit (Swift)

**AudioKit** is a powerful Swift framework. CoreMusic provides similar capabilities in Python.

### Audio Playback

**AudioKit (Swift):**

```swift
import AudioKit

let file = try AVAudioFile(forReading: URL(fileURLWithPath: "audio.wav"))
let player = AudioPlayer(file: file)
AudioKit.output = player
try AudioKit.start()
player.play()
```

**CoreMusic:**

```python
--8<-- "examples/guides/migration/from_audiokit.py:playback"
```

### Audio Effects

**AudioKit (Swift):**

```swift
import AudioKit

let player = AudioPlayer(file: file)
let reverb = Reverb(player)
reverb.dryWetMix = 0.5

AudioKit.output = reverb
try AudioKit.start()
```

**CoreMusic:**

```python
--8<-- "examples/guides/migration/from_audiokit.py:effects"
```

## Feature Comparison Matrix

| Feature | pydub | soundfile | wave | mido | CoreAudio | CoreMusic |
|---------|-------|-----------|------|------|-----------|-----------|
| Audio File I/O | Yes | Yes | Yes | No | Yes | Yes |
| Format Conversion | Yes | No | No | No | Yes | Yes |
| Real-time Audio | No | No | No | No | Yes | Yes |
| AudioUnit Support | No | No | No | No | Yes | Yes |
| MIDI I/O | No | No | No | Yes | Yes | Yes |
| MIDI Files | No | No | No | Yes | Yes | Yes |
| Hardware Control | No | No | No | No | Yes | Yes |
| Streaming | Limited | Yes | Limited | N/A | Yes | Yes |
| NumPy Integration | Limited | Yes | No | No | No | Yes |
| Cross-platform | Yes | Yes | Yes | Yes | No | No |
| External Dependencies | ffmpeg | libsndfile | None | None | None | None |
| Performance | Medium | High | Low | High | Native | Native |

## Migration Checklist

When migrating to CoreMusic:

1. **Identify Dependencies**

   - Check if your code relies on cross-platform support
   - Verify macOS version compatibility (10.13+)
   - List external tools (ffmpeg, etc.)

2. **Update Imports**

   - Replace library imports with CoreMusic
   - Update API calls to CoreMusic equivalents
   - Add NumPy if processing audio data

3. **Adapt Audio Operations**

   - Convert high-level operations to CoreMusic patterns
   - Update file I/O to use AudioFile/ExtendedAudioFile
   - Migrate streaming code to chunked processing

4. **Update MIDI Code**

   - Replace MIDI library calls with CoreMIDI via CoreMusic
   - Adapt port discovery and device enumeration
   - Update message sending/receiving patterns

5. **Test Thoroughly**

   - Verify audio quality and correctness
   - Check resource cleanup and memory usage
   - Test error handling and edge cases
   - Benchmark performance improvements

## Common Migration Patterns

### Pattern 1: Simple Audio Processing

**Before (pydub):**

```python
from pydub import AudioSegment

audio = AudioSegment.from_wav("input.wav")
audio = audio + 6  # Increase volume
audio.export("output.wav", format="wav")
```

**After (CoreMusic):**

```python
--8<-- "examples/guides/migration/patterns.py:audio"
```

### Pattern 2: MIDI Processing

**Before (mido):**

```python
import mido

with mido.open_output() as port:
    for note in [60, 64, 67]:
        msg = mido.Message('note_on', note=note)
        port.send(msg)
```

**After (CoreMusic):**

```python
--8<-- "examples/guides/migration/patterns.py:midi"
```

## See Also

- [Practical recipes](../cookbook/index.md)
- [Import patterns](imports.md)
- [Performance optimization](performance.md)
- [Complete API reference](../api/index.md)

!!! note
    Every snippet on this page is a runnable program under `examples/guides/migration/`.
