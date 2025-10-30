Migration Guide
===============

**Version:** 0.1.8

Guide for migrating from other Python audio libraries to CoreMusic, and porting CoreAudio C/Objective-C code to Python.

.. contents:: Table of Contents
   :local:
   :depth: 2

From pydub
----------

**pydub** is a high-level audio library focused on simplicity. CoreMusic provides similar ease-of-use with native performance.

Loading Audio Files
^^^^^^^^^^^^^^^^^^^

**pydub:**

.. code-block:: python

   from pydub import AudioSegment

   # Load audio
   audio = AudioSegment.from_wav("audio.wav")
   audio = AudioSegment.from_mp3("audio.mp3")

   # Get properties
   duration = len(audio)  # milliseconds
   sample_rate = audio.frame_rate
   channels = audio.channels

**CoreMusic:**

.. code-block:: python

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

Basic Operations
^^^^^^^^^^^^^^^^

**pydub:**

.. code-block:: python

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

**CoreMusic:**

.. code-block:: python

   import coremusic as cm
   import numpy as np

   # Load
   with cm.AudioFile("input.wav") as audio:
       data, count = audio.read(audio.frame_count)
       samples = np.frombuffer(data, dtype=np.float32)

       # Volume adjustment (in place)
       samples *= 1.26  # +10dB ≈ 3.16x
       samples *= 0.56  # -5dB ≈ 0.56x

       # Slicing
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

Key Differences
^^^^^^^^^^^^^^^

==================== ======================== =========================
Feature              pydub                    CoreMusic
==================== ======================== =========================
Performance          Relies on ffmpeg         Native CoreAudio
Memory Usage         High (loads all)         Low (streaming)
Platform             Cross-platform           macOS only
Real-time Audio      No                       Yes (AudioUnit)
MIDI Support         No                       Yes (CoreMIDI)
Dependencies         ffmpeg required          No external deps
Type                 Immutable segments       Mutable buffers
==================== ======================== =========================

From soundfile / libsndfile
----------------------------

**soundfile** provides NumPy-based audio I/O. CoreMusic offers similar functionality with deeper macOS integration.

Reading Audio
^^^^^^^^^^^^^

**soundfile:**

.. code-block:: python

   import soundfile as sf

   # Read entire file
   data, sample_rate = sf.read("audio.wav")

   # Read with specific dtype
   data, sample_rate = sf.read("audio.wav", dtype='float32')

   # Get info without reading
   info = sf.info("audio.wav")
   print(f"Duration: {info.duration}s")
   print(f"Channels: {info.channels}")

**CoreMusic:**

.. code-block:: python

   import coremusic as cm
   import numpy as np

   # Read entire file
   with cm.AudioFile("audio.wav") as audio:
       data, count = audio.read(audio.frame_count)
       samples = np.frombuffer(data, dtype=np.float32)
       sample_rate = audio.format.sample_rate

   # Get info without reading
   with cm.AudioFile("audio.wav") as audio:
       duration = audio.duration
       channels = audio.format.channels_per_frame
       sample_rate = audio.format.sample_rate

Writing Audio
^^^^^^^^^^^^^

**soundfile:**

.. code-block:: python

   import soundfile as sf
   import numpy as np

   # Generate audio
   data = np.random.randn(44100 * 2)  # 2 seconds

   # Write
   sf.write("output.wav", data, 44100)

**CoreMusic:**

.. code-block:: python

   import coremusic as cm
   import numpy as np

   # Generate audio
   data = np.random.randn(44100 * 2).astype(np.float32)

   # Create format
   format = cm.AudioFormat(
       sample_rate=44100.0,
       format_id=cm.capi.fourchar_to_int('lpcm'),
       format_flags=cm.capi.get_linear_pcm_format_flag_is_float(),
       channels_per_frame=1,
       bits_per_channel=32
   )

   # Write
   with cm.ExtendedAudioFile.create(
       "output.wav",
       cm.capi.fourchar_to_int('WAVE'),
       format
   ) as audio:
       audio.write(len(data), data.tobytes())

Streaming
^^^^^^^^^

**soundfile:**

.. code-block:: python

   import soundfile as sf

   # Read in blocks
   with sf.SoundFile("audio.wav") as file:
       while True:
           data = file.read(1024)
           if len(data) == 0:
               break
           # Process block

**CoreMusic:**

.. code-block:: python

   import coremusic as cm

   # Read in blocks
   with cm.AudioFile("audio.wav") as audio:
       while True:
           data, count = audio.read(1024)
           if count == 0:
               break
           # Process block

From wave / audioread
----------------------

**wave** is Python's built-in WAV module. CoreMusic provides more features and better performance.

Reading WAV
^^^^^^^^^^^

**wave:**

.. code-block:: python

   import wave

   with wave.open("audio.wav", 'rb') as wav:
       # Get parameters
       channels = wav.getnchannels()
       sample_width = wav.getsampwidth()
       framerate = wav.getframerate()
       n_frames = wav.getnframes()

       # Read frames
       frames = wav.readframes(n_frames)

**CoreMusic:**

.. code-block:: python

   import coremusic as cm

   with cm.AudioFile("audio.wav") as audio:
       # Get parameters
       channels = audio.format.channels_per_frame
       sample_rate = audio.format.sample_rate
       bits = audio.format.bits_per_channel
       n_frames = audio.frame_count

       # Read frames
       data, count = audio.read(n_frames)

Writing WAV
^^^^^^^^^^^

**wave:**

.. code-block:: python

   import wave
   import numpy as np

   data = np.random.randint(-32768, 32767, 44100, dtype=np.int16)

   with wave.open("output.wav", 'wb') as wav:
       wav.setnchannels(1)
       wav.setsampwidth(2)
       wav.setframerate(44100)
       wav.writeframes(data.tobytes())

**CoreMusic:**

.. code-block:: python

   import coremusic as cm
   import numpy as np

   data = np.random.randint(-32768, 32767, 44100, dtype=np.int16)

   format = cm.AudioFormat(
       sample_rate=44100.0,
       format_id=cm.capi.fourchar_to_int('lpcm'),
       format_flags=cm.capi.get_linear_pcm_format_flag_is_signed_integer(),
       channels_per_frame=1,
       bits_per_channel=16
   )

   with cm.ExtendedAudioFile.create(
       "output.wav",
       cm.capi.fourchar_to_int('WAVE'),
       format
   ) as audio:
       audio.write(len(data), data.tobytes())

From mido (MIDI)
----------------

**mido** is a popular MIDI library. CoreMusic provides CoreMIDI access for macOS.

Opening MIDI Ports
^^^^^^^^^^^^^^^^^^

**mido:**

.. code-block:: python

   import mido

   # List ports
   print(mido.get_output_names())

   # Open output port
   with mido.open_output('IAC Driver Bus 1') as port:
       msg = mido.Message('note_on', note=60, velocity=100)
       port.send(msg)

**CoreMusic:**

.. code-block:: python

   import coremusic.capi as capi

   # List ports
   num_dests = capi.midi_get_number_of_destinations()
   for i in range(num_dests):
       dest = capi.midi_get_destination(i)
       name = capi.midi_object_get_string_property(dest, "name")
       print(name)

   # Send MIDI
   client = capi.midi_client_create("MyApp")
   port = capi.midi_output_port_create(client, "Output")
   dest = capi.midi_get_destination(0)

   # Send note on
   note_on = bytes([0x90, 60, 100])  # Channel 0, note 60, velocity 100
   capi.midi_send(port, dest, note_on)

MIDI Files
^^^^^^^^^^

**mido:**

.. code-block:: python

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

**CoreMusic:**

.. code-block:: python

   import coremusic as cm

   # Load MIDI file
   sequence = cm.MusicSequence()
   sequence.load_from_file("song.mid")

   # Iterate through tracks
   for i in range(sequence.track_count):
       track = sequence.get_track(i)
       # Access track data

   # Create new sequence
   sequence = cm.MusicSequence()
   track = sequence.new_track()

   # Add notes
   track.add_midi_note(
       time=0.0,
       channel=0,
       note=60,
       velocity=100,
       duration=1.0
   )

   # Save (using functional API)
   # sequence.save_to_file("output.mid")  # OO API method

From CoreAudio C/Objective-C
-----------------------------

Migrating existing CoreAudio code to Python with CoreMusic.

AudioFile Operations
^^^^^^^^^^^^^^^^^^^^

**C/Objective-C:**

.. code-block:: c

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

**CoreMusic:**

.. code-block:: python

   import coremusic as cm

   # Open audio file
   with cm.AudioFile("/path/to/audio.wav") as audio:
       # Get format
       format = audio.format

       # Read packets
       data, count = audio.read(1024)

   # Automatic cleanup via context manager

Or using functional API for closer C mapping:

.. code-block:: python

   import coremusic.capi as capi

   # Open
   file_id = capi.audio_file_open_url("/path/to/audio.wav")

   # Get format
   format_data = capi.audio_file_get_property(
       file_id,
       capi.get_audio_file_property_data_format()
   )

   # Read
   data, count = capi.audio_file_read_packets(file_id, 0, 1024)

   # Close
   capi.audio_file_close(file_id)

AudioUnit Operations
^^^^^^^^^^^^^^^^^^^^

**C/Objective-C:**

.. code-block:: c

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

**CoreMusic:**

.. code-block:: python

   import coremusic as cm

   # Find and create output unit
   unit = cm.AudioUnit.default_output()

   # Initialize and start
   unit.initialize()
   unit.start()

   # Or functional API
   import coremusic.capi as capi

   desc = {
       'componentType': capi.get_audio_unit_type_output(),
       'componentSubType': capi.get_audio_unit_subtype_default_output(),
       'componentManufacturer': capi.get_audio_unit_manufacturer_apple()
   }

   comp = capi.audio_component_find_next(0, desc)
   unit = capi.audio_component_instance_new(comp)
   capi.audio_unit_initialize(unit)
   capi.audio_output_unit_start(unit)

MIDI Operations
^^^^^^^^^^^^^^^

**C/Objective-C:**

.. code-block:: objc

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

**CoreMusic:**

.. code-block:: python

   import coremusic.capi as capi

   # Create MIDI client
   client = capi.midi_client_create("MyClient")

   # Create output port
   output_port = capi.midi_output_port_create(client, "Output")

   # Get destination
   dest = capi.midi_get_destination(0)

   # Send note
   note_on = bytes([0x90, 60, 100])
   capi.midi_send(output_port, dest, note_on)

From AudioKit (Swift)
---------------------

**AudioKit** is a powerful Swift framework. CoreMusic provides similar capabilities in Python.

Audio Playback
^^^^^^^^^^^^^^

**AudioKit (Swift):**

.. code-block:: swift

   import AudioKit

   let file = try AVAudioFile(forReading: URL(fileURLWithPath: "audio.wav"))
   let player = AudioPlayer(file: file)
   AudioKit.output = player
   try AudioKit.start()
   player.play()

**CoreMusic:**

.. code-block:: python

   import coremusic as cm

   # High-level player
   player = cm.AudioPlayer("audio.wav")
   player.play()

   # Or lower-level AudioQueue
   with cm.AudioFile("audio.wav") as audio:
       format = audio.format
       queue = cm.AudioQueue.create_output(format)

       # Allocate buffers and queue playback
       # (See cookbook for complete example)

Audio Effects
^^^^^^^^^^^^^

**AudioKit (Swift):**

.. code-block:: swift

   import AudioKit

   let player = AudioPlayer(file: file)
   let reverb = Reverb(player)
   reverb.dryWetMix = 0.5

   AudioKit.output = reverb
   try AudioKit.start()

**CoreMusic:**

.. code-block:: python

   from coremusic.audio.audiounit_host import AudioUnitPlugin

   # Load reverb AudioUnit
   with AudioUnitPlugin.from_name("AUReverb") as reverb:
       reverb['Dry/Wet Mix'] = 0.5

       # Process audio
       output = reverb.process(input_data)

Feature Comparison Matrix
--------------------------

======================== ======= ========= ======== ======= ========== =========
Feature                  pydub   soundfile wave     mido    CoreAudio  CoreMusic
======================== ======= ========= ======== ======= ========== =========
Audio File I/O           ✅      ✅        ✅       ❌      ✅         ✅
Format Conversion        ✅      ❌        ❌       ❌      ✅         ✅
Real-time Audio          ❌      ❌        ❌       ❌      ✅         ✅
AudioUnit Support        ❌      ❌        ❌       ❌      ✅         ✅
MIDI I/O                 ❌      ❌        ❌       ✅      ✅         ✅
MIDI Files               ❌      ❌        ❌       ✅      ✅         ✅
Hardware Control         ❌      ❌        ❌       ❌      ✅         ✅
Streaming                ⚠️      ✅        ⚠️       N/A     ✅         ✅
NumPy Integration        ⚠️      ✅        ❌       ❌      ❌         ✅
Cross-platform           ✅      ✅        ✅       ✅      ❌         ❌
External Dependencies    ffmpeg  libsndfile ❌      ❌      ❌         ❌
Performance              Medium  High      Low      High    Native     Native
======================== ======= ========= ======== ======= ========== =========

Legend:
- ✅ Full support
- ⚠️ Limited support
- ❌ Not supported
- N/A Not applicable

Migration Checklist
-------------------

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

Common Migration Patterns
--------------------------

Pattern 1: Simple Audio Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Before (pydub):**

.. code-block:: python

   from pydub import AudioSegment

   audio = AudioSegment.from_wav("input.wav")
   audio = audio + 6  # Increase volume
   audio.export("output.wav", format="wav")

**After (CoreMusic):**

.. code-block:: python

   import coremusic as cm
   import numpy as np

   with cm.AudioFile("input.wav") as audio:
       data, count = audio.read(audio.frame_count)
       samples = np.frombuffer(data, dtype=np.float32)
       samples *= 2.0  # Increase volume (~6dB)

       with cm.ExtendedAudioFile.create(
           "output.wav",
           cm.capi.fourchar_to_int('WAVE'),
           audio.format
       ) as output:
           output.write(count, samples.tobytes())

Pattern 2: MIDI Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Before (mido):**

.. code-block:: python

   import mido

   with mido.open_output() as port:
       for note in [60, 64, 67]:
           msg = mido.Message('note_on', note=note)
           port.send(msg)

**After (CoreMusic):**

.. code-block:: python

   import coremusic.capi as capi

   client = capi.midi_client_create("App")
   port = capi.midi_output_port_create(client, "Out")
   dest = capi.midi_get_destination(0)

   for note in [60, 64, 67]:
       msg = bytes([0x90, note, 100])  # Note on
       capi.midi_send(port, dest, msg)

See Also
--------

- :doc:`/cookbook/index` - Practical recipes
- :doc:`imports` - Import patterns
- :doc:`performance` - Performance optimization
- :doc:`/api/index` - Complete API reference

.. note::
   Need help with migration? Check the examples in ``tests/demos/`` or consult the API documentation.
