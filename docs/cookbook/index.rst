Cookbook
========

Ready-to-use recipes for common audio and MIDI processing tasks.

.. toctree::
   :maxdepth: 2
   :caption: Recipes

   common_patterns
   file_operations
   audio_processing
   real_time_audio
   audiounit_hosting
   link_integration
   midi_processing

Recipe Overview
---------------

File Operations
^^^^^^^^^^^^^^^

- **File I/O**: Common file reading and writing patterns
- **Batch Processing**: Process multiple files efficiently
- **Format Detection**: Detect and validate audio formats
- **Format Conversion**: Convert between audio formats

Audio Processing
^^^^^^^^^^^^^^^^

- **Volume Control**: Normalize audio, adjust levels
- **Fades**: Apply fade in/out effects
- **Resampling**: Change sample rates with automatic conversion
- **Mixing**: Mix multiple audio tracks
- **Slicing**: Split audio into chunks
- **Concatenation**: Merge multiple audio files

Real-Time Audio
^^^^^^^^^^^^^^^

- **Recording**: Capture audio from input devices
- **Low-Latency Playback**: Minimal latency audio output
- **Level Monitoring**: Real-time audio level metering
- **Effects Processing**: Apply real-time audio effects

AudioUnit Plugin Hosting
^^^^^^^^^^^^^^^^^^^^^^^^^

- **Plugin Discovery**: Find and list available AudioUnit plugins
- **Parameter Control**: Control plugin parameters and automation
- **Preset Management**: Save, load, and share plugin presets
- **Audio Format Support**: Process audio in multiple formats (float32/64, int16/32)
- **Plugin Chains**: Create multi-effect chains with automatic routing
- **MIDI Control**: Control instrument plugins with MIDI messages

Ableton Link Integration
^^^^^^^^^^^^^^^^^^^^^^^^^

- **Tempo Synchronization**: Sync tempo across multiple applications
- **Beat-Accurate Timing**: Schedule events on specific beats
- **AudioPlayer Sync**: Synchronize audio playback with Link
- **MIDI Clock Sync**: Send MIDI clock messages synchronized to Link
- **Transport Control**: Control playback state across devices
- **Multi-Device Sync**: Connect multiple Link-enabled applications

MIDI Processing
^^^^^^^^^^^^^^^

- **Device Discovery**: Find and list MIDI sources and destinations
- **MIDI Input**: Receive and process MIDI messages
- **MIDI Output**: Send MIDI messages to devices
- **MIDI Routing**: Route MIDI between devices and channels
- **MIDI Transformation**: Transpose, scale velocity, and transform MIDI data
- **MIDI Recording**: Record and playback MIDI sequences

Integration
^^^^^^^^^^^

- **NumPy Integration**: Work with NumPy arrays for signal processing
- **SciPy Integration**: Use SciPy for advanced DSP
- **Async I/O**: Asynchronous audio file operations

Quick Reference
---------------

Common Patterns
^^^^^^^^^^^^^^^

**Read audio file:**

.. code-block:: python

   import coremusic as cm

   with cm.AudioFile("audio.wav") as audio:
       data, count = audio.read_packets(0, audio.frame_count)

**Load AudioUnit plugin:**

.. code-block:: python

   import coremusic as cm

   with cm.AudioUnitPlugin.from_name("AUDelay") as plugin:
       plugin['Delay Time'] = 0.5
       output = plugin.process(input_data)

**Create plugin chain:**

.. code-block:: python

   import coremusic as cm

   with cm.AudioUnitChain() as chain:
       chain.add_plugin("AUHighpass")
       chain.add_plugin("AUReverb")
       output = chain.process(input_data, wet_dry_mix=0.8)

**Sync with Ableton Link:**

.. code-block:: python

   import coremusic as cm

   with cm.link.LinkSession(bpm=120.0) as session:
       state = session.capture_app_session_state()
       print(f"Tempo: {state.tempo:.1f} BPM")
       print(f"Peers: {session.num_peers}")

**Send MIDI:**

.. code-block:: python

   import coremusic.capi as capi

   client = capi.midi_client_create("Output")
   port = capi.midi_output_port_create(client, "Out")
   dest = capi.midi_get_destination(0)

   # Send Note On
   capi.midi_send(port, dest, bytes([0x90, 60, 100]))

Tips and Best Practices
------------------------

Performance
^^^^^^^^^^^

- Use chunk processing for large files
- Pre-allocate buffers when possible
- Minimize memory copies
- Use appropriate buffer sizes (1024-4096 frames typical)

Resource Management
^^^^^^^^^^^^^^^^^^^

- Always use context managers (``with`` statements)
- Close resources explicitly if not using context managers
- Handle exceptions to ensure cleanup
- Dispose of MIDI clients and ports properly

Error Handling
^^^^^^^^^^^^^^

- Check file existence before opening
- Validate audio formats
- Handle CoreAudio errors gracefully
- Provide meaningful error messages

Thread Safety
^^^^^^^^^^^^^

- AudioUnits are not thread-safe by default
- Use locks when accessing shared resources
- Process audio on dedicated threads
- Avoid blocking the audio callback thread
