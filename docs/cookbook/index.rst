Cookbook
========

Ready-to-use recipes for common audio and MIDI processing tasks.

.. toctree::
   :maxdepth: 2
   :caption: File Operations

   file_operations
   batch_processing
   format_detection

.. toctree::
   :maxdepth: 2
   :caption: Audio Processing

   audio_effects
   sample_rate_conversion
   channel_manipulation
   normalization

.. toctree::
   :maxdepth: 2
   :caption: Real-time Audio

   live_audio_processing
   low_latency_playback
   audio_monitoring

.. toctree::
   :maxdepth: 2
   :caption: MIDI Processing

   midi_routing
   midi_filtering
   midi_transformation
   midi_recording

.. toctree::
   :maxdepth: 2
   :caption: Integration

   numpy_integration
   scipy_integration
   multiprocessing

Recipe Overview
---------------

File Operations
^^^^^^^^^^^^^^^

- **File Operations**: Common file I/O patterns
- **Batch Processing**: Process multiple files efficiently
- **Format Detection**: Detect and validate audio formats

Audio Processing
^^^^^^^^^^^^^^^^

- **Audio Effects**: Implement common audio effects
- **Sample Rate Conversion**: Convert between sample rates
- **Channel Manipulation**: Stereo to mono, channel splitting
- **Normalization**: Normalize audio levels

Real-time Audio
^^^^^^^^^^^^^^^

- **Live Audio Processing**: Process audio in real-time
- **Low Latency Playback**: Minimize playback latency
- **Audio Monitoring**: Monitor audio input/output

MIDI Processing
^^^^^^^^^^^^^^^

- **MIDI Routing**: Route MIDI between devices
- **MIDI Filtering**: Filter MIDI messages
- **MIDI Transformation**: Transform MIDI data
- **MIDI Recording**: Record MIDI sequences

Integration
^^^^^^^^^^^

- **NumPy Integration**: Work with NumPy arrays
- **SciPy Integration**: Use SciPy signal processing
- **Multiprocessing**: Parallel audio processing

Quick Reference
---------------

Common Patterns
^^^^^^^^^^^^^^^

**Read audio file:**

.. code-block:: python

   with cm.AudioFile("audio.wav") as audio:
       data, count = audio.read_packets(0, audio.frame_count)

**Convert sample rate:**

.. code-block:: python

   src_fmt = cm.AudioFormat(44100.0, 'lpcm', ...)
   dst_fmt = cm.AudioFormat(48000.0, 'lpcm', ...)
   converter = cm.AudioConverter(src_fmt, dst_fmt)
   output = converter.convert(input_data, frame_count)

**Process in real-time:**

.. code-block:: python

   with cm.AudioUnit.default_output() as unit:
       unit.set_stream_format(format)
       unit.start()
       # Process audio
       unit.stop()

**Route MIDI:**

.. code-block:: python

   client = cm.MIDIClient("Router")
   input_port = client.create_input_port("Input")
   output_port = client.create_output_port("Output")

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
