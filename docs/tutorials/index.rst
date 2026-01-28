Tutorials
=========

Step-by-step tutorials for common audio and MIDI tasks with coremusic.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   audio_file_basics
   audio_playback
   audio_recording
   effects_processing
   midi_basics
   midi_transform
   async_audio
   music_theory

Tutorial Overview
-----------------

Getting Started
^^^^^^^^^^^^^^^

Start here if you're new to coremusic:

1. :doc:`audio_file_basics` - Read and inspect audio files
2. :doc:`audio_playback` - Play audio files
3. :doc:`audio_recording` - Record audio from microphones
4. :doc:`midi_basics` - Send and receive MIDI messages

Audio Processing
^^^^^^^^^^^^^^^^

- :doc:`audio_file_basics` - Read, write, and analyze audio files
- :doc:`audio_playback` - Simple to advanced audio playback
- :doc:`audio_recording` - Capture audio from input devices
- :doc:`effects_processing` - Apply AudioUnit effects to audio
- :doc:`async_audio` - Non-blocking audio operations

MIDI
^^^^

- :doc:`midi_basics` - MIDI fundamentals: devices, messages, sending/receiving
- :doc:`midi_transform` - Transform MIDI with composable pipelines (transpose, quantize, humanize)

Music Theory and Generative
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :doc:`music_theory` - Notes, intervals, scales, chords, progressions
- Generative algorithms: arpeggiators, Euclidean rhythms, Markov chains

Quick Reference
---------------

Audio Files
^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   # Read audio file
   with cm.AudioFile("audio.wav") as audio:
       print(f"Duration: {audio.duration}s")
       print(f"Sample rate: {audio.format.sample_rate}")
       data, count = audio.read_packets(0, 1024)

Audio Playback
^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   player = cm.AudioPlayer()
   player.load_file("audio.wav")
   player.setup_output()
   player.start()

   while player.is_playing():
       import time
       time.sleep(0.1)

Audio Recording
^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   recorder = cm.AudioRecorder()
   recorder.setup(sample_rate=44100.0, channels=2, output_path="recording.wav")
   recorder.start()

   import time
   time.sleep(10)  # Record for 10 seconds

   recorder.stop()

Effects Processing
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   chain = cm.AudioEffectsChain()
   reverb = chain.add_effect_by_name("AUReverb2")
   output = chain.add_output()
   chain.connect(reverb, output)

   chain.open()
   chain.initialize()
   chain.start()

MIDI
^^^^

.. code-block:: python

   import coremusic as cm

   client = cm.MIDIClient("My App")
   port = client.create_output_port("Output")

   # Send Note On (middle C)
   dest = cm.midi_get_destination(0)
   port.send(dest, bytes([0x90, 60, 100]))

   client.dispose()

Command Line Examples
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Play audio
   coremusic audio play music.wav

   # Record audio
   coremusic audio record -o recording.wav --duration 10

   # Apply effect
   coremusic plugin process AUReverb2 input.wav -o output.wav

   # Monitor MIDI
   coremusic midi input monitor

   # List devices
   coremusic device list

See Also
--------

- :doc:`../getting_started` - Installation and setup
- :doc:`../cookbook/index` - Ready-to-use recipes
- :doc:`../api/index` - Complete API reference
- :doc:`../guides/cli` - Command-line interface guide
