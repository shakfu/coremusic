CoreMusic Documentation
=======================

**CoreMusic** is a comprehensive Cython wrapper for Apple's CoreAudio and CoreMIDI ecosystem, providing both functional and object-oriented Python bindings for professional audio and MIDI development on macOS.

.. image:: https://img.shields.io/badge/python-3.6+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/platform-macOS-lightgrey.svg
   :target: https://www.apple.com/macos/
   :alt: Platform

Key Features
------------

- **Dual API Design**: Both functional (C-style) and object-oriented (Pythonic) APIs
- **Complete Framework Coverage**: CoreAudio, AudioToolbox, AudioUnit, and CoreMIDI
- **High Performance**: Cython-based with near-native C performance
- **Automatic Resource Management**: Context managers and automatic cleanup
- **Professional Audio Support**: Real-time processing, multi-channel audio, hardware control
- **Comprehensive MIDI**: MIDI 1.0/2.0 support, device management, advanced routing
- **Precise Timing & Sync**: CoreAudioClock for audio/MIDI synchronization and tempo control

Quick Start
-----------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/yourusername/coremusic.git
   cd coremusic
   pip install cython
   make

Basic Audio File Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   # Object-oriented API (recommended)
   with cm.AudioFile("audio.wav") as audio:
       print(f"Duration: {audio.duration:.2f}s")
       print(f"Sample rate: {audio.format.sample_rate}Hz")
       data, count = audio.read_packets(0, 1000)

AudioUnit Processing
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   # Create and configure an AudioUnit
   with cm.AudioUnit.default_output() as unit:
       format = cm.AudioFormat(
           sample_rate=44100.0,
           format_id='lpcm',
           channels_per_frame=2,
           bits_per_channel=16
       )
       unit.set_stream_format(format)
       unit.start()
       # ... audio processing ...
       unit.stop()

MIDI Operations
^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   # Create MIDI client
   client = cm.MIDIClient("My MIDI App")
   try:
       output_port = client.create_output_port("Output")
       # Send MIDI data
       note_on = b'\\x90\\x60\\x7F'  # Note On, Middle C
       output_port.send_data(destination, note_on)
   finally:
       client.dispose()

Audio/MIDI Synchronization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   import time

   # Use AudioClock for precise timing
   with cm.AudioClock() as clock:
       clock.play_rate = 1.0  # Normal speed
       clock.start()

       # Get time in different formats
       seconds = clock.get_time_seconds()
       beats = clock.get_time_beats()
       samples = clock.get_time_samples()

       # Change speed (for tempo sync)
       clock.play_rate = 0.5  # Half speed

       clock.stop()

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   guides/index
   tutorials/index
   cookbook/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
