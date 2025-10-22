Getting Started
===============

This guide will help you get started with coremusic, from installation through your first audio and MIDI applications.

Prerequisites
-------------

Before installing coremusic, ensure you have:

- **macOS**: CoreAudio and CoreMIDI frameworks are macOS-specific
- **Python 3.6+**: Python 3.6 or higher is required
- **Xcode Command Line Tools**: Required for framework headers

Install Xcode Command Line Tools:

.. code-block:: bash

   xcode-select --install

Installation
------------

From Source
^^^^^^^^^^^

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/yourusername/coremusic.git
      cd coremusic

2. Install dependencies:

   .. code-block:: bash

      pip install cython

3. Build the extension:

   .. code-block:: bash

      make
      # or manually:
      python3 setup.py build_ext --inplace

4. Verify installation:

   .. code-block:: bash

      make test

Understanding the Dual API
---------------------------

coremusic provides two complementary APIs that can be used together or independently:

Functional API (Traditional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The functional API provides direct access to CoreAudio C functions:

**Advantages:**

- Direct mapping to CoreAudio C APIs
- Maximum performance and control
- Familiar interface for CoreAudio developers
- Fine-grained resource management

**Use when:**

- Maximum performance is critical
- Porting existing CoreAudio C code
- Need fine-grained control over resource lifetimes
- Building low-level audio processing components

**Example:**

.. code-block:: python

   import coremusic as cm

   # Open audio file (manual resource management)
   audio_file = cm.audio_file_open_url("audio.wav")
   try:
       format_data = cm.audio_file_get_property(
           audio_file,
           cm.get_audio_file_property_data_format()
       )
       data, count = cm.audio_file_read_packets(audio_file, 0, 1000)
   finally:
       cm.audio_file_close(audio_file)

Object-Oriented API (Modern)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The object-oriented API provides Pythonic wrappers with automatic resource management:

**Advantages:**

- Automatic cleanup with context managers
- Type safety with proper Python classes
- Pythonic patterns (properties, iteration, operators)
- Resource safety preventing memory leaks
- IDE autocompletion and type hints

**Use when:**

- Building new applications
- Rapid prototyping and development
- Team development where code safety is important
- Working with complex audio workflows

**Example:**

.. code-block:: python

   import coremusic as cm

   # Automatic resource management with context manager
   with cm.AudioFile("audio.wav") as audio_file:
       print(f"Duration: {audio_file.duration:.2f}s")
       print(f"Sample rate: {audio_file.format.sample_rate}Hz")
       data, count = audio_file.read_packets(0, 1000)

Your First Audio Application
-----------------------------

Audio File Information Tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's create a simple tool to display audio file information:

.. code-block:: python

   import coremusic as cm
   import sys

   def display_audio_info(filepath):
       """Display comprehensive audio file information."""
       with cm.AudioFile(filepath) as audio:
           fmt = audio.format

           print(f"File: {filepath}")
           print(f"Duration: {audio.duration:.2f} seconds")
           print(f"Sample Rate: {fmt.sample_rate} Hz")
           print(f"Channels: {fmt.channels_per_frame}")
           print(f"Bits per Channel: {fmt.bits_per_channel}")
           print(f"Format: {fmt.format_id}")
           print(f"Frame Count: {audio.frame_count}")

   if __name__ == "__main__":
       if len(sys.argv) < 2:
           print("Usage: python audio_info.py <audio_file>")
           sys.exit(1)

       display_audio_info(sys.argv[1])

Save this as ``audio_info.py`` and run:

.. code-block:: bash

   python audio_info.py path/to/audio.wav

Simple Audio Player
^^^^^^^^^^^^^^^^^^^

Create a basic audio player:

.. code-block:: python

   import coremusic as cm
   import time
   import sys

   def play_audio(filepath):
       """Play an audio file."""
       # Create audio player
       player = cm.AudioPlayer()

       # Load and setup
       player.load_file(filepath)
       player.setup_output()

       # Start playback
       print(f"Playing: {filepath}")
       player.start()

       # Wait for playback to complete
       while player.is_playing():
           progress = player.get_progress()
           print(f"Progress: {progress:.1%}", end='\\r')
           time.sleep(0.1)

       print("\\nPlayback complete!")

   if __name__ == "__main__":
       if len(sys.argv) < 2:
           print("Usage: python play_audio.py <audio_file>")
           sys.exit(1)

       play_audio(sys.argv[1])

Audio/MIDI Synchronization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use AudioClock for synchronizing audio and MIDI with precise timing:

.. code-block:: python

   import coremusic as cm
   import time

   def clock_demo():
       """Demonstrate AudioClock timing and synchronization."""
       with cm.AudioClock() as clock:
           # Set playback rate
           clock.play_rate = 1.0  # Normal speed

           # Start the clock
           clock.start()
           print("Clock started at normal speed")

           # Monitor time in different formats
           for i in range(5):
               seconds = clock.get_time_seconds()
               beats = clock.get_time_beats()
               samples = clock.get_time_samples()
               print(f"Time: {seconds:.3f}s, {beats:.2f} beats, {samples:.0f} samples")
               time.sleep(1.0)

           # Change playback rate
           clock.play_rate = 0.5  # Half speed
           print("\\nChanged to half speed")

           start_time = clock.get_time_seconds()
           time.sleep(1.0)
           end_time = clock.get_time_seconds()

           print(f"Real time elapsed: 1.0s")
           print(f"Clock time elapsed: {end_time - start_time:.3f}s")

           clock.stop()
           print("Clock stopped")

   if __name__ == "__main__":
       clock_demo()

Your First MIDI Application
----------------------------

MIDI Device Lister
^^^^^^^^^^^^^^^^^^

Create a tool to list all MIDI devices:

.. code-block:: python

   import coremusic as cm

   def list_midi_devices():
       """List all available MIDI devices, sources, and destinations."""

       # Get device counts
       device_count = cm.midi_get_number_of_devices()
       source_count = cm.midi_get_number_of_sources()
       dest_count = cm.midi_get_number_of_destinations()

       print(f"MIDI Devices: {device_count}")
       print(f"MIDI Sources: {source_count}")
       print(f"MIDI Destinations: {dest_count}")
       print()

       # List all devices
       for i in range(device_count):
           device = cm.midi_get_device(i)
           try:
               name = cm.midi_object_get_string_property(
                   device,
                   cm.get_midi_property_name()
               )
               print(f"Device {i}: {name}")
           except Exception as e:
               print(f"Device {i}: <error reading name>")

   if __name__ == "__main__":
       list_midi_devices()

Simple MIDI Monitor
^^^^^^^^^^^^^^^^^^^

Create a MIDI monitor that displays incoming messages:

.. code-block:: python

   import coremusic as cm
   import time

   def midi_callback(packet_list, src_conn_ref):
       """Callback function for MIDI input."""
       # Process MIDI packets
       print(f"Received MIDI data from connection {src_conn_ref}")

   def monitor_midi():
       """Monitor MIDI input from all sources."""
       # Create MIDI client
       client = cm.MIDIClient("MIDI Monitor")

       try:
           # Create input port
           input_port = client.create_input_port("Monitor Input")

           # Connect to all sources
           source_count = cm.midi_get_number_of_sources()
           print(f"Monitoring {source_count} MIDI sources...")
           print("Press Ctrl+C to stop")

           # Keep running
           while True:
               time.sleep(0.1)

       except KeyboardInterrupt:
           print("\\nStopping monitor...")
       finally:
           client.dispose()

   if __name__ == "__main__":
       monitor_midi()

Next Steps
----------

Now that you've created your first applications, explore:

- :doc:`tutorials/index` - Step-by-step tutorials for common tasks
- :doc:`cookbook/index` - Ready-to-use recipes for audio processing
- :doc:`examples/index` - Complete example applications
- :doc:`api/index` - Detailed API reference

Common Patterns
---------------

Context Managers
^^^^^^^^^^^^^^^^

Always use context managers for automatic resource cleanup:

.. code-block:: python

   # Good - automatic cleanup
   with cm.AudioFile("file.wav") as audio:
       data = audio.read_packets(0, 1000)

   # Also good - explicit but safe
   audio = cm.AudioFile("file.wav")
   try:
       audio.open()
       data = audio.read_packets(0, 1000)
   finally:
       audio.close()

Error Handling
^^^^^^^^^^^^^^

Handle errors appropriately:

.. code-block:: python

   import coremusic as cm

   try:
       with cm.AudioFile("file.wav") as audio:
           data = audio.read_packets(0, 1000)
   except cm.AudioFileError as e:
       print(f"Audio file error: {e}")
   except FileNotFoundError:
       print("File not found")

Resource Management
^^^^^^^^^^^^^^^^^^^

When using the functional API, always clean up resources:

.. code-block:: python

   # Functional API - manual cleanup required
   audio_file = cm.audio_file_open_url("file.wav")
   try:
       # Use the file
       data = cm.audio_file_read_packets(audio_file, 0, 1000)
   finally:
       # Always close, even if errors occur
       cm.audio_file_close(audio_file)

Troubleshooting
---------------

Build Errors
^^^^^^^^^^^^

If you encounter build errors:

1. Ensure Xcode Command Line Tools are installed:

   .. code-block:: bash

      xcode-select --install

2. Verify Cython is installed:

   .. code-block:: bash

      pip install --upgrade cython

3. Clean and rebuild:

   .. code-block:: bash

      make clean
      make

Runtime Errors
^^^^^^^^^^^^^^

**"Module not found" errors:**

- Ensure you're running Python from the project directory
- Verify the extension was built: ``ls src/coremusic/*.so``

**Audio playback issues:**

- Check audio file format is supported (WAV, AIFF, MP3, etc.)
- Verify audio file exists and is not corrupted
- Ensure macOS audio system is working

**MIDI issues:**

- Check MIDI devices are connected and powered on
- Verify MIDI devices appear in Audio MIDI Setup app
- Ensure no other application is exclusively using MIDI devices

Getting Help
------------

If you encounter issues:

1. Check the :doc:`api/index` for detailed function documentation
2. Review the :doc:`examples/index` for working code samples
3. Search existing issues on GitHub
4. Create a new issue with:
   - Your macOS version
   - Python version
   - Complete error message
   - Minimal code to reproduce the issue
