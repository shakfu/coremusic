Examples Gallery
================

Complete, working examples demonstrating coremusic capabilities.

.. toctree::
   :maxdepth: 2
   :caption: Basic Examples

   audio_player
   audio_converter
   audio_inspector
   midi_monitor

.. toctree::
   :maxdepth: 2
   :caption: Audio Processing

   real_time_processor
   batch_converter
   audio_analyzer
   waveform_generator

.. toctree::
   :maxdepth: 2
   :caption: AudioUnit Examples

   audiounit_explorer
   effect_chain
   custom_processor
   parameter_controller

.. toctree::
   :maxdepth: 2
   :caption: MIDI Examples

   midi_router
   midi_transformer
   virtual_keyboard
   midi_recorder

.. toctree::
   :maxdepth: 2
   :caption: Advanced Examples

   multi_channel_processor
   low_latency_streamer
   audio_visualizer
   scipy_integration_demo

Example Categories
------------------

Basic Examples
^^^^^^^^^^^^^^

Essential examples for getting started:

- **Audio Player**: Simple audio file playback
- **Audio Converter**: Convert between audio formats
- **Audio Inspector**: Display detailed file information
- **MIDI Monitor**: Monitor MIDI input

Audio Processing
^^^^^^^^^^^^^^^^

Audio processing and manipulation:

- **Real-time Processor**: Process audio in real-time
- **Batch Converter**: Convert multiple files
- **Audio Analyzer**: Analyze audio characteristics
- **Waveform Generator**: Generate audio waveforms

AudioUnit Examples
^^^^^^^^^^^^^^^^^^

Working with AudioUnits:

- **AudioUnit Explorer**: Discover available AudioUnits
- **Effect Chain**: Chain multiple audio effects
- **Custom Processor**: Create custom AudioUnit
- **Parameter Controller**: Automate parameters

MIDI Examples
^^^^^^^^^^^^^

MIDI processing and routing:

- **MIDI Router**: Route MIDI between devices
- **MIDI Transformer**: Transform MIDI messages
- **Virtual Keyboard**: Create virtual MIDI keyboard
- **MIDI Recorder**: Record MIDI sequences

Advanced Examples
^^^^^^^^^^^^^^^^^

Advanced techniques and integration:

- **Multi-channel Processor**: Handle surround audio
- **Low Latency Streamer**: Minimal latency streaming
- **Audio Visualizer**: Real-time visualization
- **SciPy Integration**: Signal processing with SciPy

Running the Examples
--------------------

All examples are standalone Python scripts that can be run directly:

.. code-block:: bash

   # From the project root
   python examples/audio_player.py audio.wav

   # Or from the examples directory
   cd examples
   python audio_player.py ../tests/amen.wav

Prerequisites
-------------

All examples require:

- coremusic installed and built
- macOS with CoreAudio
- Python 3.6+

Some examples have additional requirements:

- **NumPy**: For audio analysis examples
- **SciPy**: For signal processing examples
- **Matplotlib**: For visualization examples

Install optional dependencies:

.. code-block:: bash

   pip install numpy scipy matplotlib

Example Template
----------------

Use this template for creating new examples:

.. code-block:: python

   #!/usr/bin/env python3
   """
   Example: [Example Name]

   Description: [What this example demonstrates]

   Usage: python example_name.py [arguments]
   """

   import coremusic as cm
   import sys

   def main():
       """Main function."""
       # Argument parsing
       if len(sys.argv) < 2:
           print("Usage: python example_name.py <audio_file>")
           sys.exit(1)

       filepath = sys.argv[1]

       # Example implementation
       try:
           # Your code here
           pass

       except cm.AudioFileError as e:
           print(f"Audio file error: {e}")
           sys.exit(1)
       except Exception as e:
           print(f"Error: {e}")
           sys.exit(1)

   if __name__ == "__main__":
       main()

Quick Reference
---------------

Common Example Patterns
^^^^^^^^^^^^^^^^^^^^^^^

**Simple audio playback:**

.. code-block:: python

   import coremusic as cm

   player = cm.AudioPlayer()
   player.load_file("audio.wav")
   player.setup_output()
   player.start()

**Format conversion:**

.. code-block:: python

   import coremusic as cm

   # Define formats
   src_fmt = cm.AudioFormat(44100.0, 'lpcm', ...)
   dst_fmt = cm.AudioFormat(48000.0, 'lpcm', ...)

   # Convert
   converter = cm.AudioConverter(src_fmt, dst_fmt)
   output = converter.convert(input_data, frame_count)

**MIDI routing:**

.. code-block:: python

   import coremusic as cm

   client = cm.MIDIClient("Router")
   input_port = client.create_input_port("Input")
   output_port = client.create_output_port("Output")

   # Route MIDI data
   output_port.send_data(destination, midi_data)

**Real-time processing:**

.. code-block:: python

   import coremusic as cm

   with cm.AudioUnit.default_output() as unit:
       format = cm.AudioFormat(44100.0, 'lpcm', ...)
       unit.set_stream_format(format)
       unit.start()
       # Process audio
       unit.stop()

Contributing Examples
---------------------

We welcome example contributions! To add an example:

1. Create a standalone, working script
2. Add comprehensive docstrings
3. Include usage instructions
4. Handle errors gracefully
5. Add to the examples directory
6. Update this documentation

Example Guidelines:

- **Clear purpose**: Each example should demonstrate one concept
- **Self-contained**: Minimize external dependencies
- **Well-commented**: Explain non-obvious code
- **Error handling**: Handle common errors
- **Usage help**: Print usage if arguments are missing

See Also
--------

- :doc:`../tutorials/index` - Step-by-step tutorials
- :doc:`../cookbook/index` - Recipe collection
- :doc:`../api/index` - API reference
