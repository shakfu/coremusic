AudioUnit Plugin Hosting
=========================

Recipes for hosting and controlling AudioUnit plugins.

.. contents:: Topics
   :local:
   :depth: 2

Plugin Discovery
----------------

List Available Plugins
^^^^^^^^^^^^^^^^^^^^^^

Discover all AudioUnit plugins on the system:

.. code-block:: python

   import coremusic as cm

   # Create host
   host = cm.AudioUnitHost()

   # Discover all effect plugins
   effects = host.discover_plugins(type='effect')
   print(f"Found {len(effects)} effect plugins")

   for plugin_info in effects[:10]:
       print(f"  - {plugin_info['name']} ({plugin_info['manufacturer']})")

   # Discover instrument plugins
   instruments = host.discover_plugins(type='instrument')
   print(f"\nFound {len(instruments)} instrument plugins")

   # Discover by manufacturer
   apple_plugins = host.discover_plugins(manufacturer='Apple')
   print(f"\nFound {len(apple_plugins)} Apple plugins")

Load Plugin by Name
^^^^^^^^^^^^^^^^^^^

Load a specific plugin by name:

.. code-block:: python

   import coremusic as cm

   # Load plugin using context manager (automatic cleanup)
   with cm.AudioUnitPlugin.from_name("AUDelay") as plugin:
       print(f"Loaded: {plugin.name}")
       print(f"Manufacturer: {plugin.manufacturer}")
       print(f"Version: {plugin.version}")

       # Plugin is automatically disposed when exiting context

Parameter Control
-----------------

List and Control Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Discover and control plugin parameters:

.. code-block:: python

   import coremusic as cm

   with cm.AudioUnitPlugin.from_name("AUDelay") as plugin:
       # List all parameters
       print(f"Parameters ({len(plugin.parameters)}):")
       for param in plugin.parameters:
           print(f"  - {param.name}: {param.value} {param.unit}")
           print(f"    Range: [{param.min}, {param.max}], Default: {param.default}")

       # Set parameter by name
       plugin.set_parameter("Delay Time", 0.5)
       plugin.set_parameter("Feedback", 0.3)
       plugin.set_parameter("Wet/Dry Mix", 1.0)

       # Or use dictionary-style access
       plugin['Delay Time'] = 0.25
       current_delay = plugin['Delay Time']
       print(f"Current delay: {current_delay}")

Automate Parameters
^^^^^^^^^^^^^^^^^^^

Automate parameter changes over time:

.. code-block:: python

   import coremusic as cm
   import time

   with cm.AudioUnitPlugin.from_name("AUDelay") as plugin:
       # Fade delay time from 0 to 1 second
       for i in range(100):
           delay_time = i / 100.0
           plugin['Delay Time'] = delay_time
           time.sleep(0.05)  # 50ms steps

Preset Management
-----------------

Factory Presets
^^^^^^^^^^^^^^^

Browse and load factory presets:

.. code-block:: python

   import coremusic as cm

   with cm.AudioUnitPlugin.from_name("AUReverb") as plugin:
       # List factory presets
       print(f"Factory Presets ({len(plugin.factory_presets)}):")
       for preset in plugin.factory_presets:
           print(f"  - {preset.name}")

       # Load first factory preset
       if plugin.factory_presets:
           plugin.load_preset(plugin.factory_presets[0])
           print(f"Loaded preset: {plugin.factory_presets[0].name}")

User Presets
^^^^^^^^^^^^

Save and load custom user presets:

.. code-block:: python

   import coremusic as cm

   with cm.AudioUnitPlugin.from_name("AUDelay") as plugin:
       # Configure plugin
       plugin['Delay Time'] = 0.5
       plugin['Feedback'] = 0.3
       plugin['Wet/Dry Mix'] = 0.8

       # Save as user preset with description
       preset_path = plugin.save_preset(
           "My Delay Setting",
           "500ms delay with light feedback"
       )
       print(f"Saved to: {preset_path}")

       # List all user presets
       user_presets = plugin.list_user_presets()
       print(f"User presets: {user_presets}")

       # Load user preset
       plugin.load_preset("My Delay Setting")

Export and Import Presets
^^^^^^^^^^^^^^^^^^^^^^^^^^

Share presets between systems:

.. code-block:: python

   import coremusic as cm
   from pathlib import Path

   with cm.AudioUnitPlugin.from_name("AUDelay") as plugin:
       # Export preset to custom location
       export_path = Path("~/Desktop/my_delay.json").expanduser()
       plugin.export_preset("My Delay Setting", export_path)
       print(f"Exported to: {export_path}")

   # Import preset (can be on different machine)
   with cm.AudioUnitPlugin.from_name("AUDelay") as plugin:
       imported_name = plugin.import_preset(export_path)
       print(f"Imported as: {imported_name}")

       # Load the imported preset
       plugin.load_preset(imported_name)

Audio Format Support
--------------------

Custom Audio Formats
^^^^^^^^^^^^^^^^^^^^

Process audio in different formats:

.. code-block:: python

   import coremusic as cm

   with cm.AudioUnitPlugin.from_name("AUDelay") as plugin:
       # Create custom audio format (16-bit integer, 48kHz)
       fmt = cm.PluginAudioFormat(
           sample_rate=48000.0,
           channels=2,
           sample_format=cm.PluginAudioFormat.INT16,
           interleaved=True
       )

       # Set plugin to use this format
       plugin.set_audio_format(fmt)

       # Process audio (automatic conversion to/from float32 internally)
       output = plugin.process(input_data, num_frames=1024, audio_format=fmt)

Supported Formats
^^^^^^^^^^^^^^^^^

All supported audio formats:

.. code-block:: python

   import coremusic as cm

   # Float formats (32-bit and 64-bit)
   fmt_f32 = cm.PluginAudioFormat(44100.0, 2, cm.PluginAudioFormat.FLOAT32)
   fmt_f64 = cm.PluginAudioFormat(44100.0, 2, cm.PluginAudioFormat.FLOAT64)

   # Integer formats (16-bit and 32-bit)
   fmt_i16 = cm.PluginAudioFormat(44100.0, 2, cm.PluginAudioFormat.INT16)
   fmt_i32 = cm.PluginAudioFormat(44100.0, 2, cm.PluginAudioFormat.INT32)

   # Non-interleaved (planar) format
   fmt_planar = cm.PluginAudioFormat(
       44100.0, 2,
       cm.PluginAudioFormat.FLOAT32,
       interleaved=False  # Separate buffers per channel
   )

Plugin Chains
-------------

Basic Chain
^^^^^^^^^^^

Create a simple plugin chain:

.. code-block:: python

   import coremusic as cm

   # Create chain with context manager
   with cm.AudioUnitChain() as chain:
       # Add plugins
       chain.add_plugin("AUHighpass")
       chain.add_plugin("AUDelay")
       chain.add_plugin("AUReverb")

       # Configure each plugin
       chain.configure_plugin(0, {'Cutoff Frequency': 200.0})
       chain.configure_plugin(1, {'Delay Time': 0.5, 'Feedback': 0.3})
       chain.configure_plugin(2, {'Room Size': 0.8})

       # Process audio through entire chain
       output = chain.process(input_audio, num_frames=1024)

Advanced Chain with Wet/Dry Mix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Control the balance between processed and original signal:

.. code-block:: python

   import coremusic as cm

   with cm.AudioUnitChain() as chain:
       chain.add_plugin("AUDelay")
       chain.add_plugin("AUReverb")

       chain.configure_plugin(0, {'Delay Time': 0.25})
       chain.configure_plugin(1, {'Room Size': 0.7})

       # Mix settings:
       # 0.0 = 100% dry (original signal)
       # 0.5 = 50% wet, 50% dry
       # 1.0 = 100% wet (fully processed)
       output = chain.process(input_audio, num_frames=1024, wet_dry_mix=0.7)

Dynamic Chain Manipulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Modify chain during processing:

.. code-block:: python

   import coremusic as cm

   chain = cm.AudioUnitChain()

   # Add initial plugins
   chain.add_plugin("AUHighpass")
   chain.add_plugin("AUReverb")

   # Process some audio
   output1 = chain.process(audio_chunk1)

   # Insert plugin in the middle
   chain.insert_plugin(1, "AUDelay")
   chain.configure_plugin(1, {'Delay Time': 0.3})

   # Process more audio with new chain
   output2 = chain.process(audio_chunk2)

   # Remove plugin
   chain.remove_plugin(1)

   # Process final audio
   output3 = chain.process(audio_chunk3)

   # Cleanup
   chain.dispose()

MIDI Control (Instruments)
---------------------------

Basic Note Control
^^^^^^^^^^^^^^^^^^

Play notes with AudioUnit instruments:

.. code-block:: python

   import coremusic as cm
   import time

   # Load instrument plugin
   with cm.AudioUnitPlugin.from_name("DLSMusicDevice", component_type='aumu') as synth:
       # Play middle C
       synth.note_on(channel=0, note=60, velocity=100)
       time.sleep(1.0)
       synth.note_off(channel=0, note=60)

       # Play a chord (C major: C, E, G)
       notes = [60, 64, 67]
       for note in notes:
           synth.note_on(channel=0, note=note, velocity=90)

       time.sleep(1.5)

       # Stop all notes at once
       synth.all_notes_off(channel=0)

Program Changes
^^^^^^^^^^^^^^^

Change instrument sounds using General MIDI:

.. code-block:: python

   import coremusic as cm
   import time

   with cm.AudioUnitPlugin.from_name("DLSMusicDevice", component_type='aumu') as synth:
       # Acoustic Grand Piano (GM program 0)
       synth.program_change(channel=0, program=0)
       synth.note_on(channel=0, note=60, velocity=100)
       time.sleep(0.5)
       synth.note_off(channel=0, note=60)

       time.sleep(0.2)

       # Violin (GM program 40)
       synth.program_change(channel=0, program=40)
       synth.note_on(channel=0, note=60, velocity=100)
       time.sleep(0.5)
       synth.note_off(channel=0, note=60)

       time.sleep(0.2)

       # Trumpet (GM program 56)
       synth.program_change(channel=0, program=56)
       synth.note_on(channel=0, note=60, velocity=100)
       time.sleep(0.5)
       synth.note_off(channel=0, note=60)

MIDI Controllers
^^^^^^^^^^^^^^^^

Control parameters using MIDI CC messages:

.. code-block:: python

   import coremusic as cm
   import time

   with cm.AudioUnitPlugin.from_name("DLSMusicDevice", component_type='aumu') as synth:
       synth.note_on(channel=0, note=60, velocity=100)

       # Volume fade (CC 7)
       for volume in range(127, 0, -10):
           synth.control_change(channel=0, controller=7, value=volume)
           time.sleep(0.1)

       synth.note_off(channel=0, note=60)

       # Pan sweep (CC 10)
       synth.note_on(channel=0, note=60, velocity=100)
       for pan in range(0, 128, 5):
           synth.control_change(channel=0, controller=10, value=pan)
           time.sleep(0.05)

       synth.note_off(channel=0, note=60)

Pitch Bend
^^^^^^^^^^

Apply pitch bend to notes:

.. code-block:: python

   import coremusic as cm
   import time

   with cm.AudioUnitPlugin.from_name("DLSMusicDevice", component_type='aumu') as synth:
       synth.note_on(channel=0, note=60, velocity=100)

       # Center (no bend)
       synth.pitch_bend(channel=0, value=8192)
       time.sleep(0.3)

       # Bend up (one semitone)
       synth.pitch_bend(channel=0, value=12288)
       time.sleep(0.3)

       # Back to center
       synth.pitch_bend(channel=0, value=8192)
       time.sleep(0.3)

       # Bend down (one semitone)
       synth.pitch_bend(channel=0, value=4096)
       time.sleep(0.3)

       # Back to center
       synth.pitch_bend(channel=0, value=8192)
       time.sleep(0.3)

       synth.note_off(channel=0, note=60)

Multi-Channel Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use multiple MIDI channels for complex arrangements:

.. code-block:: python

   import coremusic as cm
   import time

   with cm.AudioUnitPlugin.from_name("DLSMusicDevice", component_type='aumu') as synth:
       # Setup different instruments on different channels
       synth.program_change(channel=0, program=0)   # Piano
       synth.program_change(channel=1, program=48)  # Strings
       synth.program_change(channel=2, program=56)  # Trumpet
       synth.program_change(channel=9, program=0)   # Drums (always channel 9)

       # Play multi-channel arrangement
       synth.note_on(channel=0, note=60, velocity=90)  # Piano: C
       time.sleep(0.25)

       synth.note_on(channel=1, note=64, velocity=70)  # Strings: E
       time.sleep(0.25)

       synth.note_on(channel=2, note=72, velocity=80)  # Trumpet: C (octave up)
       time.sleep(0.25)

       synth.note_on(channel=9, note=36, velocity=100)  # Drums: Kick
       time.sleep(0.5)

       # Clean stop all channels
       for ch in range(10):
           synth.all_notes_off(channel=ch)

Complete Example: Reverb Effect
--------------------------------

Full example processing audio with reverb:

.. code-block:: python

   import coremusic as cm

   # Load audio file
   with cm.AudioFile("input.wav") as audio_file:
       # Read audio data
       audio_data, frame_count = audio_file.read_packets(0, audio_file.frame_count)

       # Get audio format
       sample_rate = audio_file.format.sample_rate
       channels = audio_file.format.channels_per_frame

   # Create plugin format
   fmt = cm.PluginAudioFormat(
       sample_rate=sample_rate,
       channels=channels,
       sample_format=cm.PluginAudioFormat.FLOAT32,
       interleaved=True
   )

   # Process with reverb
   with cm.AudioUnitPlugin.from_name("AUReverb") as reverb:
       reverb.set_audio_format(fmt)

       # Configure reverb
       reverb['Room Size'] = 0.8
       reverb['Wet/Dry Mix'] = 0.5

       # Process audio
       output_data = reverb.process(audio_data, num_frames=frame_count, audio_format=fmt)

   # Save processed audio
   with cm.AudioFile.create("output.wav", audio_file.format) as output_file:
       output_file.write_packets(output_data, frame_count)

   print("Processing complete!")

Best Practices
--------------

Resource Management
^^^^^^^^^^^^^^^^^^^

Always use context managers for automatic cleanup:

.. code-block:: python

   # Good: Automatic cleanup
   with cm.AudioUnitPlugin.from_name("AUDelay") as plugin:
       output = plugin.process(input_data)

   # Avoid: Manual cleanup (error-prone)
   plugin = cm.AudioUnitPlugin.from_name("AUDelay")
   try:
       output = plugin.process(input_data)
   finally:
       plugin.dispose()

Error Handling
^^^^^^^^^^^^^^

Handle plugin errors gracefully:

.. code-block:: python

   import coremusic as cm

   try:
       with cm.AudioUnitPlugin.from_name("NonExistentPlugin") as plugin:
           pass
   except RuntimeError as e:
       print(f"Plugin not found: {e}")

   # Check if plugin exists before loading
   host = cm.AudioUnitHost()
   effects = host.discover_plugins(type='effect')
   plugin_names = [p['name'] for p in effects]

   if "AUDelay" in plugin_names:
       with cm.AudioUnitPlugin.from_name("AUDelay") as plugin:
           output = plugin.process(input_data)

Performance
^^^^^^^^^^^

Tips for optimal performance:

.. code-block:: python

   import coremusic as cm

   with cm.AudioUnitPlugin.from_name("AUReverb") as plugin:
       # 1. Set format once, not per-process call
       fmt = cm.PluginAudioFormat(44100.0, 2, cm.PluginAudioFormat.FLOAT32)
       plugin.set_audio_format(fmt)

       # 2. Process in chunks (1024-4096 frames typical)
       chunk_size = 2048

       # 3. Pre-allocate buffers when possible
       for i in range(0, total_frames, chunk_size):
           frames_to_process = min(chunk_size, total_frames - i)
           output = plugin.process(audio_data[i:i+frames_to_process],
                                  num_frames=frames_to_process,
                                  audio_format=fmt)

See Also
--------

- :doc:`/api/index` - Complete API reference
- :doc:`file_operations` - File I/O recipes
- :doc:`link_integration` - Ableton Link tempo sync
