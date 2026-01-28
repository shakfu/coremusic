Effects Processing
==================

This tutorial covers audio effects processing using AudioUnits with coremusic.

Prerequisites
-------------

- coremusic installed and built
- Basic Python knowledge
- Audio files to process

Understanding AudioUnits
------------------------

AudioUnits are macOS audio plugins that process audio:

- **Effects (aufx)**: Modify audio (reverb, delay, EQ, compression)
- **Instruments (aumu)**: Generate audio from MIDI
- **Generators (augn)**: Generate audio (test tones, noise)
- **Mixers (aumx)**: Mix multiple audio streams

Discovering Available Effects
-----------------------------

List All AudioUnits
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def list_all_audio_units():
       """List all available AudioUnits."""
       units = cm.list_available_audio_units()

       print(f"Found {len(units)} AudioUnits:\n")

       # Group by type
       by_type = {}
       for unit in units:
           unit_type = unit['type']
           if unit_type not in by_type:
               by_type[unit_type] = []
           by_type[unit_type].append(unit)

       type_names = {
           'aufx': 'Effects',
           'aumu': 'Instruments',
           'augn': 'Generators',
           'aumx': 'Mixers',
           'aufc': 'Format Converters',
           'auou': 'Output Units',
       }

       for unit_type, units_list in sorted(by_type.items()):
           name = type_names.get(unit_type, unit_type)
           print(f"{name} ({unit_type}): {len(units_list)} plugins")
           for unit in sorted(units_list, key=lambda x: x['name']):
               print(f"  - {unit['name']}")
           print()

   list_all_audio_units()

List Effects Only
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def list_effects():
       """List only effect AudioUnits."""
       names = cm.get_audiounit_names(filter_type='aufx')

       print("Available Effects:")
       for name in sorted(names):
           print(f"  {name}")

       return names

   effects = list_effects()

Find Specific Effect
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def find_effect(name):
       """Find an effect by name."""
       component = cm.find_audio_unit_by_name(name)

       if component:
           desc = component._description
           print(f"Found: {name}")
           print(f"  Type: {desc.type}")
           print(f"  Subtype: {desc.subtype}")
           print(f"  Manufacturer: {desc.manufacturer}")
           return component
       else:
           print(f"Not found: {name}")
           return None

   # Find AUDelay
   delay = find_effect("AUDelay")

   # Find by partial name
   reverb = find_effect("Reverb")

Using the CLI
^^^^^^^^^^^^^

.. code-block:: bash

   # List all plugins
   coremusic plugin list

   # List effects only
   coremusic plugin list --type aufx

   # Get plugin info
   coremusic plugin info AUDelay

Creating an Effects Chain
-------------------------

Simple Effect Chain
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def create_simple_chain():
       """Create a simple effect chain."""
       chain = cm.AudioEffectsChain()

       # Add effect by name
       delay_node = chain.add_effect_by_name("AUDelay")

       # Add output
       output_node = chain.add_output()

       # Connect effect to output
       chain.connect(delay_node, output_node)

       print(f"Created chain with {chain.node_count} nodes")
       return chain

   chain = create_simple_chain()
   chain.dispose()

Multiple Effects Chain
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def create_multi_effect_chain():
       """Create chain with multiple effects."""
       chain = cm.AudioEffectsChain()

       # Add effects in series: EQ -> Compressor -> Reverb -> Output
       eq_node = chain.add_effect_by_name("AUGraphicEQ")
       comp_node = chain.add_effect_by_name("AUDynamicsProcessor")
       reverb_node = chain.add_effect_by_name("AUReverb2")
       output_node = chain.add_output()

       # Connect: EQ -> Compressor -> Reverb -> Output
       chain.connect(eq_node, comp_node)
       chain.connect(comp_node, reverb_node)
       chain.connect(reverb_node, output_node)

       print("Created effects chain:")
       print("  Input -> EQ -> Compressor -> Reverb -> Output")

       return chain

   chain = create_multi_effect_chain()
   chain.dispose()

Using Effect Descriptors
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def create_chain_from_descriptors():
       """Create chain using explicit descriptors."""
       # Effect descriptors: (type, subtype, manufacturer)
       effects = [
           ("aufx", "dely", "appl"),  # Apple Delay
           ("aufx", "rvb2", "appl"),  # Apple Reverb
       ]

       chain = cm.create_simple_effect_chain(effects)

       print(f"Created chain with {chain.node_count} nodes")
       return chain

   chain = create_chain_from_descriptors()
   chain.dispose()

Processing Audio Files
----------------------

Using the CLI
^^^^^^^^^^^^^

.. code-block:: bash

   # Apply effect to audio file
   coremusic plugin process AUDelay input.wav -o output.wav

   # Use a preset
   coremusic plugin process AUDelay input.wav -o output.wav --preset "Long Delay"

   # List available presets
   coremusic plugin preset list AUDelay

Programmatic Processing
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def process_audio_with_effect(input_path, output_path, effect_name):
       """Process audio file through an effect."""
       # Create effect chain
       chain = cm.AudioEffectsChain()
       effect_node = chain.add_effect_by_name(effect_name)
       output_node = chain.add_output()
       chain.connect(effect_node, output_node)

       # Initialize chain
       chain.open()
       chain.initialize()

       try:
           # Open input file
           with cm.ExtendedAudioFile(input_path) as input_audio:
               # Set up format
               input_format = input_audio.format

               # Process in chunks
               chunk_size = 4096
               output_data = []

               while True:
                   data, count = input_audio.read(chunk_size)
                   if count == 0:
                       break

                   # Process through chain
                   processed = chain.process(data)
                   output_data.append(processed)

               # Write output
               # (Simplified - actual implementation needs proper file writing)

           print(f"Processed {input_path} -> {output_path}")

       finally:
           chain.stop()
           chain.dispose()

   process_audio_with_effect("input.wav", "output.wav", "AUReverb2")

Configuring Effect Parameters
-----------------------------

Listing Parameters
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def list_effect_parameters(effect_name):
       """List all parameters of an effect."""
       component = cm.find_audio_unit_by_name(effect_name)
       if not component:
           print(f"Effect not found: {effect_name}")
           return

       # Create instance
       unit = component.create_instance()
       unit.initialize()

       try:
           # Get parameter list
           params = unit.get_parameter_list()

           print(f"Parameters for {effect_name}:")
           print("-" * 50)

           for param in params:
               info = unit.get_parameter_info(param)
               print(f"  {info.name}")
               print(f"    ID: {param}")
               print(f"    Range: {info.min_value} - {info.max_value}")
               print(f"    Default: {info.default_value}")
               print(f"    Unit: {info.unit_name}")
               print()

       finally:
           unit.dispose()

   list_effect_parameters("AUDelay")

Setting Parameters
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def configure_delay_effect():
       """Configure delay effect parameters."""
       component = cm.find_audio_unit_by_name("AUDelay")
       unit = component.create_instance()
       unit.initialize()

       try:
           # Common AUDelay parameters:
           # - Delay Time (seconds)
           # - Feedback (%)
           # - Wet/Dry Mix (%)

           # Set delay time to 0.25 seconds
           unit.set_parameter(0, 0.25)  # Parameter 0 = Delay Time

           # Set feedback to 50%
           unit.set_parameter(1, 50.0)  # Parameter 1 = Feedback

           # Set wet/dry mix to 30%
           unit.set_parameter(2, 30.0)  # Parameter 2 = Mix

           print("Delay configured:")
           print(f"  Delay Time: {unit.get_parameter(0)}s")
           print(f"  Feedback: {unit.get_parameter(1)}%")
           print(f"  Mix: {unit.get_parameter(2)}%")

       finally:
           unit.dispose()

   configure_delay_effect()

Using Presets
^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def use_effect_preset(effect_name, preset_name):
       """Apply a preset to an effect."""
       component = cm.find_audio_unit_by_name(effect_name)
       unit = component.create_instance()
       unit.initialize()

       try:
           # List available presets
           presets = unit.get_factory_presets()

           print(f"Available presets for {effect_name}:")
           for i, preset in enumerate(presets):
               print(f"  [{i}] {preset.name}")

           # Find and apply preset
           for preset in presets:
               if preset_name.lower() in preset.name.lower():
                   unit.set_preset(preset)
                   print(f"\nApplied preset: {preset.name}")
                   return

           print(f"\nPreset not found: {preset_name}")

       finally:
           unit.dispose()

   use_effect_preset("AUReverb2", "Large Hall")

Real-Time Effects Processing
----------------------------

.. code-block:: python

   import coremusic as cm
   import time

   class RealTimeEffectsProcessor:
       """Process audio in real-time with effects."""

       def __init__(self):
           self.chain = None
           self.running = False

       def setup(self, effect_names):
           """Set up effects chain."""
           self.chain = cm.AudioEffectsChain()

           # Add effects
           prev_node = None
           for name in effect_names:
               node = self.chain.add_effect_by_name(name)
               if node is None:
                   print(f"Warning: Effect not found: {name}")
                   continue

               if prev_node is not None:
                   self.chain.connect(prev_node, node)
               prev_node = node

           # Add output
           output_node = self.chain.add_output()
           if prev_node:
               self.chain.connect(prev_node, output_node)

           # Initialize
           self.chain.open()
           self.chain.initialize()

           print(f"Effects chain ready with {self.chain.node_count} nodes")

       def start(self):
           """Start real-time processing."""
           if self.chain:
               self.chain.start()
               self.running = True
               print("Effects processing started")

       def stop(self):
           """Stop processing."""
           if self.chain:
               self.chain.stop()
               self.running = False
               print("Effects processing stopped")

       def cleanup(self):
           """Clean up resources."""
           if self.chain:
               self.chain.dispose()
               self.chain = None

   # Use the processor
   processor = RealTimeEffectsProcessor()
   processor.setup(["AUDelay", "AUReverb2"])
   processor.start()

   # Let it run for a while
   time.sleep(5)

   processor.stop()
   processor.cleanup()

Common Effect Configurations
----------------------------

Reverb
^^^^^^

.. code-block:: python

   import coremusic as cm

   def create_reverb_effect(room_size="medium"):
       """Create configured reverb effect."""
       chain = cm.AudioEffectsChain()
       reverb_node = chain.add_effect("aufx", "rvb2", "appl")
       output_node = chain.add_output()
       chain.connect(reverb_node, output_node)

       chain.open()
       chain.initialize()

       # Get the reverb unit to configure
       # Note: Implementation depends on how chain exposes nodes
       # This is a conceptual example

       presets = {
           "small": {"decay": 0.5, "mix": 20},
           "medium": {"decay": 1.5, "mix": 30},
           "large": {"decay": 3.0, "mix": 40},
           "hall": {"decay": 5.0, "mix": 50},
       }

       if room_size in presets:
           settings = presets[room_size]
           print(f"Reverb configured: {room_size}")
           print(f"  Decay: {settings['decay']}s")
           print(f"  Mix: {settings['mix']}%")

       return chain

   reverb = create_reverb_effect("large")
   reverb.dispose()

Delay
^^^^^

.. code-block:: python

   import coremusic as cm

   def create_delay_effect(tempo_bpm=120, note_value="1/4"):
       """Create tempo-synced delay effect."""
       # Calculate delay time from tempo
       beat_duration = 60.0 / tempo_bpm

       note_values = {
           "1/1": 4.0,
           "1/2": 2.0,
           "1/4": 1.0,
           "1/8": 0.5,
           "1/16": 0.25,
           "1/8T": 1.0/3.0,  # Triplet
           "1/8D": 0.75,      # Dotted
       }

       multiplier = note_values.get(note_value, 1.0)
       delay_time = beat_duration * multiplier

       chain = cm.AudioEffectsChain()
       delay_node = chain.add_effect("aufx", "dely", "appl")
       output_node = chain.add_output()
       chain.connect(delay_node, output_node)

       print(f"Delay configured for {tempo_bpm} BPM:")
       print(f"  Note value: {note_value}")
       print(f"  Delay time: {delay_time:.3f}s")

       return chain

   delay = create_delay_effect(tempo_bpm=120, note_value="1/8")
   delay.dispose()

EQ
^^

.. code-block:: python

   import coremusic as cm

   def create_eq_preset(preset_name="flat"):
       """Create EQ with preset configuration."""
       chain = cm.AudioEffectsChain()
       eq_node = chain.add_effect_by_name("AUNBandEQ")
       output_node = chain.add_output()
       chain.connect(eq_node, output_node)

       # EQ presets (band gains in dB)
       presets = {
           "flat": [0, 0, 0, 0, 0],
           "bass_boost": [6, 3, 0, 0, 0],
           "treble_boost": [0, 0, 0, 3, 6],
           "vocal": [-2, 0, 3, 2, -1],
           "rock": [4, 2, -1, 2, 4],
       }

       if preset_name in presets:
           gains = presets[preset_name]
           print(f"EQ Preset: {preset_name}")
           print(f"  Bands: {gains}")

       return chain

   eq = create_eq_preset("vocal")
   eq.dispose()

Complete Example: Audio Processor
---------------------------------

.. code-block:: python

   import coremusic as cm
   import sys
   from pathlib import Path

   class AudioProcessor:
       """Process audio files with effects."""

       def __init__(self):
           self.chain = None

       def setup_chain(self, effects):
           """Set up effects chain."""
           self.chain = cm.AudioEffectsChain()

           nodes = []
           for effect in effects:
               if isinstance(effect, str):
                   # Effect name
                   node = self.chain.add_effect_by_name(effect)
               else:
                   # Tuple: (type, subtype, manufacturer)
                   node = self.chain.add_effect(*effect)

               if node is not None:
                   nodes.append(node)

           # Add output
           output_node = self.chain.add_output()
           nodes.append(output_node)

           # Connect in series
           for i in range(len(nodes) - 1):
               self.chain.connect(nodes[i], nodes[i + 1])

           self.chain.open()
           self.chain.initialize()

       def process_file(self, input_path, output_path):
           """Process audio file."""
           if not self.chain:
               raise RuntimeError("Chain not set up")

           print(f"Processing: {input_path}")
           print(f"Output: {output_path}")

           # This is a simplified example
           # Real implementation needs proper audio file I/O

           with cm.AudioFile(input_path) as audio:
               duration = audio.duration
               print(f"Duration: {duration:.2f}s")

           print("Processing complete!")

       def cleanup(self):
           """Clean up resources."""
           if self.chain:
               self.chain.dispose()
               self.chain = None

   def main():
       if len(sys.argv) < 3:
           print("Usage: python audio_processor.py <input.wav> <output.wav> [effects...]")
           print("Example: python audio_processor.py in.wav out.wav AUDelay AUReverb2")
           sys.exit(1)

       input_file = sys.argv[1]
       output_file = sys.argv[2]
       effects = sys.argv[3:] if len(sys.argv) > 3 else ["AUReverb2"]

       if not Path(input_file).exists():
           print(f"Error: Input file not found: {input_file}")
           sys.exit(1)

       processor = AudioProcessor()

       try:
           print(f"Setting up effects: {', '.join(effects)}")
           processor.setup_chain(effects)
           processor.process_file(input_file, output_file)
       except Exception as e:
           print(f"Error: {e}")
           sys.exit(1)
       finally:
           processor.cleanup()

   if __name__ == "__main__":
       main()

Next Steps
----------

- :doc:`../cookbook/audiounit_hosting` - Advanced AudioUnit hosting
- :doc:`audio_playback` - Play processed audio
- :doc:`../cookbook/real_time_audio` - Real-time processing techniques

See Also
--------

- :doc:`../api/index` - Complete API reference
- :doc:`../guides/cli` - CLI plugin commands
