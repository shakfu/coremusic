MIDI Transformation Pipeline
============================

This tutorial covers the MIDI transformation pipeline for loading, transforming,
and saving MIDI files using composable transformers.

Overview
--------

The ``coremusic.midi.transform`` module provides a pipeline architecture for
processing MIDI sequences. Transformers can be chained together to create
complex processing workflows.

**Key Features:**

- Load and save Standard MIDI Files
- Composable transformer pipeline
- 15+ built-in transformers for pitch, time, velocity, and filtering
- Reproducible results with seed parameters
- Fluent API for easy chaining

Quick Start
-----------

.. code-block:: python

   from coremusic.midi.utilities import MIDISequence
   from coremusic.midi.transform import Pipeline, Transpose, Quantize, Humanize

   # Load MIDI file
   seq = MIDISequence.load("input.mid")

   # Create transformation pipeline
   pipeline = Pipeline([
       Transpose(semitones=5),              # Up a perfect fourth
       Quantize(grid=0.125, strength=0.8),  # Quantize to 16th notes
       Humanize(timing=0.01, velocity=5),   # Add human feel
   ])

   # Apply and save
   transformed = pipeline.apply(seq)
   transformed.save("output.mid")

Pipeline Basics
---------------

Creating a Pipeline
^^^^^^^^^^^^^^^^^^^

A pipeline chains multiple transformers together:

.. code-block:: python

   from coremusic.midi.transform import Pipeline, Transpose, VelocityScale

   # Create with list of transformers
   pipeline = Pipeline([
       Transpose(5),
       VelocityScale(factor=0.8),
   ])

   # Or build incrementally
   pipeline = Pipeline()
   pipeline.add(Transpose(5))
   pipeline.add(VelocityScale(factor=0.8))

   # Apply to sequence
   result = pipeline.apply(sequence)

   # Pipelines are callable
   result = pipeline(sequence)

Using Individual Transformers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each transformer can be used standalone:

.. code-block:: python

   from coremusic.midi.transform import Transpose, Reverse

   # Direct transform call
   transposed = Transpose(12).transform(sequence)

   # Transformers are callable
   reversed_seq = Reverse()(sequence)

Pitch Transformers
------------------

Transpose
^^^^^^^^^

Shift all notes by a fixed number of semitones:

.. code-block:: python

   from coremusic.midi.transform import Transpose

   # Transpose up an octave
   up_octave = Transpose(12).transform(sequence)

   # Transpose down a fifth
   down_fifth = Transpose(-7).transform(sequence)

   # Notes are clamped to valid MIDI range (0-127)

Invert
^^^^^^

Mirror melody around a pivot note:

.. code-block:: python

   from coremusic.midi.transform import Invert

   # Invert around middle C (MIDI 60)
   inverted = Invert(pivot=60).transform(sequence)

   # Notes above pivot go below, and vice versa

Harmonize
^^^^^^^^^

Add parallel intervals to create harmonies:

.. code-block:: python

   from coremusic.midi.transform import Harmonize

   # Add a major third above each note
   thirds = Harmonize([4]).transform(sequence)

   # Add third and fifth (triads)
   triads = Harmonize([4, 7], velocity_scale=0.7).transform(sequence)

   # Add power chord (fifth and octave)
   power = Harmonize([7, 12]).transform(sequence)

Time Transformers
-----------------

Quantize
^^^^^^^^

Snap timing to a grid with optional swing:

.. code-block:: python

   from coremusic.midi.transform import Quantize

   # Full quantize to 16th notes (0.125s at 120 BPM)
   quantized = Quantize(grid=0.125, strength=1.0).transform(sequence)

   # Partial quantize (preserves some groove)
   soft_quant = Quantize(grid=0.125, strength=0.5).transform(sequence)

   # Add swing feel
   swing = Quantize(grid=0.125, swing=0.3).transform(sequence)

TimeStretch
^^^^^^^^^^^

Change tempo by stretching or compressing time:

.. code-block:: python

   from coremusic.midi.transform import TimeStretch

   # Double the tempo (half the time)
   faster = TimeStretch(0.5).transform(sequence)

   # Half the tempo (double the time)
   slower = TimeStretch(2.0).transform(sequence)

TimeShift
^^^^^^^^^

Move all events forward or backward in time:

.. code-block:: python

   from coremusic.midi.transform import TimeShift

   # Delay by 1 second
   delayed = TimeShift(1.0).transform(sequence)

   # Shift earlier (with clamping at 0)
   earlier = TimeShift(-0.5).transform(sequence)

Reverse
^^^^^^^

Reverse the sequence (retrograde):

.. code-block:: python

   from coremusic.midi.transform import Reverse

   # Reverse note order, preserving durations
   reversed_seq = Reverse().transform(sequence)

Velocity Transformers
---------------------

VelocityScale
^^^^^^^^^^^^^

Scale velocities by factor or to a range:

.. code-block:: python

   from coremusic.midi.transform import VelocityScale

   # Scale by factor
   quieter = VelocityScale(factor=0.5).transform(sequence)
   louder = VelocityScale(factor=1.5).transform(sequence)

   # Compress to range
   compressed = VelocityScale(min_vel=40, max_vel=100).transform(sequence)

VelocityCurve
^^^^^^^^^^^^^

Apply a velocity curve for dynamic shaping:

.. code-block:: python

   from coremusic.midi.transform import VelocityCurve

   # Built-in curves
   soft = VelocityCurve(curve='soft').transform(sequence)   # Softer dynamics
   hard = VelocityCurve(curve='hard').transform(sequence)   # Harder dynamics
   log = VelocityCurve(curve='log').transform(sequence)     # Logarithmic
   exp = VelocityCurve(curve='exp').transform(sequence)     # Exponential

   # Custom curve function (input/output 0.0-1.0)
   custom = VelocityCurve(curve=lambda x: x ** 0.7).transform(sequence)

Humanize
^^^^^^^^

Add human-like timing and velocity variation:

.. code-block:: python

   from coremusic.midi.transform import Humanize

   # Add subtle variation
   humanized = Humanize(
       timing=0.01,    # +/- 10ms timing variation
       velocity=5,     # +/- 5 velocity variation
   ).transform(sequence)

   # Reproducible with seed
   reproducible = Humanize(timing=0.02, velocity=10, seed=42).transform(sequence)

Filter Transformers
-------------------

NoteFilter
^^^^^^^^^^

Filter notes by pitch, velocity, or channel:

.. code-block:: python

   from coremusic.midi.transform import NoteFilter

   # Keep only bass notes (MIDI 24-48)
   bass = NoteFilter(min_note=24, max_note=48).transform(sequence)

   # Keep only loud notes
   loud = NoteFilter(min_velocity=80).transform(sequence)

   # Keep specific channels
   channel_0 = NoteFilter(channels={0}).transform(sequence)

   # Remove matching notes (invert filter)
   no_bass = NoteFilter(min_note=24, max_note=48, invert=True).transform(sequence)

EventTypeFilter
^^^^^^^^^^^^^^^

Filter by MIDI event type:

.. code-block:: python

   from coremusic.midi.transform import EventTypeFilter
   from coremusic.midi.utilities import MIDIStatus

   # Keep only note events
   notes_only = EventTypeFilter(
       keep=[MIDIStatus.NOTE_ON, MIDIStatus.NOTE_OFF]
   ).transform(sequence)

   # Remove control changes
   no_cc = EventTypeFilter(
       remove=[MIDIStatus.CONTROL_CHANGE]
   ).transform(sequence)

Track Transformers
------------------

ChannelRemap
^^^^^^^^^^^^

Remap MIDI channels:

.. code-block:: python

   from coremusic.midi.transform import ChannelRemap

   # Move melody from channel 0 to channel 1
   remapped = ChannelRemap({0: 1}).transform(sequence)

   # Move to drums channel
   drums = ChannelRemap({0: 9}).transform(sequence)

TrackMerge
^^^^^^^^^^

Merge all tracks into one:

.. code-block:: python

   from coremusic.midi.transform import TrackMerge

   merged = TrackMerge(name="Combined").transform(sequence)

Arpeggiate
^^^^^^^^^^

Convert chords to arpeggios:

.. code-block:: python

   from coremusic.midi.transform import Arpeggiate

   # Arpeggiate upward
   arp_up = Arpeggiate(
       pattern='up',
       note_duration=0.1
   ).transform(sequence)

   # Available patterns: 'up', 'down', 'up_down', 'down_up', 'random'
   arp_down = Arpeggiate(pattern='down', note_duration=0.1).transform(sequence)
   arp_random = Arpeggiate(pattern='random', note_duration=0.1, seed=42).transform(sequence)

Convenience Functions
---------------------

For common operations, convenience functions are available:

.. code-block:: python

   from coremusic.midi.transform import (
       transpose, quantize, humanize, reverse, scale_velocity
   )

   # Quick transformations
   result = transpose(sequence, 5)
   result = quantize(sequence, 0.125)
   result = humanize(sequence, timing=0.01, velocity=5)
   result = reverse(sequence)
   result = scale_velocity(sequence, factor=0.8)

   # Chain them
   result = humanize(quantize(transpose(sequence, 12), 0.25), timing=0.01)

Complete Example
----------------

Here's a complete workflow processing a MIDI file:

.. code-block:: python

   from coremusic.midi.utilities import MIDISequence
   from coremusic.midi.transform import (
       Pipeline, Transpose, Quantize, VelocityScale, VelocityCurve,
       Humanize, NoteFilter, Harmonize
   )

   # Load source MIDI
   original = MIDISequence.load("piano_solo.mid")
   print(f"Loaded: {len(original.tracks)} tracks, {original.duration:.2f}s")

   # Create processing pipeline
   pipeline = Pipeline([
       # Fix timing
       Quantize(grid=0.125, strength=0.7),

       # Transpose to different key
       Transpose(5),  # Up a fourth

       # Shape dynamics
       VelocityCurve(curve='soft'),
       VelocityScale(min_vel=50, max_vel=110),

       # Add expression
       Humanize(timing=0.015, velocity=8, seed=42),
   ])

   # Apply transformations
   processed = pipeline.apply(original)

   # Save result
   processed.save("piano_solo_processed.mid")
   print(f"Saved processed file")

   # Create harmony version
   harmony_pipeline = Pipeline([
       Harmonize([4, 7]),  # Add thirds and fifths
       VelocityScale(factor=0.7),  # Reduce volume
   ])
   harmony = harmony_pipeline.apply(original)
   harmony.save("piano_solo_harmony.mid")

See Also
--------

- :doc:`music_theory` - Music theory fundamentals
- ``coremusic.midi.utilities`` - MIDI file I/O
- ``coremusic.midi.link`` - Ableton Link integration
