Command Line Interface
======================

CoreMusic includes a comprehensive command-line interface for common audio and MIDI operations.
The CLI provides quick access to audio analysis, format conversion, device management, plugin discovery,
MIDI operations, and generative music features without writing Python code.

Installation and Usage
----------------------

The CLI is automatically available after installing CoreMusic:

.. code-block:: bash

   # Show help
   coremusic --help

   # Show version
   coremusic --version

   # Get help for a specific command
   coremusic audio --help

Global Options
--------------

All commands support these global options:

``--json``
   Output results in JSON format (machine-readable)

``--version``
   Show version number and exit

``--help``
   Show help message and exit

Available Commands
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Command
     - Description
   * - ``audio``
     - Audio file operations (info, duration, metadata)
   * - ``devices``
     - Audio device management (list, default, info)
   * - ``plugins``
     - AudioUnit plugin discovery (list, find, info, params)
   * - ``analyze``
     - Audio analysis (peak, rms, silence, tempo, spectrum, key, mfcc)
   * - ``convert``
     - Convert audio files between formats
   * - ``midi``
     - MIDI device discovery (devices, inputs, outputs, send, file)
   * - ``generate``
     - Generative music algorithms (arpeggio, euclidean, melody)
   * - ``sequence``
     - MIDI sequence operations (info, play, tracks)

Audio Command
-------------

Audio file information and metadata operations.

.. code-block:: bash

   # Display audio file information
   coremusic audio info song.wav

   # Get duration in different formats
   coremusic audio duration song.wav
   coremusic audio duration song.wav --format mm:ss
   coremusic audio duration song.wav --format samples

   # Show metadata/tags
   coremusic audio metadata song.mp3

**Subcommands:**

``info <file>``
   Display comprehensive audio file information including format, sample rate,
   channels, bit depth, duration, and file size.

``duration <file> [--format FORMAT]``
   Get audio file duration. Format options: ``seconds`` (default), ``mm:ss``, ``samples``.

``metadata <file>``
   Show audio file metadata and tags (title, artist, album, etc.).

Devices Command
---------------

Audio device discovery and management.

.. code-block:: bash

   # List all audio devices
   coremusic devices list

   # Show default input/output devices
   coremusic devices default
   coremusic devices default --input
   coremusic devices default --output

   # Show detailed device information
   coremusic devices info "Built-in Output"

**Subcommands:**

``list``
   List all available audio devices with name, manufacturer, and sample rate.

``default [--input] [--output]``
   Show system default audio devices. Use flags to show only input or output.

``info <device>``
   Show detailed information for a specific device (by name or UID).

Plugins Command
---------------

Discover and inspect AudioUnit plugins installed on the system.

.. code-block:: bash

   # List all plugins
   coremusic plugins list

   # Filter by type
   coremusic plugins list --type effect
   coremusic plugins list --type instrument
   coremusic plugins list --type generator

   # Search for plugins
   coremusic plugins find "reverb"
   coremusic plugins find "synth" --type instrument

   # Show plugin details
   coremusic plugins info "AUGraphicEQ"

   # Show plugin parameters
   coremusic plugins params "AUBandpass"

**Plugin Types:**

- ``effect`` - Audio effect processors
- ``instrument`` - Virtual instruments
- ``generator`` - Audio generators
- ``music_effect`` - MIDI-enabled effects
- ``mixer`` - Mixer units
- ``panner`` - Panning units
- ``output`` - Output units
- ``format_converter`` - Format converters

**Subcommands:**

``list [--type TYPE] [--manufacturer NAME]``
   List available AudioUnit plugins with optional filtering.

``find <query> [--type TYPE]``
   Search for plugins by name.

``info <name>``
   Show detailed information about a specific plugin.

``params <name>``
   Display all parameters for a plugin with their ranges and units.

Analyze Command
---------------

Audio analysis and feature extraction.

.. code-block:: bash

   # Measure peak amplitude
   coremusic analyze peak song.wav
   coremusic analyze peak song.wav --db

   # Calculate RMS level
   coremusic analyze rms song.wav --db

   # Show both peak and RMS
   coremusic analyze levels song.wav

   # Detect silence regions
   coremusic analyze silence song.wav --threshold -40 --min-duration 0.5

   # Detect tempo/BPM
   coremusic analyze tempo song.wav

   # Analyze frequency spectrum
   coremusic analyze spectrum song.wav --peaks 10
   coremusic analyze spectrum song.wav --time 5.0

   # Detect musical key
   coremusic analyze key song.wav

   # Extract MFCC features
   coremusic analyze mfcc song.wav --coefficients 13

**Subcommands:**

``peak <file> [--db]``
   Calculate peak amplitude. Use ``--db`` for decibel output.

``rms <file> [--db]``
   Calculate RMS (average) level.

``levels <file>``
   Show both peak and RMS levels in dB.

``silence <file> [--threshold DB] [--min-duration SEC]``
   Detect silence regions. Default threshold: -40 dB, minimum duration: 0.5s.

``tempo <file>``
   Detect tempo in BPM (requires scipy).

``spectrum <file> [--time SEC] [--peaks N]``
   Analyze frequency spectrum at a specific time position.

``key <file>``
   Detect the musical key of the audio.

``mfcc <file> [--coefficients N] [--time SEC]``
   Extract Mel-frequency cepstral coefficients for audio fingerprinting.

Convert Command
---------------

Convert audio files between formats with sample rate and channel conversion.

.. code-block:: bash

   # Basic format conversion
   coremusic convert file input.wav output.aac

   # Specify output format explicitly
   coremusic convert file input.wav output.m4a --format aac

   # Change sample rate
   coremusic convert file input.wav output.wav --rate 48000

   # Convert to mono
   coremusic convert file input.wav output.wav --channels 1

   # Full conversion with quality
   coremusic convert file input.wav output.aac --rate 44100 --channels 2 --quality high

   # Batch conversion
   coremusic convert batch input_dir/ output_dir/ --format wav

**Supported Formats:**

- ``wav`` - Uncompressed PCM
- ``aif`` / ``aiff`` - Audio Interchange File Format
- ``caf`` - Core Audio Format
- ``aac`` / ``m4a`` - AAC (lossy compression)
- ``alac`` - Apple Lossless
- ``flac`` - Free Lossless Audio Codec
- ``mp3`` - MPEG Layer 3

**Quality Levels:**

- ``min`` - Minimum quality (smallest file)
- ``low`` - Low quality
- ``medium`` - Medium quality
- ``high`` - High quality (default)
- ``max`` - Maximum quality (largest file)

**Subcommands:**

``file <input> <output> [options]``
   Convert a single audio file.

``batch <input_dir> <output_dir> [--format FORMAT]``
   Batch convert all audio files in a directory.

MIDI Command
------------

MIDI device discovery and basic MIDI operations.

.. code-block:: bash

   # List all MIDI devices
   coremusic midi devices

   # List input sources only
   coremusic midi inputs

   # List output destinations only
   coremusic midi outputs

   # Send a note
   coremusic midi send --note 60 --velocity 100 --duration 0.5

   # Send to specific device
   coremusic midi send --device 1 --note 64

   # Send control change
   coremusic midi send --cc 1 64

   # Send program change
   coremusic midi send --program 5

   # Show MIDI file info
   coremusic midi file song.mid

**Subcommands:**

``devices``
   List all MIDI devices (physical and virtual).

``inputs``
   List MIDI input sources.

``outputs``
   List MIDI output destinations.

``send [options]``
   Send MIDI messages. Options:

   - ``--device``, ``-d`` - Output device index (default: 0)
   - ``--note``, ``-n`` - MIDI note number (0-127)
   - ``--velocity``, ``-v`` - Note velocity (default: 100)
   - ``--channel``, ``-c`` - MIDI channel 0-15 (default: 0)
   - ``--cc NUM VAL`` - Send control change
   - ``--program``, ``-p`` - Send program change
   - ``--duration`` - Note duration in seconds (default: 0.5)

``file <path>``
   Show MIDI file information.

Generate Command
----------------

Generative music algorithms for creating MIDI patterns.

.. code-block:: bash

   # Generate arpeggio pattern
   coremusic generate arpeggio output.mid --root C4 --chord major --pattern up

   # Generate with options
   coremusic generate arpeggio output.mid \
       --root F#3 --chord min7 --pattern up_down \
       --cycles 8 --tempo 140 --velocity 90

   # Generate Euclidean rhythm
   coremusic generate euclidean drums.mid --pulses 5 --steps 8

   # Generate melody from scale
   coremusic generate melody melody.mid --root A4 --scale minor_pentatonic

**Arpeggio Patterns:**

- ``up`` - Ascending notes
- ``down`` - Descending notes
- ``up_down`` - Ascending then descending
- ``down_up`` - Descending then ascending
- ``random`` - Random order
- ``as_played`` - Original order

**Chord Types:**

- ``major``, ``minor``, ``dim``, ``aug``
- ``sus2``, ``sus4``
- ``maj7``, ``min7``, ``dom7``, ``dim7``, ``m7b5``

**Scale Types:**

- ``major``, ``natural_minor``, ``harmonic_minor``, ``melodic_minor``
- ``dorian``, ``phrygian``, ``lydian``, ``mixolydian``, ``aeolian``, ``locrian``
- ``major_pentatonic``, ``minor_pentatonic``
- ``blues``, ``chromatic``, ``whole_tone``

**Subcommands:**

``arpeggio <output> [options]``
   Generate arpeggio pattern from a chord.

``euclidean <output> [options]``
   Generate Euclidean rhythm pattern (e.g., 5 hits in 8 steps = Cinquillo).

``melody <output> [options]``
   Generate random melody from a scale.

Sequence Command
----------------

MIDI sequence operations for working with MIDI files.

.. code-block:: bash

   # Show MIDI file information
   coremusic sequence info song.mid

   # List tracks in MIDI file
   coremusic sequence tracks song.mid

   # Play MIDI file
   coremusic sequence play song.mid

   # Play with tempo override
   coremusic sequence play song.mid --tempo 140

   # Play to specific device
   coremusic sequence play song.mid --device 1

**Subcommands:**

``info <file>``
   Display MIDI file information (format, tracks, duration, events).

``tracks <file>``
   List all tracks with event counts and note ranges.

``play <file> [options]``
   Play MIDI file through an output device. Options:

   - ``--device``, ``-d`` - MIDI output device index (default: 0)
   - ``--tempo``, ``-t`` - Override tempo in BPM

JSON Output
-----------

All commands support ``--json`` for machine-readable output:

.. code-block:: bash

   # Get audio info as JSON
   coremusic --json audio info song.wav

   # List devices as JSON
   coremusic --json devices list

   # Pipe to jq for processing
   coremusic --json plugins list | jq '.[] | select(.type == "Effect")'

Environment Variables
---------------------

``DEBUG``
   Set to ``1`` to enable debug logging:

   .. code-block:: bash

      DEBUG=1 coremusic analyze tempo song.wav

``COLOR``
   Set to ``0`` to disable colored output (default: enabled).

Examples
--------

Complete Workflow Example
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # 1. Check audio file info
   coremusic audio info recording.wav

   # 2. Analyze levels
   coremusic analyze levels recording.wav

   # 3. Detect tempo
   coremusic analyze tempo recording.wav

   # 4. Convert to AAC
   coremusic convert file recording.wav recording.aac --quality high

   # 5. Check available MIDI outputs
   coremusic midi outputs

   # 6. Generate accompaniment
   coremusic generate arpeggio accomp.mid --root C4 --chord maj7

   # 7. Play the generated MIDI
   coremusic sequence play accomp.mid

Batch Processing Script
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   #!/bin/bash
   # Analyze all WAV files in a directory

   for file in *.wav; do
       echo "=== $file ==="
       coremusic analyze levels "$file"
       coremusic analyze tempo "$file"
       echo
   done

Plugin Parameter Export
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Export all effect plugins and their parameters to JSON
   coremusic --json plugins list --type effect > effects.json

   # Get parameters for a specific plugin
   coremusic --json plugins params "AUGraphicEQ" > eq_params.json

See Also
--------

- :doc:`/getting_started` - Installation and setup
- :doc:`/api/index` - Python API reference
- :doc:`/cookbook/index` - Code recipes for common tasks
