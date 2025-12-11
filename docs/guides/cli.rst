Command Line Interface
======================

CoreMusic includes a comprehensive command-line interface for common audio and MIDI operations.
The CLI provides quick access to audio playback, recording, analysis, format conversion, device management,
plugin processing, and MIDI operations without writing Python code.

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
     - Audio file operations (info, play, record, duration, metadata)
   * - ``devices``
     - Audio device management (list, info, volume, mute, set-default)
   * - ``plugin``
     - AudioUnit plugin discovery and processing (list, find, info, params, process, render)
   * - ``analyze``
     - Audio analysis (levels, tempo, key, spectrum, loudness, onsets)
   * - ``convert``
     - Audio conversion (file, batch, normalize, trim)
   * - ``midi``
     - MIDI operations (devices, input, output, file)
   * - ``sequence``
     - MIDI sequence operations (info, play, tracks)

Audio Command
-------------

Audio file information, playback, and recording operations.

.. code-block:: bash

   # Display audio file information
   coremusic audio info song.wav

   # Play audio file
   coremusic audio play song.wav
   coremusic audio play song.wav --loop

   # Record audio from input device
   coremusic audio record -o recording.wav -d 10

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

``play <file> [--loop]``
   Play an audio file with optional looping. Shows progress bar during playback.

``record -o <output> [-d <duration>] [--sample-rate RATE] [--channels N]``
   Record audio from the default input device. Options:

   - ``-o, --output`` - Output file path (required)
   - ``-d, --duration`` - Recording duration in seconds
   - ``--sample-rate`` - Sample rate (default: 44100)
   - ``--channels`` - Number of channels (default: 2)

``duration <file> [--format FORMAT]``
   Get audio file duration. Format options: ``seconds`` (default), ``mm:ss``, ``samples``.

``metadata <file>``
   Show audio file metadata and tags (title, artist, album, etc.).

Devices Command
---------------

Audio device discovery, management, and control.

.. code-block:: bash

   # List all audio devices
   coremusic devices list

   # Show default input/output devices
   coremusic devices default
   coremusic devices default --input
   coremusic devices default --output

   # Show detailed device information
   coremusic devices info "Built-in Output"

   # Get/set device volume (0.0-1.0)
   coremusic devices volume "MacBook Pro Speakers"
   coremusic devices volume "MacBook Pro Speakers" 0.5

   # Get/set device mute state
   coremusic devices mute "MacBook Pro Speakers"
   coremusic devices mute "MacBook Pro Speakers" on

   # Set default device
   coremusic devices set-default "External Headphones" --output
   coremusic devices set-default "USB Microphone" --input

**Subcommands:**

``list``
   List all available audio devices with name, manufacturer, and sample rate.

``default [--input] [--output]``
   Show system default audio devices. Use flags to show only input or output.

``info <device>``
   Show detailed information for a specific device (by name or UID).

``volume <device> [level]``
   Get or set device volume (0.0-1.0). Options:

   - ``--scope`` - Volume scope: ``output`` (default) or ``input``
   - ``--channel`` - Channel number (default: 0 for master)

``mute <device> [on|off]``
   Get or set device mute state.

``set-default <device> [--input|--output]``
   Set the system default input or output device.

Plugin Command
--------------

Discover, inspect, and use AudioUnit plugins installed on the system.

.. code-block:: bash

   # List all plugins
   coremusic plugin list

   # List only plugin names (for scripting)
   coremusic plugin list --name-only

   # Filter by type
   coremusic plugin list --type effect
   coremusic plugin list --type instrument

   # Search for plugins
   coremusic plugin find "reverb"
   coremusic plugin find "synth" --type instrument

   # Show plugin details
   coremusic plugin info "AUGraphicEQ"

   # Show plugin parameters
   coremusic plugin params "AUBandpass"

   # List factory presets
   coremusic plugin preset list "AUReverb2"

   # Process audio through effect plugin
   coremusic plugin process "AUDelay" input.wav -o output.wav
   coremusic plugin process "AUReverb2" input.wav -o output.wav --preset "Large Hall"

   # Render MIDI through instrument plugin
   coremusic plugin render "DLSMusicDevice" song.mid -o rendered.wav
   coremusic plugin render "DLSMusicDevice" song.mid -o rendered.wav --preset 0

**Plugin Types:**

- ``effect`` - Audio effect processors (reverb, delay, EQ, etc.)
- ``instrument`` - Virtual instruments (synths, samplers)
- ``generator`` - Audio generators
- ``music_effect`` - MIDI-enabled effects
- ``mixer`` - Mixer units
- ``panner`` - Panning units
- ``output`` - Output units
- ``format_converter`` - Format converters

**Subcommands:**

``list [--type TYPE] [--manufacturer NAME] [--name-only]``
   List available AudioUnit plugins with optional filtering.

   - ``--type`` - Filter by plugin type
   - ``--manufacturer`` - Filter by manufacturer
   - ``--name-only`` - Print only unique plugin names (one per line)

``find <query> [--type TYPE]``
   Search for plugins by name.

``info <name>``
   Show detailed information about a specific plugin.

``params <name>``
   Display all parameters for a plugin with their ranges and units.

``preset list <name>``
   List factory presets for a plugin.

``process <name> <input> -o <output> [--preset NAME|NUMBER]``
   Apply an effect plugin to an audio file.

``render <name> <midi> -o <output> [--preset NAME|NUMBER] [--sample-rate RATE] [--duration SEC]``
   Render a MIDI file through an instrument plugin to audio.

Analyze Command
---------------

Audio analysis and feature extraction.

.. code-block:: bash

   # Show both peak and RMS levels
   coremusic analyze levels song.wav

   # Measure peak amplitude
   coremusic analyze peak song.wav
   coremusic analyze peak song.wav --db

   # Calculate RMS level
   coremusic analyze rms song.wav --db

   # Detect silence regions
   coremusic analyze silence song.wav --threshold -40 --min-duration 0.5

   # Detect tempo/BPM
   coremusic analyze tempo song.wav

   # Analyze frequency spectrum
   coremusic analyze spectrum song.wav --peaks 10
   coremusic analyze spectrum song.wav --time 5.0

   # Detect musical key
   coremusic analyze key song.wav

   # LUFS loudness measurement
   coremusic analyze loudness song.wav

   # Onset detection
   coremusic analyze onsets song.wav --threshold 0.5 --min-gap 0.1

   # Extract MFCC features
   coremusic analyze mfcc song.wav --coefficients 13

**Subcommands:**

``levels <file>``
   Show both peak and RMS levels in dB.

``peak <file> [--db]``
   Calculate peak amplitude. Use ``--db`` for decibel output.

``rms <file> [--db]``
   Calculate RMS (average) level.

``silence <file> [--threshold DB] [--min-duration SEC]``
   Detect silence regions. Default threshold: -40 dB, minimum duration: 0.5s.

``tempo <file>``
   Detect tempo in BPM (requires scipy).

``spectrum <file> [--time SEC] [--peaks N]``
   Analyze frequency spectrum at a specific time position.

``key <file>``
   Detect the musical key of the audio.

``loudness <file>``
   LUFS loudness measurement with integrated LUFS, loudness range, peak, and RMS.

``onsets <file> [--threshold FLOAT] [--min-gap SEC]``
   Onset detection using spectral flux.

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

   # Normalize audio
   coremusic convert normalize input.wav output.wav --target -1.0
   coremusic convert normalize input.wav output.wav --target -14.0 --mode rms

   # Trim audio
   coremusic convert trim input.wav output.wav --start 10 --end 30
   coremusic convert trim input.wav output.wav --start 5 --duration 10

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

``normalize <input> <output> [--target DB] [--mode peak|rms]``
   Normalize audio to target peak or RMS level.

``trim <input> <output> [--start SEC] [--end SEC] [--duration SEC]``
   Extract a portion of an audio file.

MIDI Command
------------

MIDI device discovery, monitoring, recording, and basic MIDI operations.

.. code-block:: bash

   # List all MIDI devices
   coremusic midi devices

   # Show detailed device info
   coremusic midi device info "USB MIDI Controller"

   # Monitor MIDI input
   coremusic midi input monitor
   coremusic midi input monitor 0

   # Record MIDI input to file
   coremusic midi input record -o recorded.mid -d 30
   coremusic midi input record -o recorded.mid --tempo 120

   # Send test note
   coremusic midi output test
   coremusic midi output test --device 1

   # Send panic (all notes off)
   coremusic midi output panic

   # Show MIDI file info
   coremusic midi file info song.mid

   # Hex dump of MIDI file
   coremusic midi file dump song.mid

   # Play MIDI file
   coremusic midi file play song.mid
   coremusic midi file play song.mid --device 1

   # Quantize MIDI file
   coremusic midi file quantize input.mid -o output.mid --grid 1/16
   coremusic midi file quantize input.mid -o output.mid --grid 1/8 --strength 0.8

**Subcommands:**

``devices``
   List all MIDI devices (physical and virtual).

``device info <name>``
   Show detailed MIDI device info including entities, sources, and destinations.

``input monitor [index]``
   Monitor MIDI input in real-time. Press Ctrl+C to stop.

``input record -o <file> [-d <duration>] [--tempo BPM]``
   Record MIDI input to a file.

``output test [--device INDEX]``
   Send a test note (middle C) to verify MIDI connectivity.

``output panic [--device INDEX]``
   Send all-notes-off (CC 123) and all-sound-off (CC 120) on all 16 channels.

``file info <path>``
   Show MIDI file information.

``file dump <path>``
   Hex dump of raw MIDI events with time, track, channel, type, and data.

``file play <path> [--device INDEX] [--tempo BPM]``
   Play MIDI file through an output device.

``file quantize <input> -o <output> [--grid GRID] [--strength FLOAT]``
   Quantize MIDI note timing to grid. Grid options: ``1/4``, ``1/8``, ``1/16``, ``1/32``.

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

   # List plugins as JSON
   coremusic --json plugin list --type effect

   # Pipe to jq for processing
   coremusic --json plugin list | jq '.[] | select(.type == "Effect")'

Environment Variables
---------------------

``DEBUG``
   Set to ``1`` to enable debug logging:

   .. code-block:: bash

      DEBUG=1 coremusic analyze tempo song.wav

Examples
--------

Complete Workflow Example
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # 1. Check audio file info
   coremusic audio info recording.wav

   # 2. Analyze levels and tempo
   coremusic analyze levels recording.wav
   coremusic analyze tempo recording.wav

   # 3. Normalize and trim
   coremusic convert normalize recording.wav normalized.wav --target -1.0
   coremusic convert trim normalized.wav trimmed.wav --start 2 --end 30

   # 4. Process through effect plugin
   coremusic plugin process "AUReverb2" trimmed.wav -o final.wav --preset "Medium Hall"

   # 5. Check available MIDI outputs
   coremusic midi devices

   # 6. Play MIDI accompaniment
   coremusic midi file play accompaniment.mid

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

Plugin Discovery Script
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Find all reverb plugins
   coremusic plugin list --name-only | grep -i reverb

   # Export all effect plugins to JSON
   coremusic --json plugin list --type effect > effects.json

   # Get parameters for a specific plugin
   coremusic --json plugin params "AUGraphicEQ" > eq_params.json

See Also
--------

- :doc:`/getting_started` - Installation and setup
- :doc:`/api/index` - Python API reference
- :doc:`/cookbook/index` - Code recipes for common tasks
