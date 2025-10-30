Guides
======

Comprehensive guides for using CoreMusic effectively.

.. toctree::
   :maxdepth: 2
   :caption: Available Guides

   imports
   performance
   migration

Guide Overview
--------------

Import Guide
^^^^^^^^^^^^

Complete reference for importing modules and classes from CoreMusic.

**Topics covered:**

- Hierarchical package structure
- Object-oriented vs functional API
- Audio, MIDI, and DAW subpackages
- Best practices and common patterns
- Type hints and IDE support
- Troubleshooting import issues

**Target audience:** All users, especially those new to CoreMusic

:doc:`Read the Import Guide → <imports>`

Performance Guide
^^^^^^^^^^^^^^^^^

Best practices, benchmarks, and optimization techniques for optimal performance.

**Topics covered:**

- Performance characteristics and tiers
- API selection for different use cases
- Memory management and buffer optimization
- Large file and real-time audio processing
- Parallel processing strategies
- Profiling and debugging techniques

**Target audience:** Users building performance-critical applications

:doc:`Read the Performance Guide → <performance>`

Migration Guide
^^^^^^^^^^^^^^^

Guide for migrating from other Python audio libraries to CoreMusic.

**Topics covered:**

- Migrating from pydub
- Migrating from soundfile/libsndfile
- Migrating from wave/audioread
- Migrating from mido (MIDI)
- Porting CoreAudio C/Objective-C code
- Migrating from AudioKit (Swift)
- Feature comparison matrix
- Common migration patterns

**Target audience:** Users with existing audio code in other libraries

:doc:`Read the Migration Guide → <migration>`

Quick Navigation
----------------

**New to CoreMusic?**

Start with the :doc:`imports` to understand the package structure and import patterns.

**Building performance-critical applications?**

Check the :doc:`performance` for optimization techniques and benchmarks.

**Migrating existing code?**

The :doc:`migration` provides side-by-side comparisons with other libraries.

**Looking for practical examples?**

See the :doc:`/cookbook/index` for ready-to-use recipes.

**Need API reference?**

Browse the complete :doc:`/api/index`.

Additional Resources
--------------------

Tutorials
^^^^^^^^^

Step-by-step tutorials for common tasks:

- :doc:`/tutorials/index` - Comprehensive tutorials

Cookbook
^^^^^^^^

Practical recipes for common operations:

- :doc:`/cookbook/file_operations` - File I/O recipes
- :doc:`/cookbook/audio_processing` - Audio processing recipes
- :doc:`/cookbook/real_time_audio` - Real-time audio recipes
- :doc:`/cookbook/midi_processing` - MIDI recipes
- :doc:`/cookbook/audiounit_hosting` - AudioUnit plugin hosting
- :doc:`/cookbook/link_integration` - Ableton Link integration

API Reference
^^^^^^^^^^^^^

Complete API documentation:

- :doc:`/api/index` - Full API reference

Examples
^^^^^^^^

Working example applications:

- ``tests/demos/`` directory in the source repository

Getting Help
------------

**Documentation:**

- Browse the guides and cookbook for comprehensive information
- Check the API reference for detailed function/class documentation

**Examples:**

- Review the demo scripts in ``tests/demos/``
- Study the test suite for usage patterns

**Source Code:**

- Examine the implementation in ``src/coremusic/``
- Read inline documentation and docstrings

**Issues:**

- Report bugs or request features on GitHub

See Also
--------

- :doc:`/getting_started` - Installation and setup
- :doc:`/cookbook/index` - Practical recipes
- :doc:`/tutorials/index` - Step-by-step tutorials
- :doc:`/api/index` - API reference
