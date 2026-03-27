# Guides

Comprehensive guides for using CoreMusic effectively.

## Guide Overview

### Command Line Interface

Complete reference for the CoreMusic CLI tool.

**Topics covered:**

- All available commands and subcommands
- Audio file operations and analysis
- Device and plugin discovery
- Format conversion
- MIDI operations
- Generative music algorithms
- JSON output and scripting

**Target audience:** Users who prefer command-line tools or need scripting capabilities

[Read the CLI Guide ->](cli.md)

### Import Guide

Complete reference for importing modules and classes from CoreMusic.

**Topics covered:**

- Hierarchical package structure
- Object-oriented vs functional API
- Audio, MIDI, and DAW subpackages
- Best practices and common patterns
- Type hints and IDE support
- Troubleshooting import issues

**Target audience:** All users, especially those new to CoreMusic

[Read the Import Guide](imports.md)

### Performance Guide

Best practices, benchmarks, and optimization techniques for optimal performance.

**Topics covered:**

- Performance characteristics and tiers
- API selection for different use cases
- Memory management and buffer optimization
- Large file and real-time audio processing
- Parallel processing strategies
- Profiling and debugging techniques

**Target audience:** Users building performance-critical applications

[Read the Performance Guide](performance.md)

### Migration Guide

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

[Read the Migration Guide](migration.md)

## Quick Navigation

**New to CoreMusic?**

Start with the [Import Guide](imports.md) to understand the package structure and import patterns.

**Building performance-critical applications?**

Check the [Performance Guide](performance.md) for optimization techniques and benchmarks.

**Migrating existing code?**

The [Migration Guide](migration.md) provides side-by-side comparisons with other libraries.

**Looking for practical examples?**

See the [Cookbook](../cookbook/index.md) for ready-to-use recipes.

**Prefer command-line tools?**

The [CLI Guide](cli.md) covers all CLI commands for audio and MIDI operations.

**Need API reference?**

Browse the complete [API Reference](../api/index.md).

## Additional Resources

### Tutorials

Step-by-step tutorials for common tasks:

- [Comprehensive tutorials](../tutorials/index.md)

### Cookbook

Practical recipes for common operations:

- [File I/O recipes](../cookbook/file_operations.md)
- [Audio processing recipes](../cookbook/audio_processing.md)
- [Real-time audio recipes](../cookbook/real_time_audio.md)
- [MIDI recipes](../cookbook/midi_processing.md)
- [AudioUnit plugin hosting](../cookbook/audiounit_hosting.md)
- [Ableton Link integration](../cookbook/link_integration.md)

### API Reference

Complete API documentation:

- [Full API reference](../api/index.md)

### Examples

Working example applications:

- `tests/demos/` directory in the source repository

## Getting Help

**Documentation:**

- Browse the guides and cookbook for comprehensive information
- Check the API reference for detailed function/class documentation

**Examples:**

- Review the demo scripts in `tests/demos/`
- Study the test suite for usage patterns

**Source Code:**

- Examine the implementation in `src/coremusic/`
- Read inline documentation and docstrings

**Issues:**

- Report bugs or request features on GitHub

## See Also

- [Installation and setup](../getting_started.md)
- [Practical recipes](../cookbook/index.md)
- [Step-by-step tutorials](../tutorials/index.md)
- [API reference](../api/index.md)
