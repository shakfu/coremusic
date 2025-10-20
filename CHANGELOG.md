# CHANGELOG

All notable project-wide changes will be documented in this file. Note that each subproject has its own CHANGELOG.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Commons Changelog](https://common-changelog.org). This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Types of Changes

- Added: for new features.
- Changed: for changes in existing functionality.
- Deprecated: for soon-to-be removed features.
- Removed: for now removed features.
- Fixed: for any bug fixes.
- Security: in case of vulnerabilities.

---

## [Unreleased]

### Added

- **Complex Audio Conversion Support** - Full callback-based AudioConverter API for advanced audio format conversions
  - **Callback Infrastructure** in Cython layer (`src/coremusic/capi.pyx`)
    - `AudioConverterCallbackData` struct for passing data between Python and C callback
    - `audio_converter_input_callback()` - C callback function with `nogil` and `noexcept` for providing input data on demand
    - `audio_converter_fill_complex_buffer()` - Python wrapper for Apple's `AudioConverterFillComplexBuffer` API
    - Proper GIL management for thread-safe operation
    - Safe memory allocation/deallocation with automatic cleanup
  - **Enhanced AudioConverter class** (`src/coremusic/objects.py`)
    - `convert_with_callback()` method supporting all conversion types:
      - Sample rate changes (e.g., 44.1kHz → 48kHz, 48kHz → 96kHz)
      - Bit depth changes (e.g., 16-bit → 24-bit)
      - Channel count changes (stereo ↔ mono)
      - Combined conversions (e.g., 44.1kHz stereo → 48kHz mono)
    - Auto-calculation of output packet count based on sample rate ratio
    - Comprehensive documentation with usage examples
  - **Updated utilities** (`src/coremusic/utilities.py`)
    - `convert_audio_file()` now supports ALL conversion types (previously only channel count)
    - Automatically chooses between simple buffer API and callback API based on conversion type
    - Added `_formats_match()` helper function for format comparison
    - Removed NotImplementedError for complex conversions
  - **Comprehensive test coverage**
    - 6 new tests in `test_objects_audio_converter.py`:
      - Sample rate conversion (44.1kHz ↔ 48kHz)
      - Real file sample rate conversion with verification
      - Combined sample rate and channel conversion
      - Auto output packet count calculation
    - 3 previously skipped tests now enabled in `test_utilities.py`:
      - `test_convert_audio_file_sample_rate`
      - `test_convert_audio_file_bit_depth`
      - `test_convert_audio_file_combined_conversions`
    - All tests passing (474 passed, 36 skipped, 0 failures)
    - Duration preservation verified (< 0.000003s error for 2.743s audio)
  - **Documentation** in `docs/COMPLEX_AUDIO_CONVERSION.md`
    - Complete implementation guide with code examples
    - Technical details on callback mechanism and memory management
    - Usage examples and best practices
    - Implementation status updated

## [0.1.4]

### Added

- **Async I/O Support** - Complete async/await support for non-blocking audio operations
  - `AsyncAudioFile` class for asynchronous file reading with chunk streaming
  - `AsyncAudioQueue` class for non-blocking audio queue operations
  - Async context manager support (`async with`) for automatic resource cleanup
  - Async chunk streaming via `read_chunks_async()` - yields audio data without blocking event loop
  - Async packet reading via `read_packets_async()` for fine-grained control
  - NumPy integration with `read_as_numpy_async()` and `read_chunks_numpy_async()`
  - Executor-based approach using `asyncio.to_thread()` for CPU-bound operations
  - Convenience functions: `open_audio_file_async()`, `create_output_queue_async()`
  - Full backward compatibility - existing synchronous API completely untouched
  - Enables concurrent file processing and integration with modern async frameworks (FastAPI, aiohttp, etc.)

- **Comprehensive async test coverage**
  - `test_async_io.py` - 22 async tests covering all async functionality
  - Tests for async file operations (open, close, context managers)
  - Tests for async packet reading and chunk streaming
  - Tests for concurrent file access and processing pipelines
  - Tests for AudioQueue lifecycle management with async operations
  - Tests for NumPy integration with async streaming
  - Real-world async processing pipeline examples
  - 100% pass rate (22/22 tests passing when NumPy available)

- **Demo script for async I/O** (`demo_async_io.py`)
  - 6 comprehensive examples demonstrating async capabilities
  - Basic async file reading with format inspection
  - Streaming large files in chunks without blocking
  - Async AudioQueue creation and playback control
  - Concurrent file processing (batch operations)
  - Real-world processing pipeline (Read → Analyze → Save)
  - NumPy integration for signal processing workflows

- **High-Level Audio Processing Utilities** - Convenient utilities for common audio tasks
  - `AudioAnalyzer` class for audio analysis operations
    - `detect_silence()` - Detect silence regions in audio files with configurable threshold and duration
    - `get_peak_amplitude()` - Extract peak amplitude from audio files
    - `calculate_rms()` - Calculate RMS (Root Mean Square) amplitude
    - `get_file_info()` - Extract comprehensive file metadata (format, duration, sample rate, etc.)
    - All methods support both file paths and AudioFile objects
    - NumPy integration for efficient audio data processing
  - `AudioFormatPresets` class with common audio format presets
    - `wav_44100_stereo()` - CD quality WAV (44.1kHz, 16-bit, stereo)
    - `wav_44100_mono()` - Mono WAV (44.1kHz, 16-bit, mono)
    - `wav_48000_stereo()` - Pro audio WAV (48kHz, 16-bit, stereo)
    - `wav_96000_stereo()` - High-res WAV (96kHz, 24-bit, stereo)
  - `convert_audio_file()` - Simple file format conversion
    - Supports stereo ↔ mono conversion at same sample rate and bit depth
    - Automatic file copy for exact format matches
    - Raises NotImplementedError for complex conversions (guides users to AudioConverter)
  - `batch_convert()` - Batch convert multiple files with glob patterns
    - Supports custom output directory and file extension
    - Optional progress callback for UI integration
    - Automatic directory creation and file overwrite control
  - `trim_audio()` - Extract time ranges from audio files
    - Supports start and end time specification
    - Preserves audio format during trimming
  - `AudioEffectsChain` class for high-level AUGraph management
    - Pythonic wrapper for audio processing graphs with automatic resource management
    - Methods: `add_effect()`, `add_output()`, `connect()`, `open()`, `initialize()`, `start()`, `stop()`
    - Support for method chaining (e.g., `chain.open().initialize().start()`)
    - Context manager support for automatic cleanup
    - Node management with FourCC-based AudioUnit identification
  - `create_simple_effect_chain()` - Convenience function for quick effect chain creation
  - Comprehensive test coverage with 35 tests (28 passing, 7 skipped)
  - Demo script (`tests/demos/demo_utilities.py`) with 10 working examples

- **AudioUnit Name-Based Discovery** - Find and load AudioUnits by name instead of FourCC codes
  - `find_audio_unit_by_name()` - Search for AudioUnits by name (e.g., 'AUDelay')
    - Returns `AudioComponent` object (can create instances directly)
    - Case-insensitive substring matching by default
    - Returns `None` if no matching AudioUnit found
    - Iterates through all available AudioComponents using CoreAudio's `AudioComponentFindNext`
    - Example: `component = cm.find_audio_unit_by_name('AUDelay')`
  - `list_available_audio_units()` - List all available AudioUnits on the system
    - Returns list of dicts with 'name', 'type', 'subtype', 'manufacturer', 'flags'
    - Optional filtering by FourCC type code (e.g., 'aufx' for audio effects)
    - Discovers 676 AudioUnits on typical macOS system
    - Example: `units = cm.list_available_audio_units(filter_type='aufx')`
  - `get_audiounit_names()` - Get simple list of AudioUnit names
    - Returns list of strings (names only, lightweight)
    - Optional filtering by FourCC type code
    - Example: `names = cm.get_audiounit_names()`
  - `AudioEffectsChain.add_effect_by_name()` - Add effects to chain by name
    - Convenience method that automatically finds and adds AudioUnits
    - Example: `delay_node = chain.add_effect_by_name('AUDelay')`
  - Low-level C API wrappers in `src/coremusic/capi.pyx`:
    - `audio_component_copy_name()` - Get human-readable AudioComponent name
    - `audio_component_get_description()` - Get AudioComponentDescription
    - Updated `audio_component_find_next()` with iteration support
  - Proper CoreFoundation memory management with CFRelease for CFStringRef
  - Comprehensive test coverage with 11 tests (100% passing)
  - Documentation in `docs/audiounit_name_lookup.md` with usage examples
  - Demo examples in `tests/demos/demo_utilities.py` (Example 10)

### Fixed

- **Music device test fixture** - Improved error handling for component instantiation
  - Added graceful skip when `AudioComponentInstanceNew` returns status -128
  - Status -128 indicates macOS security restrictions preventing instantiation
  - Tests now properly skip instead of erroring when components cannot be instantiated
  - Improved test robustness across different macOS security configurations
  - Affects `test_audiotoolbox_music_device.py` fixture for music device unit tests

## [0.1.3]

### Added

- **AudioConverter API** - Complete audio format conversion framework
  - Functional API with 13 wrapper functions for AudioConverter operations
  - `audio_converter_new()`, `audio_converter_dispose()`, `audio_converter_convert_buffer()`
  - `audio_converter_get_property()`, `audio_converter_set_property()`, `audio_converter_reset()`
  - 6 property ID getter functions for converter configuration
  - Object-oriented `AudioConverter` class with automatic resource management
  - Context manager support for safe resource cleanup
  - Support for stereo↔mono conversion, bit depth changes, and format conversions

- **ExtendedAudioFile API** - High-level audio file I/O with automatic format conversion
  - Functional API with 14 wrapper functions for ExtendedAudioFile operations
  - `extended_audio_file_open_url()`, `extended_audio_file_create_with_url()`
  - `extended_audio_file_read()`, `extended_audio_file_write()`, `extended_audio_file_dispose()`
  - `extended_audio_file_get_property()`, `extended_audio_file_set_property()`
  - 7 property ID getter functions for file format access
  - Object-oriented `ExtendedAudioFile` class with context manager support
  - Automatic format conversion on read/write via client format property
  - Simplified file I/O compared to lower-level AudioFile API

- **AUGraph API** - Audio Unit graph framework for managing and connecting multiple AudioUnits
  - Functional API with 21 wrapper functions for AUGraph operations
  - `au_graph_new()`, `au_graph_dispose()`, `au_graph_open()`, `au_graph_close()`
  - `au_graph_initialize()`, `au_graph_uninitialize()`, `au_graph_start()`, `au_graph_stop()`
  - `au_graph_add_node()`, `au_graph_remove_node()`, `au_graph_get_node_count()`
  - `au_graph_connect_node_input()`, `au_graph_disconnect_node_input()`, `au_graph_update()`
  - 3 state query functions: `au_graph_is_open()`, `au_graph_is_initialized()`, `au_graph_is_running()`
  - CPU load monitoring: `au_graph_get_cpu_load()`, `au_graph_get_max_cpu_load()`
  - 5 error code getter functions for AUGraph-specific errors
  - Object-oriented `AUGraph` class with automatic resource management
  - Context manager support for safe graph lifecycle management
  - Node management with `AudioComponentDescription` integration
  - Connection management for building audio processing graphs
  - Method chaining support for fluent API (e.g., `graph.open().initialize()`)
  - Properties for state queries: `is_open`, `is_initialized`, `is_running`, `cpu_load`, `node_count`

- **Comprehensive test coverage** for new APIs
  - `test_audiotoolbox_audio_converter.py` - 12 functional API tests
  - `test_audiotoolbox_extended_audio_file.py` - 14 functional API tests
  - `test_objects_audio_converter.py` - 29 object-oriented wrapper tests
  - `test_augraph.py` - 16 AUGraph tests (4 functional, 11 OO, 1 integration)
  - Tests cover creation, conversion, I/O operations, property access, error handling
  - Real-world testing with actual audio files

- **Exception hierarchy** expanded
  - Added `AudioConverterError` for converter-specific exceptions
  - Added `AUGraphError` for graph operation exceptions
  - Proper error propagation with detailed error messages

### Changed

- Enhanced `AudioFormat` class integration with converter APIs
- Improved error handling consistency across audio conversion operations

### Fixed

- **Critical fix for AudioDevice string properties** - Added proper CFStringRef handling
  - Previously, `audio_object_get_property_data()` returned raw CFStringRef pointers instead of actual string content
  - Added new `audio_object_get_property_string()` function that properly dereferences CFStringRef using CoreFoundation APIs
  - Device names, UIDs, and manufacturer strings now correctly use CFStringGetCString for stable, proper string extraction
  - Fixes unstable device name/UID issues where properties returned random garbage on each read
  - All AudioDevice string properties (name, uid, manufacturer, model_uid) now work correctly
- Fixed UID string handling in `AudioDevice._get_property_string()` to strip both leading and trailing null bytes (changed from `.rstrip('\x00')` to `.strip('\x00')`)
- Improved `test_audio_device_manager_find_by_uid` test resilience to handle devices with inconsistent UID encoding

---

## [0.1.2]

### Added

- Object-oriented API layer with automatic resource management
  - Added `CoreAudioObject` base class with proper disposal
  - Added `AudioFile`, `AudioQueue`, `AudioUnit` classes with context manager support
  - Added `MIDIClient`, `MIDIPort` classes for MIDI operations
  - Added `AudioFormat`, `AudioComponentDescription` helper classes
  - Added comprehensive exception hierarchy with `CoreAudioError` base class

- API documentation file (API.md) with implementation status

- Dual API architecture supporting both functional and object-oriented patterns

- Enhanced package structure with proper __init__.py imports

- Comprehensive test coverage for object-oriented APIs
  - Added tests for AudioFile, AudioUnit, AudioQueue OO classes
  - Added MIDI object-oriented API tests
  - Added comprehensive integration tests

### Changed

- Updated README with dual API examples and migration guide
- Enhanced project description to reflect comprehensive framework coverage
- Improved developer experience documentation

### Fixed

- Resource management issues with automatic cleanup via Cython __dealloc__
- Memory leaks in audio operations through proper disposal patterns

---

## [0.1.0] - Previous Release

### Added

- Added namespaces to cimports

- Added a bunch of tests

- Renamed project from `cycoreaudio` to `coremusic`

- Added CoreMIDI wrapper

- Added CoreAudio wrapper


