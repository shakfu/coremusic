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

- **Comprehensive test coverage** for new APIs
  - `test_audiotoolbox_audio_converter.py` - 12 functional API tests
  - `test_audiotoolbox_extended_audio_file.py` - 14 functional API tests
  - `test_objects_audio_converter.py` - 29 object-oriented wrapper tests
  - Tests cover creation, conversion, I/O operations, property access, error handling
  - Real-world testing with actual audio files

- **Exception hierarchy** expanded
  - Added `AudioConverterError` for converter-specific exceptions
  - Proper error propagation with detailed error messages

### Changed

- Enhanced `AudioFormat` class integration with converter APIs
- Improved error handling consistency across audio conversion operations

### Fixed

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


