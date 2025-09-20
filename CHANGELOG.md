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


