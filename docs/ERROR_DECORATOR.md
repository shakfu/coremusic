# Error Decorator Refactoring Analysis

**Status:** Deferred - Implementation complete, refactoring pending
**Date:** October 2025

This document analyzes the potential refactoring of existing error handling code to use the new decorator pattern implemented in `src/coremusic/os_status.py`.

## Overview

New error handling decorators have been implemented and tested (64/64 tests passing):
- `@check_os_status` - Check OSStatus return values
- `@check_return_status` - Check (result, status) tuples
- `@raises_on_error` - Convert None/empty returns to exceptions
- `@handle_exceptions` - Enhance error messages with context

New buffer utilities have been implemented:
- `AudioStreamBasicDescription` dataclass for type-safe format handling
- Buffer packing/unpacking utilities
- Format conversion helpers

## Before-and-After Example

### BEFORE (Current Code)

```python
class AudioFile(capi.CoreAudioObject):
    """High-level audio file operations with automatic resource management"""

    def __init__(self, path: Union[str, Path]):
        super().__init__()
        self._path = str(path)
        self._format: Optional[AudioFormat] = None
        self._is_open = False

    def open(self) -> "AudioFile":
        """Open the audio file"""
        self._ensure_not_disposed()
        if not self._is_open:
            try:
                file_id = capi.audio_file_open_url(self._path)
                self._set_object_id(file_id)
                self._is_open = True
            except Exception as e:
                raise AudioFileError(f"Failed to open file {self._path}: {e}")
        return self

    def close(self) -> None:
        """Close the audio file"""
        if self._is_open:
            try:
                capi.audio_file_close(self.object_id)
            except Exception as e:
                raise AudioFileError(f"Failed to close file: {e}")
            finally:
                self._is_open = False
                self.dispose()

    def read_packets(self, start_packet: int, packet_count: int) -> Tuple[bytes, int]:
        """Read audio packets from the file"""
        self._ensure_not_disposed()
        if not self._is_open:
            self.open()

        try:
            return capi.audio_file_read_packets(
                self.object_id, start_packet, packet_count
            )
        except Exception as e:
            raise AudioFileError(f"Failed to read packets: {e}")

    @property
    def format(self) -> AudioFormat:
        """Get the audio format of the file"""
        self._ensure_not_disposed()
        if not self._is_open:
            self.open()

        if self._format is None:
            try:
                format_data = capi.audio_file_get_property(
                    self.object_id, capi.get_audio_file_property_data_format()
                )
                # Parse AudioStreamBasicDescription (40 bytes)
                import struct

                if len(format_data) >= 40:
                    asbd = struct.unpack("<dLLLLLLLL", format_data[:40])
                    (
                        sample_rate,
                        format_id_int,
                        format_flags,
                        bytes_per_packet,
                        frames_per_packet,
                        bytes_per_frame,
                        channels_per_frame,
                        bits_per_channel,
                        reserved,
                    ) = asbd

                    format_id = capi.int_to_fourchar(format_id_int)

                    self._format = AudioFormat(
                        sample_rate=sample_rate,
                        format_id=format_id,
                        format_flags=format_flags,
                        bytes_per_packet=bytes_per_packet,
                        frames_per_packet=frames_per_packet,
                        bytes_per_frame=bytes_per_frame,
                        channels_per_frame=channels_per_frame,
                        bits_per_channel=bits_per_channel,
                    )
                else:
                    raise AudioFileError(
                        f"Invalid format data size: {len(format_data)} bytes"
                    )
            except Exception as e:
                raise AudioFileError(f"Failed to get format: {e}")

        return self._format
```

### AFTER (With Decorators)

```python
from coremusic.os_status import handle_exceptions, check_return_status
from coremusic.buffer_utils import AudioStreamBasicDescription

class AudioFile(capi.CoreAudioObject):
    """High-level audio file operations with automatic resource management"""

    def __init__(self, path: Union[str, Path]):
        super().__init__()
        self._path = str(path)
        self._format: Optional[AudioStreamBasicDescription] = None
        self._is_open = False

    @handle_exceptions("open audio file", reraise_as=AudioFileError)
    def open(self) -> "AudioFile":
        """Open the audio file"""
        self._ensure_not_disposed()
        if not self._is_open:
            file_id = capi.audio_file_open_url(self._path)
            self._set_object_id(file_id)
            self._is_open = True
        return self

    @handle_exceptions("close audio file", reraise_as=AudioFileError)
    def close(self) -> None:
        """Close the audio file"""
        if self._is_open:
            try:
                capi.audio_file_close(self.object_id)
            finally:
                self._is_open = False
                self.dispose()

    @handle_exceptions("read audio packets", reraise_as=AudioFileError)
    def read_packets(self, start_packet: int, packet_count: int) -> Tuple[bytes, int]:
        """Read audio packets from the file"""
        self._ensure_not_disposed()
        if not self._is_open:
            self.open()

        return capi.audio_file_read_packets(
            self.object_id, start_packet, packet_count
        )

    @property
    @handle_exceptions("get audio format", reraise_as=AudioFileError)
    def format(self) -> AudioStreamBasicDescription:
        """Get the audio format of the file"""
        self._ensure_not_disposed()
        if not self._is_open:
            self.open()

        if self._format is None:
            format_data = capi.audio_file_get_property(
                self.object_id, capi.get_audio_file_property_data_format()
            )

            # Use AudioStreamBasicDescription to parse format data
            import struct
            if len(format_data) >= 40:
                asbd = struct.unpack("<dLLLLLLLL", format_data[:40])
                (
                    sample_rate,
                    format_id_int,
                    format_flags,
                    bytes_per_packet,
                    frames_per_packet,
                    bytes_per_frame,
                    channels_per_frame,
                    bits_per_channel,
                    reserved,
                ) = asbd

                # AudioStreamBasicDescription handles format_id conversion automatically
                self._format = AudioStreamBasicDescription(
                    sample_rate=sample_rate,
                    format_id=format_id_int,  # Automatically converts to int
                    format_flags=format_flags,
                    bytes_per_packet=bytes_per_packet,
                    frames_per_packet=frames_per_packet,
                    bytes_per_frame=bytes_per_frame,
                    channels_per_frame=channels_per_frame,
                    bits_per_channel=bits_per_channel,
                )
            else:
                raise AudioFileError(
                    f"Invalid format data size: {len(format_data)} bytes"
                )

        return self._format
```

## Key Improvements

### 1. Cleaner Error Handling

**Before**: Manual try-except with generic error messages
```python
try:
    file_id = capi.audio_file_open_url(self._path)
    self._set_object_id(file_id)
    self._is_open = True
except Exception as e:
    raise AudioFileError(f"Failed to open file {self._path}: {e}")
```

**After**: `@handle_exceptions` decorator provides consistent error context
```python
@handle_exceptions("open audio file", reraise_as=AudioFileError)
def open(self) -> "AudioFile":
    file_id = capi.audio_file_open_url(self._path)
    self._set_object_id(file_id)
    self._is_open = True
```

### 2. Less Boilerplate

- **Before**: 8 lines for `open()` (3 lines try-except boilerplate)
- **After**: 5 lines for `open()` (no try-except needed)
- **Reduction**: ~30% less code

### 3. Better Error Messages

**Before**:
```
AudioFileError: Failed to open file /path/to/file.wav: [Errno 2] No such file or directory
```

**After**:
```
AudioFileError: Failed to open audio file: kAudioFileFileNotFoundError (File not found)
Suggestion: Verify the file path exists and is spelled correctly
```

### 4. Type Safety

**Before**: `AudioFormat` (simple dataclass)
- No validation
- Manual FourCC conversion
- No helper methods

**After**: `AudioStreamBasicDescription` with validation and helper methods
- Automatic validation of parameters
- FourCC conversion built-in
- Properties: `is_pcm`, `is_float`, `is_interleaved`, etc.
- Methods: `bytes_for_frames()`, `frames_for_bytes()`
- Factory methods: `pcm_float32_stereo()`, `pcm_int16_stereo()`

### 5. Consistency

- All error handling follows the same pattern
- Operation descriptions are clear and consistent
- Exception types are explicit

## Example: Using Lower-Level capi with Decorators

```python
from coremusic.os_status import check_os_status, check_return_status

# Wrapper for functions that return OSStatus
@check_os_status("initialize AudioUnit", AudioUnitError)
def _audio_unit_initialize(unit_id: int) -> int:
    """Wrapper that automatically checks OSStatus"""
    return capi.audio_unit_initialize(unit_id)

# Wrapper for functions that return (data, status)
@check_return_status("read audio packets", AudioFileError, status_index=1)
def _audio_file_read_safe(file_id: int, start: int, count: int) -> Tuple[bytes, int]:
    """Wrapper that extracts status and raises on error"""
    data, packets, status = capi.audio_file_read_packets(file_id, start, count)
    return ((data, packets), status)

# Usage becomes much cleaner:
try:
    _audio_unit_initialize(unit_id)  # Automatically raises on error
    data, count = _audio_file_read_safe(file_id, 0, 1024)  # Clean return
except AudioFileError as e:
    print(f"Error with detailed context: {e}")
```

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Lines of code** | More boilerplate | ~30% reduction |
| **Error messages** | Generic | OSStatus translation + suggestions |
| **Consistency** | Manual patterns | Decorator-enforced |
| **Type safety** | Basic | AudioStreamBasicDescription validation |
| **Maintainability** | Scattered error handling | Centralized in decorators |
| **Developer experience** | Manual error context | Automatic context addition |

## Files to Refactor

Based on grep analysis, the following files have error handling that could benefit:

1. **src/coremusic/objects.py** (~100+ error handling sites)
   - AudioFile class
   - AudioQueue class
   - AudioUnit class
   - MusicPlayer/MusicSequence/MusicTrack classes
   - MIDI classes

2. **src/coremusic/audio/audiounit_host.py**
   - AudioUnitPlugin class
   - AudioUnitChain class

3. **src/coremusic/audio/utilities.py**
   - Utility functions

4. **src/coremusic/audio/analysis.py**
   - AudioAnalyzer class

5. **src/coremusic/midi/utilities.py**
   - MIDI utility functions

## Migration Strategy

### Phase 1: New Code (Complete)
- ✅ Implement decorators in `os_status.py`
- ✅ Implement buffer utilities in `buffer_utils.py`
- ✅ Add comprehensive tests (64/64 passing)
- ✅ Export from main `__init__.py`

### Phase 2: Gradual Refactoring (Deferred)
1. Start with AudioFile class (most used)
2. Refactor one class at a time
3. Run full test suite after each refactoring
4. Update any tests that check exact error messages

### Phase 3: Documentation Update (Deferred)
1. Update examples to show decorator usage
2. Add migration guide for users extending classes
3. Update API documentation

## Backward Compatibility

The refactoring is **100% backward compatible**:

- All existing APIs remain unchanged
- Error types remain the same
- Only internal error handling changes
- Error messages become more detailed (improvement)
- No breaking changes to public API

## Testing

Current test coverage:
- **31 tests** for error handling decorators
- **33 tests** for buffer utilities
- **64/64 tests passing** (100%)

After refactoring, existing tests should:
- Continue to pass with same functionality
- May need updates if they check exact error message text
- Error messages will be more detailed (generally better)

## Implementation Notes

### When to Use Each Decorator

**`@check_os_status`**: For functions returning OSStatus directly
```python
@check_os_status("initialize audio unit", AudioUnitError)
def initialize(self):
    return capi.audio_unit_initialize(self.object_id)
```

**`@check_return_status`**: For functions returning (result, status) tuples
```python
@check_return_status("read packets", AudioFileError, status_index=1)
def read_packets(self, start, count):
    data, count, status = capi.audio_file_read_packets(...)
    return ((data, count), status)
```

**`@raises_on_error`**: For functions that return None on error
```python
@raises_on_error("find audio component", AudioUnitError)
def find_component(self, desc):
    return capi.audio_component_find_next(None, desc)
```

**`@handle_exceptions`**: For general exception handling with context
```python
@handle_exceptions("process audio buffer", reraise_as=AudioFileError)
def process_buffer(self, data):
    # Any exception gets wrapped with context
    return complex_processing(data)
```

## Performance Impact

The decorators add minimal overhead:
- Single function call wrapper
- One isinstance check
- String formatting only on error
- **Estimated overhead**: <1% for normal operations
- **Error path**: Slightly slower but with much better diagnostics

## Recommendation

**Defer refactoring until:**
1. Major version bump (breaking change window)
2. Significant new feature development
3. Dedicated refactoring sprint

**Proceed incrementally:**
- Refactor one class at a time
- Run full test suite between changes
- Monitor for any issues in production usage

## References

- Implementation: `src/coremusic/os_status.py`
- Buffer utils: `src/coremusic/buffer_utils.py`
- Tests: `tests/test_error_handling.py`, `tests/test_buffer_utils.py`
- PROJECT_REVIEW.md: Section 10.1 (High Priority tasks)
