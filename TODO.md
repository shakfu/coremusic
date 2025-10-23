# TODO

## Summary

**CoreMusic Status:** Production-ready professional audio framework for Python

**Completed Major Features:**
- [x] AudioUnit Host (190 plugins: 111 effects, 62 instruments)
  - [x] Audio Format Support (float32, float64, int16, int32, interleaved/non-interleaved)
  - [x] User Preset Management (save/load/export/import)
  - [x] AudioUnitChain (automatic plugin routing and format conversion)
- [x] Full MIDI support for instruments (note on/off, CC, program change, pitch bend)
- [x] Ableton Link integration (tempo sync, network music)
- [x] CoreMIDI (complete MIDI I/O)
- [x] CoreAudio (file I/O, queues, converters, devices)
- [x] **736 tests passing** (100% success rate)

**Future Development:**
- Plugin UI Integration (Cocoa view integration)
- Link Integration for tempo-synced plugins
- Advanced MIDI features (file playback, live routing)

---

## Active Tasks

### AudioUnit Host - Completed Enhancements ✅

**Current Status:** Core implementation complete with **736 tests passing** (100% success rate). The following enhancements have been successfully implemented:

#### 1. Audio Format Support ✅ COMPLETED
**Status:** Fully implemented and tested

**Implementation:**
- ✅ `AudioFormat` class supporting float32, float64, int16, int32
- ✅ Interleaved and non-interleaved buffer support
- ✅ `AudioFormatConverter` with automatic format conversions
- ✅ Two-stage conversion pipeline (source → float32 → destination)
- ✅ Proper audio normalization to [-1.0, 1.0] range
- ✅ 7 comprehensive format conversion tests

**Usage:**
```python
import coremusic as cm

# Create custom audio format
fmt = cm.PluginAudioFormat(44100.0, 2, cm.PluginAudioFormat.INT16, interleaved=True)
plugin.set_audio_format(fmt)

# Automatic conversion during processing
output = plugin.process(input_data, num_frames, fmt)
```

#### 2. User Preset Management ✅ COMPLETED
**Status:** Fully implemented and tested

**Implementation:**
- ✅ `PresetManager` class for preset save/load/export/import
- ✅ JSON-based preset storage in `~/Library/Audio/Presets/coremusic/`
- ✅ Complete parameter state capture and restoration
- ✅ Preset metadata (name, description, plugin info, timestamp)
- ✅ Export/import for preset sharing
- ✅ 6 comprehensive preset management tests

**Usage:**
```python
# Save current plugin state as preset
plugin.save_preset("My Reverb Setting", "Large hall with 3s decay")

# Load preset
plugin.load_preset("My Reverb Setting")

# List all user presets
presets = plugin.list_user_presets()

# Export/import presets
plugin.export_preset("My Reverb Setting", "/path/to/export.json")
plugin.import_preset("/path/to/preset.json")
```

#### 3. AudioUnitChain Class ✅ COMPLETED
**Status:** Fully implemented and tested

**Implementation:**
- ✅ `AudioUnitChain` class for sequential plugin processing
- ✅ Automatic format conversion between plugins
- ✅ Wet/dry mixing support (0.0 = dry, 1.0 = wet)
- ✅ Plugin insertion, removal, and configuration
- ✅ Context manager support for automatic cleanup
- ✅ 14 comprehensive chain operation tests

**Usage:**
```python
# Create and configure a plugin chain
chain = cm.AudioUnitChain()
chain.add_plugin("AUHighpass")
chain.add_plugin("AUDelay")
chain.add_plugin("AUReverb")

# Configure individual plugins
chain.configure_plugin(0, {'Cutoff Frequency': 200.0})
chain.configure_plugin(1, {'Delay Time': 0.5})

# Process audio through entire chain
output = chain.process(input_audio, num_frames, wet_dry_mix=0.8)

# Use as context manager
with cm.AudioUnitChain() as chain:
    chain.add_plugin("AUDelay")
    output = chain.process(input_data)
```

### AudioUnit Host - Future Enhancements

The following enhancements remain for future releases:

#### 4. Plugin UI Integration
**Priority: MEDIUM-LOW** - Display plugin user interfaces

**Current state:** Headless operation only (no GUI)

**Planned features:**
- Cocoa view instantiation (macOS plugin UIs)
- Window management
- UI update synchronization
- Generic UI fallback for plugins without custom UI

**Implementation effort:** 2-3 weeks
**Note:** Requires Objective-C bridge or PyObjC integration

#### 5. Link Integration for Tempo-Synced Plugins
**Priority: MEDIUM** - Tempo-aware plugin parameters

**Current state:** Manual tempo calculation required

**Planned features:**
- Tempo callback integration
- Automatic delay time sync to BPM
- Beat/bar position for tempo-synced effects
- Transport state synchronization

**Implementation effort:** 3-5 days

#### 6. Advanced MIDI Features
**Priority: LOW** - Extended MIDI capabilities

**Current state:** [x] Basic MIDI fully implemented (note on/off, CC, program change, pitch bend)

**Planned enhancements:**
- MIDI file playback through AudioUnit instruments
- Live CoreMIDI routing to instruments
- MIDI learn for parameter automation
- MIDI clock sync with Link

**Implementation effort:** 1-2 weeks

---

## Remaining Missing/Unwrapped APIs

### [ ] **AudioWorkInterval** (macOS 10.16+, iOS 14.0+)
**Priority: Medium-Low** - Advanced feature for realtime workgroup management

**What it provides:**
- OS workgroup creation for realtime audio threads
- Thread deadline coordination across processes
- CPU usage optimization for power vs. performance

**Relevance:** Highly specialized - needed only for advanced audio apps creating custom realtime threads. Most apps use device-owned workgroups automatically.

**Recommendation:** Low priority - niche use case for professional audio developers.

---

### [ ] **AudioHardwareTapping** (macOS 14.2+)
**Priority: Low** - Recent addition, Objective-C only

**What it provides:**
- Process audio tapping (capture audio from other processes)
- Requires Objective-C (`CATapDescription` class)

**Relevance:** Very specialized - audio monitoring/routing utilities, system-wide audio capture.

**Recommendation:** Low priority - requires Objective-C bridge, limited applicability.

---

### [ ] **CAFFile Data Structures**
**Priority: Low** - Informational only

**What it provides:**
- Core Audio Format (CAF) file chunk definitions
- CAF header structures

**Relevance:** Informational header - actual CAF file I/O is already handled by `AudioFile` API. Adding these structures would only help developers parsing CAF files manually (rare).

**Recommendation:** Very low priority - no functional gap.

---

### [ ] **AudioCodec Component API**
**Priority: Low-Medium** - Low-level codec interface

**What it provides:**
- Direct codec component management
- Custom encoder/decoder control
- Packet-level audio translation

**Relevance:** Very low-level API. Most use cases covered by `AudioConverter` (higher-level) and `ExtendedAudioFile` (even higher-level). Direct codec access needed only for custom codec implementations or highly specialized workflows.

**Recommendation:** Low priority - `AudioConverter` covers 95% of use cases.


## Enhancement Opportunities

### [ ] **Performance Optimizations**
**Priority: MEDIUM-LOW**

**Current state:** Already using Cython for performance.

**Potential improvements:**
```python
# Memory-mapped file reading for very large files
audio = cm.AudioFile("huge_file.wav", mode='mmap')

# Zero-copy NumPy array views
data = audio.read_frames_view()  # Returns view, not copy

# Parallel processing utilities
cm.AudioFile.batch_process_parallel(
    files=["1.wav", "2.wav", ...],
    processor=my_processing_func,
    num_workers=4
)
```

**Benefits:**
- Better performance for large files
- Reduced memory footprint
- Faster batch operations

**Implementation Effort:** Medium - requires careful memory management

---

### [ ] **Plugin/Extension System**
**Priority: LOW**

**What to add:**
```python
# User-defined AudioUnit-compatible effects
@cm.register_audio_unit(type='effect', subtype='custom')
class MyCustomReverb(cm.AudioUnit):
    def process(self, input_buffer, output_buffer, frame_count):
        # Custom DSP here
        pass

# Plugin discovery
available_plugins = cm.discover_audio_units(type='effect')
```

**Benefits:**
- Extensibility for advanced users
- Community-contributed effects
- Integration with third-party AudioUnits

**Implementation Effort:** High - requires AudioUnit host implementation

---

## Prioritized Roadmap

- [ ] **AudioUnit Host Implementation** Remaining Plugin hosting infrastructure

    1. [ ] **Audio I/O Format** - Currently processes float32 interleaved audio
       - Future: Support for more formats and non-interleaved

    2. [ ] **User Presets** - Can load factory presets but not save user presets
       - Future: ClassInfo serialization for user preset save/load

    3. [ ] **MIDI Support** - Instrument plugins can't receive MIDI yet
       - Future: MIDI event scheduling and routing

    4. [ ] **Plugin Chains** - Manual chaining required
       - Future: AudioUnitChain class for automatic routing

    5. [ ] **UI Integration** - No plugin UI display
       - Future: Cocoa view integration for plugin UIs

### Advanced Features (6-12 months)
- [ ] **Performance Optimizations** - Memory mapping, zero-copy, parallel processing
- [ ] **Plugin System** - Custom AudioUnit registration (advanced users)

### Specialized (12+ months, optional -- unlikely)
- [ ] **AudioWorkInterval** - For advanced realtime audio developers
- [ ] **AudioCodec API** - Direct codec component access (niche)
- [ ] **AudioHardwareTapping** - Process tapping (requires ObjC bridge)
