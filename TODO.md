# TODO

## Done

### [x] **Packaging and Distribution**
**Priority: MEDIUM**

**Current state:** Builds from source.

**Improvements:**
- **PyPI distribution** with pre-built wheels for macOS
- **Conda package** for conda-forge
- **ARM64 optimization** for Apple Silicon
- **Universal2 binaries** (x86_64 + arm64)

**Benefits:**
- Easier installation (`pip install coremusic`)
- Broader reach
- No compilation required for end users

**Implementation Effort:** Medium - CI/CD setup for wheel building


### [x] **CoreAudioClock** (macOS-only)
**Priority: Medium** - Synchronization and timing services

**What it provides:**
- Audio/MIDI synchronization
- SMPTE timecode support
- MIDI Time Code (MTC) and MIDI beat clock
- Tempo maps and time conversions
- Clock sources (audio devices, host time, external sync)

**Relevance:** Essential for DAWs, sequencers, and sync-dependent applications. Overlaps partially with MusicPlayer tempo functionality but provides broader sync capabilities.



## Missing/Unwrapped APIs

### [ ] **AudioUnit Host Implementation**
**Priority: HIGH** - Essential for plugin hosting and advanced audio processing

**What it provides:**
- Host AudioUnit plugins (VST-like plugins in macOS)
- Load and instantiate third-party AudioUnits
- Route audio through plugin chains
- Automate plugin parameters
- Save/restore plugin state
- Build custom DAW-like applications

**Key capabilities:**
- **Plugin discovery** - Find available AudioUnits by type/subtype/manufacturer
- **Plugin instantiation** - Load and configure AudioUnit plugins
- **Audio routing** - Connect plugins in processing chains
- **Parameter automation** - Control plugin parameters in realtime
- **Preset management** - Save/load plugin presets
- **UI integration** - Show plugin UI windows (via Cocoa bridge)

**Current state:** CoreMusic has low-level AudioUnit wrapping (`audio_component_find_next`, `audio_component_instance_new`, etc.) but lacks high-level host infrastructure.

**What's needed:**
```python
# High-level AudioUnit host API
host = cm.AudioUnitHost()

# Discover available plugins
plugins = host.discover_audio_units(type='effect', subtype='reverb')
print(f"Found {len(plugins)} reverb plugins")

# Load a plugin
reverb = host.load_audio_unit(plugins[0])
reverb.initialize(sample_rate=44100, channels=2)

# Set parameters
reverb.set_parameter('room_size', 0.8)
reverb.set_parameter('decay_time', 2.5)

# Process audio
output = reverb.process(input_audio)

# Save preset
reverb.save_preset('my_favorite_reverb.aupreset')
```

**Integration points:**
- Build on existing AudioComponent/AudioUnit wrappers
- Add parameter enumeration and control
- Add preset serialization (CFPropertyList)
- Add audio buffer routing
- Add render callback management

**Benefits:**
- Host third-party audio plugins in Python
- Build custom DAWs and audio tools
- Leverage the vast AudioUnit ecosystem
- Professional audio processing workflows
- Unique capability in Python ecosystem

**Implementation Effort:** HIGH (2-3 weeks)
- Plugin discovery and loading: 3-4 days
- Parameter management: 2-3 days
- Audio routing infrastructure: 4-5 days
- Preset management: 2-3 days
- Testing and documentation: 3-4 days

**Recommendation:** HIGH priority - Enables professional plugin hosting, differentiates CoreMusic from basic audio libraries, provides significant value to music production community.

---

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

---

## Integration Opportunities

### [ ] **Ableton Link Integration**
**Priority: HIGH** - Tempo synchronization and network music capabilities

**What it provides:**
- Multi-device tempo synchronization over local network
- Shared beat grid and phase alignment
- Transport state synchronization (play/stop)
- Sub-millisecond precision timing
- Automatic peer discovery and connection

**Key capabilities:**
- **Network tempo sync** - Multiple apps stay in perfect tempo sync
- **Beat quantization** - Start/stop on beat boundaries
- **DAW integration** - Sync with Ableton Live, Bitwig, etc.
- **Collaborative jamming** - Multiple musicians playing in sync over network
- **Professional workflows** - Industry-standard sync protocol

**Current state:** Ableton Link library exists in `thirdparty/link/` but not yet integrated with CoreMusic.

**What's needed:**
```python
# Create Link session
link = cm.Link(bpm=120.0)
link.enabled = True

# Monitor peers
print(f"Connected to {link.num_peers} peers")

# Get synchronized beat/phase
state = link.capture_audio_session_state()
beat = state.beat_at_time(host_time, quantum=4.0)
phase = state.phase_at_time(host_time, quantum=4.0)

# Use in audio callback for beat-accurate playback
player = cm.AudioPlayer(link=link, quantum=4.0)
player.load_file("loop.wav")
player.start()  # Automatically quantizes to beat grid
```

**Integration points:**
- Cython wrapper for C++ Link API (`link.pyx`)
- Clock integration with `mach_absolute_time()`
- AudioUnit render callback integration
- Output latency compensation
- Python high-level API with callbacks

**Benefits:**
- **Unique in Python** - No other Python Link wrapper exists
- **Professional workflows** - Enables serious music production
- **Network music** - Multi-device jam sessions
- **Beat-accurate** - Quantized playback and recording
- **Cross-platform** - Works with 100+ Link-enabled apps

**Implementation Effort:** MEDIUM-HIGH (7-10 days)
- Basic C++ wrapper: 2-3 days
- AudioEngine integration: 2-3 days
- Python API + callbacks: 1-2 days
- Testing + documentation: 2 days

**Documentation:** See `docs/dev/ableton_link.md` for complete integration analysis

**Recommendation:** HIGH priority - Positions CoreMusic as complete professional audio framework, provides unique value in Python ecosystem, proven stable technology with minimal risk.

---

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

---

## Prioritized Roadmap

### Foundation (Immediate)
1. [x] **PyPI Distribution** - Pre-built wheels
2. [x] **CoreAudioClock** - Sync and timecode support

### High-Priority Integrations (0-6 months)
3. [ ] **Ableton Link Integration** - Network tempo sync and beat quantization (7-10 days)
   - Enables professional music production workflows
   - Unique capability in Python ecosystem
   - See `docs/dev/ableton_link.md` for details

4. [ ] **AudioUnit Host Implementation** - Plugin hosting infrastructure (2-3 weeks)
   - Load and route third-party AudioUnit plugins
   - Parameter automation and preset management
   - Build custom DAWs and effects chains

### Advanced Features (6-12 months)
5. [ ] **Performance Optimizations** - Memory mapping, zero-copy, parallel processing
6. [ ] **Plugin System** - Custom AudioUnit registration (advanced users)

### Specialized (12+ months, optional)
7. [ ] **AudioWorkInterval** - For advanced realtime audio developers
8. [ ] **AudioCodec API** - Direct codec component access (niche)
9. [ ] **AudioHardwareTapping** - Process tapping (requires ObjC bridge)

---

## Final Recommendations

### Skip or Defer
- **AudioHardwareTapping** - Too specialized, requires ObjC
- **AudioCodec API** - Already covered by AudioConverter
- **CAFFile structures** - No functional value
