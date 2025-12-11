# TODO

See [CHANGELOG.md](CHANGELOG.md) for completed features.

---

## Active Development Tasks

### AudioUnit Host - Future Enhancements

#### Plugin UI Integration

**Priority: MEDIUM-LOW** - Display plugin user interfaces

**Current state:** Headless operation only (no GUI)

**Planned features:**
- Cocoa view instantiation (macOS plugin UIs)
- Window management
- UI update synchronization
- Generic UI fallback for plugins without custom UI

**Note:** Requires Objective-C bridge or PyObjC integration

#### Link Integration for Tempo-Synced Plugins

**Priority: MEDIUM** - Tempo-aware plugin parameters

**Current state:** Manual tempo calculation required

**Planned features:**
- Tempo callback integration
- Automatic delay time sync to BPM
- Beat/bar position for tempo-synced effects
- Transport state synchronization

#### Advanced MIDI Features

**Priority: LOW** - Extended MIDI capabilities

**Current state:** Basic MIDI fully implemented (note on/off, CC, program change, pitch bend)

**Planned enhancements:**
- MIDI file playback through AudioUnit instruments
- Live CoreMIDI routing to instruments
- MIDI learn for parameter automation
- MIDI clock sync with Link

---

### Music Module - Future Enhancements

#### Live Performance Integration

**Priority: MEDIUM** - Real-time generative performance

**Planned features:**
- Link-synchronized generators (tempo-aware)
- Real-time parameter modulation
- Pattern morphing and transitions
- Live recording of generated sequences

---

## Missing/Unwrapped CoreAudio APIs

### AudioWorkInterval (macOS 10.16+, iOS 14.0+)

**Priority: Low** - Advanced realtime workgroup management

**What it provides:**
- OS workgroup creation for realtime audio threads
- Thread deadline coordination across processes
- CPU usage optimization for power vs. performance

**Relevance:** Highly specialized - needed only for advanced audio apps creating custom realtime threads. Most apps use device-owned workgroups automatically.

---

### AudioHardwareTapping (macOS 14.2+)

**Priority: Low** - Recent addition, Objective-C only

**What it provides:**
- Process audio tapping (capture audio from other processes)
- Requires Objective-C (`CATapDescription` class)

**Relevance:** Very specialized - audio monitoring/routing utilities, system-wide audio capture.

---

### CAFFile Data Structures

**Priority: Very Low** - Informational only

**What it provides:**
- Core Audio Format (CAF) file chunk definitions
- CAF header structures

**Relevance:** Informational header - actual CAF file I/O is already handled by `AudioFile` API.

---

### AudioCodec Component API

**Priority: Low** - Low-level codec interface

**What it provides:**
- Direct codec component management
- Custom encoder/decoder control
- Packet-level audio translation

**Relevance:** Very low-level API. Most use cases covered by `AudioConverter` and `ExtendedAudioFile`.

---

## Enhancement Opportunities

### Plugin/Extension System
**Priority: LOW**

**Proposed features:**
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

---

### Documentation Improvements

**Needed:**
- Comprehensive API documentation with examples
- Tutorial series for common use cases
- Video demonstrations of key features
- Migration guides for users coming from other audio libraries

**Priority: MEDIUM** - ongoing documentation work

---

## Prioritized Roadmap

### Near-term
- [ ] **Plugin UI Integration** - Cocoa view integration for plugin UIs
- [ ] **Link-Aware Plugins** - Tempo callback integration for tempo-synced effects
- [ ] **Advanced MIDI** - MIDI file playback, live CoreMIDI routing, MIDI learn
- [ ] **Live Generative Performance** - Link-synchronized generators, real-time parameter modulation
- [ ] **Documentation** - Comprehensive API documentation with examples

### Future Enhancements
- [ ] **Plugin System** - Custom AudioUnit registration (advanced users)
- [ ] **Additional Generative Algorithms** - L-systems, cellular automata, genetic algorithms
- [ ] **Additional Examples** - More comprehensive demo applications
- [ ] **Performance Profiling Tools** - Built-in profiling for audio pipelines

### Specialized (Optional)
- [ ] **AudioWorkInterval** - For advanced realtime audio developers
- [ ] **AudioCodec API** - Direct codec component access (niche use case)
- [ ] **AudioHardwareTapping** - Process tapping (requires Objective-C bridge)

---

## Notes

For completed features and historical changes, see:
- **CHANGELOG.md** - Detailed version history and completed features
