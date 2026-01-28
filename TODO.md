# TODO

See [CHANGELOG.md](CHANGELOG.md) for completed features.

---

## High Priority

### Documentation

**Priority: HIGH** - Critical for adoption

**Current state:** Comprehensive documentation with tutorials, cookbook, and API reference

**Completed:**
- [x] Comprehensive API documentation with examples
- [x] Tutorial series for common use cases (audio playback, MIDI, effects processing)
- [x] Quick-start guide for new users
- [x] Migration guides for users coming from other audio libraries

**Tutorials added:**
- `audio_playback.rst` - Simple to advanced audio playback
- `audio_recording.rst` - Recording from input devices
- `effects_processing.rst` - AudioUnit effects processing
- `midi_basics.rst` - MIDI fundamentals
- `quickstart.rst` - 5-minute getting started guide

---

### CI/CD Pipeline

**Priority: HIGH** - Ensures code quality and release automation

**Needed:**
- [ ] GitHub Actions workflow for automated testing on push/PR
- [ ] Multi-Python version testing (3.11, 3.12, 3.13, 3.14)
- [ ] Automated wheel builds for releases
- [ ] Test coverage reporting

---

## Medium Priority

### Link Integration for Tempo-Synced Plugins

**Priority: MEDIUM** - Tempo-aware plugin parameters

**Current state:** Manual tempo calculation required

**Planned features:**
- [ ] Tempo callback integration
- [ ] Automatic delay time sync to BPM
- [ ] Beat/bar position for tempo-synced effects
- [ ] Transport state synchronization

---

### Live Performance Integration

**Priority: MEDIUM** - Real-time generative performance

**Planned features:**
- [ ] Link-synchronized generators (tempo-aware)
- [ ] Real-time parameter modulation
- [ ] Pattern morphing and transitions
- [ ] Live recording of generated sequences

---

## Lower Priority

### Plugin UI Integration

**Priority: LOW** - Display plugin user interfaces

**Current state:** Headless operation only (no GUI)

**Planned features:**
- [ ] Cocoa view instantiation (macOS plugin UIs)
- [ ] Window management
- [ ] UI update synchronization
- [ ] Generic UI fallback for plugins without custom UI

**Note:** Requires Objective-C bridge or PyObjC integration. Significant undertaking.

---

### Advanced MIDI Features

**Priority: LOW** - Extended MIDI capabilities

**Current state:** Basic MIDI fully implemented (note on/off, CC, program change, pitch bend)

**Planned enhancements:**
- [ ] MIDI file playback through AudioUnit instruments
- [ ] Live CoreMIDI routing to instruments
- [ ] MIDI learn for parameter automation
- [ ] MIDI clock sync with Link

---

### Plugin/Extension System

**Priority: LOW** - Custom AudioUnit registration

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

## Backlog (Specialized APIs)

These are specialized CoreAudio APIs with niche use cases. Implement only if specific need arises.

### AudioWorkInterval (macOS 10.16+, iOS 14.0+)

**Priority: BACKLOG** - Advanced realtime workgroup management

- OS workgroup creation for realtime audio threads
- Thread deadline coordination across processes
- CPU usage optimization for power vs. performance

**Relevance:** Needed only for advanced audio apps creating custom realtime threads. Most apps use device-owned workgroups automatically.

---

### AudioHardwareTapping (macOS 14.2+)

**Priority: BACKLOG** - Process audio tapping

- Process audio tapping (capture audio from other processes)
- Requires Objective-C (`CATapDescription` class)

**Relevance:** Very specialized - audio monitoring/routing utilities, system-wide audio capture.

---

### AudioCodec Component API

**Priority: BACKLOG** - Low-level codec interface

- Direct codec component management
- Custom encoder/decoder control
- Packet-level audio translation

**Relevance:** Very low-level API. Most use cases covered by `AudioConverter` and `ExtendedAudioFile`.

---

### CAFFile Data Structures

**Priority: BACKLOG** - Informational only

- Core Audio Format (CAF) file chunk definitions
- CAF header structures

**Relevance:** Informational header - actual CAF file I/O is already handled by `AudioFile` API.

---

## Prioritized Roadmap Summary

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| HIGH | Documentation | Medium | High |
| HIGH | CI/CD Pipeline | Low | High |
| MEDIUM | Link-Tempo Plugin Sync | Medium | Medium |
| MEDIUM | Live Performance Integration | High | Medium |
| LOW | Plugin UI Integration | High | Low |
| LOW | Advanced MIDI Features | Medium | Low |
| LOW | Plugin/Extension System | High | Low |
| BACKLOG | Specialized CoreAudio APIs | Variable | Niche |

---

## Notes

- **macOS-only:** This project targets macOS exclusively (CoreAudio, CoreMIDI, AudioToolbox frameworks)
- **Python 3.11+:** Minimum supported Python version is 3.11
- For completed features and historical changes, see **CHANGELOG.md**
