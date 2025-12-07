# TODO

## Summary

**CoreMusic Status:** Production-ready professional audio framework for Python with **1,329+ tests passing** (100% success rate)

**Current Capabilities:**
- Complete AudioUnit Host (190 plugins: 111 effects, 62 instruments)
- Full MIDI support with AudioUnit instruments
- Ableton Link tempo synchronization
- Complete CoreMIDI and CoreAudio wrappers
- Performance-optimized with Cython operations, buffer pooling, and memory-mapped I/O
- Proper optional dependency handling (numpy, scipy, matplotlib)
- DAW essentials (Timeline, tracks, clips, automation)
- Music theory primitives (Note, Interval, Scale, Chord, ChordProgression)
- Generative algorithms (Arpeggiator, Euclidean, Markov, Probabilistic, Sequence, Melody, Polyrhythm)

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

#### Additional Generative Algorithms

**Priority: LOW** - Expand generative capabilities

**Current state:** 7 generators (Arpeggiator, Euclidean, Markov, Probabilistic, Sequence, Melody, Polyrhythm)

**Potential additions:**
- L-system based melody generation
- Cellular automata rhythms
- Genetic algorithm composition
- Neural network integration (optional dependency)
- Stochastic grammar-based generation

#### Live Performance Integration

**Priority: MEDIUM** - Real-time generative performance

**Planned features:**
- Link-synchronized generators (tempo-aware)
- Real-time parameter modulation
- Pattern morphing and transitions
- Live recording of generated sequences

#### MIDI File Transformation Pipeline

**Priority: MEDIUM** - Analyze and transform MIDI files

**Current state:** MIDI file read/write supported via `MIDISequence`, but no transformation pipeline

**Planned features:**
- Load and analyze existing MIDI files (extract notes, timing, structure)
- Chain of MIDI transformers for processing:
  - Transpose (shift pitch by semitones or interval)
  - Quantize (snap to grid, swing)
  - Velocity scaling/compression
  - Time stretch/compress
  - Note filtering (by pitch range, velocity, channel)
  - Humanize (add timing/velocity variation)
  - Harmonize (add parallel intervals)
  - Arpeggiate chords
  - Reverse/retrograde
  - Invert melody
- Composable pipeline with fluent API
- Export transformed MIDI to new file

**Example API:**
```python
from coremusic.midi.utilities import MIDISequence
from coremusic.midi.transform import (
    Transpose, Quantize, VelocityScale, Humanize, Pipeline
)

# Load MIDI file
seq = MIDISequence.load("input.mid")

# Create transformation pipeline
pipeline = Pipeline([
    Transpose(semitones=5),           # Up a perfect fourth
    Quantize(grid=1/16, strength=0.8),  # Quantize to 16th notes
    VelocityScale(min_vel=40, max_vel=100),  # Compress velocity
    Humanize(timing=0.02, velocity=10),  # Add human feel
])

# Apply transformations
transformed = pipeline.apply(seq)

# Save result
transformed.save("output.mid")
```

---

## Missing/Unwrapped CoreAudio APIs

### AudioWorkInterval (macOS 10.16+, iOS 14.0+)

**Priority: Medium-Low** - Advanced realtime workgroup management

**What it provides:**
- OS workgroup creation for realtime audio threads
- Thread deadline coordination across processes
- CPU usage optimization for power vs. performance

**Relevance:** Highly specialized - needed only for advanced audio apps creating custom realtime threads. Most apps use device-owned workgroups automatically.

**Priority: Low priority**  - niche use case for professional audio developers.

---

### AudioHardwareTapping (macOS 14.2+)

**Priority: Low** - Recent addition, Objective-C only

**What it provides:**
- Process audio tapping (capture audio from other processes)
- Requires Objective-C (`CATapDescription` class)

**Relevance:** Very specialized - audio monitoring/routing utilities, system-wide audio capture.

**Priority: Low priority** - requires Objective-C bridge, limited applicability.

---

### CAFFile Data Structures

**Priority: Low** - Informational only

**What it provides:**
- Core Audio Format (CAF) file chunk definitions
- CAF header structures

**Relevance:** Informational header - actual CAF file I/O is already handled by `AudioFile` API. Adding these structures would only help developers parsing CAF files manually (rare).

**Priority: Very Low priority** - no functional gap.

---

### AudioCodec Component API

**Priority: Low-Medium** - Low-level codec interface

**What it provides:**
- Direct codec component management
- Custom encoder/decoder control
- Packet-level audio translation

**Relevance:** Very low-level API. Most use cases covered by `AudioConverter` (higher-level) and `ExtendedAudioFile` (even higher-level). Direct codec access needed only for custom codec implementations or highly specialized workflows.

**Priority: Low priority** - `AudioConverter` covers 95% of use cases.

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

**Priority: High priority** - requires custom AudioUnit host implementation

---

### Documentation Improvements

**Needed:**
- Comprehensive API documentation with examples
- Tutorial series for common use cases
- Video demonstrations of key features
- Migration guides for users coming from other audio libraries

**Priority: MEDIUM**- ongoing documentation work

---

## Prioritized Roadmap

### Near-term (Next 3-6 months)
- [ ] **MIDI Transformation Pipeline** - Chainable MIDI transformers (transpose, quantize, humanize, etc.)
- [ ] **Plugin UI Integration** - Cocoa view integration for plugin UIs
- [ ] **Link-Aware Plugins** - Tempo callback integration for tempo-synced effects
- [ ] **Advanced MIDI** - MIDI file playback, live CoreMIDI routing, MIDI learn
- [ ] **Documentation** - Comprehensive API documentation with examples

### Future Enhancements (6-12 months)
- [ ] **Plugin System** - Custom AudioUnit registration (advanced users)
- [ ] **Additional Generative Algorithms** - L-systems, cellular automata, genetic algorithms
- [ ] **Additional Examples** - More comprehensive demo applications
- [ ] **Performance Profiling Tools** - Built-in profiling for audio pipelines

### Specialized (Optional, unlikely)
- [ ] **AudioWorkInterval** - For advanced realtime audio developers
- [ ] **AudioCodec API** - Direct codec component access (niche use case)
- [ ] **AudioHardwareTapping** - Process tapping (requires Objective-C bridge)

---

## Notes

For completed features and historical changes, see:
- **CHANGELOG.md** - Detailed version history and completed features
