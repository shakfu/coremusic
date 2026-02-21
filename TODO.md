# TODO

See [CHANGELOG.md](CHANGELOG.md) for completed features.

---

## Quick Fixes

All completed -- see CHANGELOG.md.

---

## Code Health

Internal quality improvements that reduce maintenance burden and prevent bugs.

### Error Handling Audit

All completed -- see CHANGELOG.md.

### Type Safety

All completed -- see CHANGELOG.md.

### Refactoring

- [x] Extract ASBD parsing into `AudioFormat.from_asbd_bytes()`
- [x] Add `AudioFormat.pcm()` factory method
- [x] Split `constants.py` into `constants/` subpackage
- [ ] Evaluate whether `CoreAudioObject` can be defined in pure Python -- **dropped**: uses Cython `__dealloc__` for guaranteed C-level cleanup, not replicable with `__del__`

### Testing

All completed -- see CHANGELOG.md.

---

## CLI

### UX Improvements

- [ ] Add usage examples to `--help` output
- [ ] Progress indicators for `play` (elapsed/total time) and `record` (elapsed time, level)
- [ ] CLI module auto-discovery via `importlib` or decorator pattern to reduce boilerplate in `main.py`

### New Commands

- [ ] `coremusic doctor` -- diagnose installation (optional deps, hardware access, available frameworks)
- [ ] `coremusic shell` -- Python REPL with coremusic preloaded
- [ ] `coremusic diff file1.wav file2.wav` -- compare spectral content, loudness, duration
- [ ] `coremusic analyze batch *.wav --output results.csv` -- batch analysis with structured export
- [ ] `coremusic watch` -- monitor directory for new audio files and auto-process (convert, normalize, analyze)

---

## API

- [ ] Rename `play_async` to `play_background` to avoid confusion with `async/await` (it is not an `async def`)
- [ ] Audio file metadata read/write (ID3 tags, iTunes metadata via `kAudioFilePropertyInfoDictionary`)
- [ ] Plugin parameter presets as YAML/JSON for reproducible processing pipelines
- [ ] MIDI learn / CC mapping for AudioUnit parameter automation

---

## Documentation

- [ ] Publish hosted API reference (Sphinx is configured but not published -- GitHub Pages or Read the Docs)
- [ ] Performance guide for real-time audio work (buffer sizing, latency, threading)

---

## Build and Distribution

- [ ] Pre-built wheels on PyPI (currently requires Xcode CLI tools to install from source)
- [ ] Wheel caching in CI to speed up builds
- [ ] Verify license implications of Ableton Link thirdparty headers in sdist

---

## Larger Initiatives

Multi-sprint efforts. Each requires design before implementation.

### Link Integration for Tempo-Synced Plugins

- [ ] Tempo callback integration
- [ ] Automatic delay time sync to BPM
- [ ] Beat/bar position for tempo-synced effects
- [ ] Transport state synchronization

### Live Performance

- [ ] Link-synchronized generators (tempo-aware)
- [ ] Real-time parameter modulation
- [ ] Pattern morphing and transitions
- [ ] Live recording of generated sequences

### Advanced MIDI

- [ ] MIDI file playback through AudioUnit instruments
- [ ] Live CoreMIDI routing to instruments
- [ ] MIDI clock sync with Link

### Plugin UI Integration

Requires Objective-C bridge or PyObjC. Significant undertaking.

- [ ] Cocoa view instantiation (macOS plugin UIs)
- [ ] Window management and UI update synchronization
- [ ] Generic UI fallback for plugins without custom UI

### Plugin/Extension System

- [ ] User-defined AudioUnit-compatible effects
- [ ] Plugin discovery API
- [ ] Community-contributed effects integration

### Real-Time Monitoring

- [ ] Terminal-based live audio dashboard (levels, spectrum, device status)

---

## Backlog (Specialized APIs)

Implement only if specific need arises.

- **AudioWorkInterval** (macOS 10.16+) -- OS workgroup creation for realtime audio threads, deadline coordination, CPU usage optimization
- **AudioHardwareTapping** (macOS 14.2+) -- Process audio tapping; requires Objective-C (`CATapDescription`)
- **AudioCodec Component API** -- Direct codec component management, custom encoder/decoder control
- **CAFFile Data Structures** -- CAF file chunk definitions and header structures (actual I/O already handled by `AudioFile`)
- **Man page generation** -- Generate man pages from argparse definitions

---

## Notes

- **macOS-only:** CoreAudio, CoreMIDI, AudioToolbox frameworks
- **Python 3.11+:** Minimum supported version
- For completed features, see **CHANGELOG.md**
