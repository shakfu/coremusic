# TODO

See [CHANGELOG.md](CHANGELOG.md) for completed features.

---

## High Priority

Small-to-medium effort with immediate user-facing value.

### API

- [x] Rename `play_async` to `play_background` to clarify it is thread-based, not `async def`
- [x] Audio file metadata read/write (iTunes metadata via `kAudioFilePropertyInfoDictionary`)

### Build and Distribution

- [x] Verify license implications of Ableton Link thirdparty headers in sdist
- [x] Change project license from MIT to GPLv3 for Ableton Link compatibility

### Documentation

- [ ] Publish hosted API reference

---

## Medium Priority

Meaningful improvements, moderate effort.

### CLI UX

- [ ] Add usage examples to `--help` output
- [ ] Progress indicators for `play` (elapsed/total time) and `record` (elapsed time, level)

### New CLI Commands

- [ ] `coremusic audio metadata --set title="X" artist="Y"` -- CLI write path for `set_metadata` (API exists, CLI surface missing)
- [ ] `coremusic doctor` -- diagnose installation (optional deps, hardware access, available frameworks)
- [ ] `coremusic analyze batch *.wav --output results.csv` -- batch analysis with structured export

### Build

- [ ] Wheel caching in CI to speed up builds

---

## Lower Priority

Nice-to-have features. Implement when needed or when higher-priority items are done.

### CLI

- [ ] `coremusic plugin chain <file> -p "AUDelay" -p "AUReverb2" -o out.wav` -- sequential multi-plugin processing
- [ ] `coremusic convert resample <file> --rate 44100` -- explicit sample rate conversion
- [ ] `coremusic audio concat a.wav b.wav -o combined.wav` -- concatenate audio files
- [ ] `coremusic analyze compare file1.wav file2.wav` -- diff two files by duration, loudness, spectrum, format
- [ ] `coremusic device monitor` -- live stream of device changes (connect/disconnect, sample rate, volume) via CoreAudio property listeners
- [ ] `coremusic midi monitor` -- formatted MIDI input display for debugging (timestamp, channel, note name, velocity)
- [ ] `coremusic watch` -- monitor directory for new audio files and auto-process
- [ ] `coremusic shell` -- Python REPL with coremusic preloaded
- [ ] CLI module auto-discovery via `importlib` or decorator to reduce boilerplate in `main.py`

### API

- [ ] Plugin parameter presets as YAML/JSON for reproducible processing pipelines
- [ ] MIDI learn / CC mapping for AudioUnit parameter automation

### Documentation

- [ ] Performance guide for real-time audio work (buffer sizing, latency, threading)

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
- **Python 3.10+:** Minimum supported version
- For completed features, see **CHANGELOG.md**
