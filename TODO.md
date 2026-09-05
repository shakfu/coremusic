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
- [x] Dual license structure: MIT for core, GPLv2 for optional `coremusic.link` module

### Documentation

- [x] Publish hosted API reference

---

## Medium Priority

Meaningful improvements, moderate effort.

### CLI UX

- [ ] Add usage examples to `--help` output
- [ ] Progress indicators for `play` (elapsed/total time) and `record` (elapsed time, level)

### New CLI Commands

- [x] `coremusic audio metadata --set title="X" artist="Y"` -- CLI write path for `set_metadata`
- [x] `coremusic doctor` -- diagnose installation (optional deps, hardware access, available frameworks)
- [ ] `coremusic analyze batch *.wav --output results.csv` -- batch analysis with structured export

### Build

- [ ] Wheel caching in CI to speed up builds

### Repository Layout

- [ ] Consolidate `examples/`, `demos/`, and `extras/` under a single root
      directory. Three top-level directories of runnable-but-not-library code
      is a lot of root clutter, and the similar names invite the question of
      which is which. Keep them as distinct subdirectories of the new root
      rather than merging their contents: they hold different kinds of thing
      and are exercised by different harnesses (doc snippets pulled into pages
      by `--8<--` and run by `tests/test_examples.py`; complete programs run by
      `make demos` and smoke-checked by `tests/test_demos.py`; uninstalled
      utilities and experimental modules covered by `tests/test_extras.py` and
      the four `test_daw`/`test_music_*` modules). See the "Where new code
      goes" table in `CONTRIBUTING.md`.

      Scope is mostly mechanical but wide: 330 `--8<--` directives across the
      docs reference 241 example files by path, and 74 files outside the three
      directories name them - `tests/test_examples.py`, `test_demos.py`,
      `test_extras.py`, `test_readme.py`, and the four experimental-module test
      modules all hardcode a root; `make demos`, `make lint`, and `make format`
      name paths; `CONTRIBUTING.md`, `README.md`, `docs/examples/index.md`,
      `docs/guides/index.md`, and each directory's own README describe them.
      `mkdocs.yml` needs no change: `base_path` is anchored to
      `!relative $config_dir`, so only the include paths themselves move.

      Use `git mv` so history follows. Done when `make test`, `make test-all`,
      `make lint`, `make docs`, and `make demos` are all green, and a grep for
      the old paths outside `CHANGELOG.md` returns nothing.

---

## Lower Priority

Nice-to-have features. Implement when needed or when higher-priority items are done.

### CLI

- [x] `coremusic plugin chain <file> -p "AUDelay" -p "AUReverb2" -o out.wav` -- sequential multi-plugin processing
- [x] `coremusic device monitor` -- live stream of device changes (connect/disconnect, sample rate, volume) via polling
- [x] `coremusic midi monitor` -- formatted MIDI input display for debugging (timestamp, channel, note name, velocity)

### API

- [x] Plugin parameter presets as YAML/JSON for reproducible processing pipelines
- [ ] MIDI learn / CC mapping for AudioUnit parameter automation
- [ ] Generative MIDI toolkit -- Euclidean rhythm generator
      (`bjorklund(pulses, steps)`), a step sequencer that emits a `MIDISequence`
      or scheduled events, and chord triggers (chord name / scale degree ->
      simultaneous note-ons). Promotes the patterns in `demos/link_sequencer.py`
      and the `extras/generative` package to first-class library APIs.
- [ ] Chord and key recognition from notes -- symbolic analysis over MIDI/note
      input: identify chords from simultaneous notes (root, quality, inversion)
      and infer key from a note sequence (e.g. Krumhansl-Schmuckler key
      profiles). Distinct from the existing audio `analyze_key`, which operates
      on rendered audio rather than notes.
- [ ] Lossless Standard MIDI File round-trip -- guarantee
      `MIDISequence.load(path).save(path2)` preserves all tracks, delta timing,
      running status, tempo/time-signature/meta events, and PPQ (byte-for-byte
      where possible, otherwise semantically identical). Add a round-trip test
      over `tests/data/midi/`.
- [ ] VBR quality knob for AAC encoding. `convert`/`shortcuts.convert` currently
      expose only a target `bitrate` (ABR). True VBR-by-quality needs
      `kAudioConverterBitRateControlMode = Variable` plus
      `kAudioConverterEncodeSoundQualityForVBR` (0-127). Empirically these two
      properties are rejected via `AudioConverterSetProperty` on both the
      ExtAudioFile-owned converter and a standalone one
      (`kAudioFormatUnsupportedPropertyError`), so they likely must be pushed
      through `kExtAudioFileProperty_ConverterConfig` as a `CFArray` of
      `CFDictionary` settings -- which requires a small dedicated Cython helper
      to build the CF objects (the current bytes-based property setter cannot).
      Scope: verify the ConverterConfig CFArray route, add a `quality` param
      alongside `bitrate` (mutually exclusive), reject on lossless/PCM.

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
