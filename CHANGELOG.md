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

## [Unreleased]

## [0.2.7]

### Changed

- **101 broad `except Exception` handlers narrowed to `FRAMEWORK_ERRORS`** - a handler that catches `Exception` and returns a fallback cannot tell a CoreAudio refusal from a defect in this library, so it converted `AttributeError`, `TypeError` and `NameError` into a plausible-looking value the caller could not question. `AudioFileStream.ready_to_produce_packets` is the worked example: it called a function that did not exist and reported "not ready" for every stream in every state.

  `coremusic.exceptions.FRAMEWORK_ERRORS` names what the layers below actually raise - `CoreAudioError` and its subclasses from the object wrappers, `RuntimeError` from `capi` for a non-zero `OSStatus`, and `OSError` from file operations. `ValueError` (which `capi` raises for an invalid argument), `MemoryError`, `TypeError` and `IndexError` are deliberately excluded: they signal a bad call or an exhausted machine, not a refused operation.

  Two handlers written as `except (AudioDeviceError, Exception)` were also narrowed. The tuple form reads as though it is specific while `Exception` subsumes the other member, and it hid from the first pass of this audit.

- **11 handlers stay broad, each with the reason beside it** - invoking a caller-supplied callback or work function (which may raise anything), the top of a worker-thread loop (where dying silently is worse than continuing), and `cli doctor` (whose contract is to report any failure rather than raise). Each carries a `noqa: BLE001` next to the explanation.

- **`convert batch` keeps going when one file has an unsupported format** - narrowing this handler would have let the `ValueError` that `convert_audio_file` raises for an unwritable format abort the whole batch, so it catches `ValueError` alongside the framework errors. For one file among many, a rejected format is a per-file result.

- **Malformed plugin presets are now detected rather than caught** - `AudioUnitPlugin.load_preset` checked `param_data["value"]` inside the guarded block, so a preset entry missing its value was indistinguishable from the plugin refusing the value. The shape is validated explicitly and a warning names the offending parameter.

- **`ruff` now enforces `BLE` (blind-except) on `src/`** - the lint policy in `pyproject.toml` had this family deferred with a note that adopting it "would mean narrowing 242 `except Exception` clauses ... see docs/dev/error_decorator.md, where that refactor was deferred". That work is done, so the rule is on, with `tests/`, `examples/`, `extras/` and `demos/` exempt.

### Added

- **`coremusic.exceptions.FRAMEWORK_ERRORS`** - the shared tuple described above, documented with what belongs in it and what deliberately does not.

- **`tests/test_exception_narrowing.py`** - pins the meaning of `FRAMEWORK_ERRORS` (it must cover what `capi` raises and must not cover the types that signal a bug), checks at a real call site that a programming error propagates while a CoreAudio refusal is still absorbed, and fails if a broad swallowing handler appears outside the documented allowlist or if a listed exemption goes stale. The detector counts the tuple form, which is how the two `except (AudioDeviceError, Exception)` sites were found.


### Fixed

- **52 wrong values in `coremusic.constants`** - the enum layer restated CoreAudio constants as hand-typed integers, and 42% of those carrying a FourCC comment did not encode the FourCC they documented. Several decoded to byte sequences that are not valid FourCC at all (`'nsr\xf2'`, `'q_u`'`, `'fn`j'`), so they were never transcribed from any header. Anyone using the documented enum API rather than the `capi.get_*` getters passed a wrong property ID to CoreAudio: `ExtendedAudioFileProperty.FILE_LENGTH_FRAMES` raised `Unknown error code -66561`, and `AudioDeviceProperty.NOMINAL_SAMPLE_RATE` - off by 126 in its last byte - raised `kAudioCodecUnknownPropertyError`.

  Values are no longer typed by hand. Each enum member is mapped to the name of the C constant it stands for, a C probe printing all 228 is compiled against the macOS SDK, and its output is what appears in the source. A name that does not exist in the SDK fails to compile, which is how the fabricated constants below were found rather than guessed at. Every constant now carries its C name in a trailing comment, so the mapping is auditable from the source alone.

  Corrections were not limited to FourCC typos: `AudioUnitProperty.OFFLINE_RENDER` was 55 rather than 37, `CHANNEL_MAP` was 33 - colliding with `PARAMETER_STRING_FROM_VALUE` - rather than 2002, and the three offline `AudioUnitRenderActionFlags` were each shifted one bit position.

- **`AudioFileStream.ready_to_produce_packets` always returned `False`** - it called `capi.audio_file_stream_get_property_ready_to_produce_packets`, which does not exist in the compiled extension. The resulting `AttributeError` was caught by an enclosing `except Exception: return False`, so the property reported "not ready" for every stream in every state. `capi.pyi` declared the function, so mypy could not see the error, and the test asserted only `not ready` on an empty stream - true whether or not the implementation worked. The property now uses the generic `audio_file_stream_get_property` and narrows its handler to `except RuntimeError` so coding errors propagate.

- **CoreAudio handles leaked when objects were garbage collected** - `CoreAudioObject` documented automatic resource management, but `__dealloc__` could only reach the `cdef` `_dispose_internal`, which a Python subclass cannot override. Every subclass `dispose()` was therefore skipped at collection time, leaking the underlying handle unless the caller used `with` or `close()`. Measured at one file descriptor per unclosed `AudioFile`. Release now happens in `__del__` (`tp_finalize`), which runs before deallocation and does normal method lookup, so subclass overrides are reached; `__dealloc__` remains as a backstop. Explicit `with`/`close()` is still preferred, since audio hardware should be released at a predictable point.

- **12 phantom declarations in `capi.pyi`** - names the stub promised that the compiled module does not export. Calls to them raise `AttributeError` at runtime while mypy reports the code as clean.

### Removed

- **`MIDIObjectProperty`** - a 26-member `IntEnum` of CoreMIDI property IDs describing an API that does not exist. CoreMIDI keys properties by `CFStringRef`, not integer FourCC; the library's own `coremidi.pxd` already declares the correct `MIDIObjectGetStringProperty` and `MIDIObjectGetIntegerProperty` signatures and uses them. The enum could not be passed to any CoreMIDI call, had no call sites, and no test coverage.

- **8 fabricated constants naming no SDK symbol** - `AudioFileProperty.DATA_SIZE` and `DATA_IS_BIG_ENDIAN`; `AudioFormatID.GSM610` and `ADPCM_IMA_WAV` (WAV format tags, not `AudioFormatID`s); `AudioFormatID.MPEG4_AAC_LD_V2` and `MPEG4_AAC_HE_V2_SBR`; `AudioObjectProperty.DEVICE_NAME_IN_OWNER_USER_INTERFACE` and `CLASS_NAME`.

### Added

- **`tests/test_constants_integrity.py`** - reconciles the enum layer against the compiled getters and the SDK, the check that was missing. Four layers: FourCC comments must round-trip their integer; enum members must equal the `capi.get_*` getter naming the same constant (36 pairs, up from the 4 checked previously); all 205 annotated constants are verified against a probe compiled at test time (marked `slow`, skipped where no compiler or SDK is present); and `capi.pyi` must not declare names the extension lacks. A fifth test guards the parser the others depend on. Verified to fail, not merely to pass: corrupting one constant by a single digit fails three checks independently.

- **`tests/test_resource_release.py`** - covers the subclass-dispatch regression directly and measures descriptor counts across 50 unclosed `AudioFile` opens.

- **10 real constants** replacing removed fabrications or filling gaps the compiled getters already exposed - `AudioFileProperty.IS_OPTIMIZED` and `MAGIC_COOKIE_DATA`; `AudioObjectProperty.MODEL_NAME`, `SERIAL_NUMBER`, `FIRMWARE_VERSION`; `AudioDeviceProperty.STREAM_CONFIGURATION`, `VOLUME_SCALAR`, `MUTE`, `IS_HIDDEN`, `PREFERRED_CHANNELS_FOR_STEREO`.

### Changed

- **`AudioQueueParameter.VolumeRampTime` renamed to `VOLUME_RAMP_TIME`** to match the UPPER_SNAKE_CASE of every other member. The old name is kept as an enum alias, so existing callers are unaffected.

- **`AudioUnit.sample_rate` uses `AudioUnitProperty.SAMPLE_RATE`** instead of a bare literal `2` explained by a trailing comment, at four call sites.


## [0.2.6]

### Added

- **MIDI channel voice message builders (`coremusic.midi.messages`)** - `note_on`, `note_off`, `control_change`, `program_change`, `pitch_bend`, `poly_aftertouch`, `channel_aftertouch`, `all_notes_off`, and `all_sound_off`, each returning the `bytes` that `send_data()` takes. Nothing in the library previously produced a wire message: `capi.midi_note_on` returns a tuple for the AudioUnit MusicDevice call, `MIDIEvent.to_bytes()` needs an event time you do not have when sending, and `MIDIStatus` only names the status nibble. So every send in the codebase, including the CLI, hand-assembled `bytes([0x90 | channel, note, velocity])`.

  A note may be a MIDI number, a name, or a `Note`: `note_on(60, 100)`, `note_on("C4", 100)`, and `note_on(Note("C", 4), 100)` are the same message. This connects `coremusic.music.theory`, which has had `Note`, `note_name_to_midi`, and `Scale` all along, to the wire layer for the first time. Octave numbering follows scientific pitch notation, matching `note_name_to_midi`, so middle C is `"C4"` is 60; Ableton Live and Logic display that note as C3.

  Channel is keyword-only. `capi.midi_note_on` takes `(channel, note, velocity)`, so a positional channel would make `note_on(0, 60, 100)` build a valid but entirely different message; it raises `TypeError` instead. When a `Note` is passed and no velocity is given, the note's own velocity applies.

  Out-of-range arguments raise `ValueError` naming the offending argument. `MIDIEvent.to_bytes()` masks instead, silently turning a velocity of 200 into 72, and a data byte above 127 reads as a status byte and desynchronises the rest of the stream.

### Changed

- **Every hand-assembled MIDI message now uses the builders** - across the CLI (`midi send`, `midi panic`, `midi play`, `sequence play`), the examples, the tutorials, the README, and the tests. Verified byte-identical to the expressions they replace across all 16 channels and every message type. The byte-level tests in `TestMIDIMessageSplitter` deliberately keep their hex literals: they cover running status, orphan data bytes, aborted SysEx, and a realtime byte landing mid-message, none of which a builder can express.

- **MIDI transport tests name a shared constant for each message** - the send and the assertion refer to the same `NOTE_ON_E4`, so they cannot drift apart. That drift is exactly what made `test_input_port_receives_from_virtual_source` send two messages while asserting on one packet.

- **The MIDI Message Reference in the MIDI Basics tutorial** is now a runnable, tested program (`examples/tutorials/midi_basics/message_reference.py`) rather than five blocks of byte literals, and documents that the `capi.midi_*` triples are not interchangeable with wire messages: `bytes(capi.midi_program_change(...))` appends a `0x00` that a receiver reads as data for a running-status message, because Program Change is two bytes.

- **The MIDI Basics tutorial no longer teaches hand-rolled note-name conversion** - two doctests reimplemented `note_name_to_midi` and `midi_to_note_name` inline; they now use the library functions, which validate their input and cover enharmonic spellings.

### Fixed

- **`ScaleFilter` documented a constructor that does not exist** - its docstring showed `Scale(Note.from_name("C4"), ...)`; `Note` has no `from_name`. Corrected to `Scale(Note("C", 4), ...)`.

## [0.2.5]

### Added

- **`demos/plot_audio.py`** - Renders a waveform, spectrogram, or frequency spectrum to a PNG, replacing the three separate visualization scripts from `tests/demos/`. Plotting was the one capability those scripts covered that nothing else demonstrates as a runnable program.

- **`demos/stream_latency.py`** - What each real-time buffer size costs in latency, computed or measured against a live loopback with `--measure`.

- **`MIDIEndpoint` and endpoint discovery ([#2](https://github.com/shakfu/coremusic/issues/2))** - `MIDIOutputPort.send_data()` takes a destination endpoint, but the object layer had no way to obtain one: `MIDIClient` could create ports and nothing else, and endpoints only existed as integer handles in `capi`. Every documented send example was therefore unrunnable, referring to an undefined `destination` or to a `client.create_virtual_destination()` that did not exist. Added `MIDIEndpoint`, wrapping a MIDI source or destination with its name and, for virtual endpoints, ownership; `MIDIClient.create_virtual_source()` and `create_virtual_destination()`, both disposed with the client; and the module-level `get_sources()`, `get_destinations()`, `find_source()`, and `find_destination()` for the endpoints published by the system. A virtual destination buffers or dispatches incoming packets exactly as an input port does (`poll()`, `wait()`, `pending`, `dropped`), and `MIDIEndpoint.send()` produces MIDI from a virtual source.

- **Context managers for the MIDI objects** - `MIDIClient`, `MIDIPort`, and `MIDIEndpoint` support `with`, matching the audio objects and the documentation's claim that the object API cleans up automatically.

- **`AudioFile.path` and `AudioFile.packet_count`** - The packet count is the bound `read_packets()` is checked against, so reading a file in chunks previously meant either reading everything with `read_as_numpy()` or going to `capi` for the property. Both are now readable from the object.

- **`LinkSession` is a context manager** - `with LinkSession(bpm=120.0) as session:` enables networking on entry and disables it on exit, which is how every Link example was already written.

- **`MIDIClientCreate` explains a dead connection** - Statuses -2 and -304 mean this process lost its connection to MIDIServer, after the daemon exited following the disconnection of its last client. Nothing in the process can create a client again, so the bare "Unknown error code -2" CoreMIDI reports is the least useful moment to leave a caller guessing. The error now names the cause, says that retrying cannot help, and says what to do instead. `docs/tutorials/midi_basics.md` gains a "How Long to Keep a Client" section with a runnable example, and a troubleshooting entry for the status code itself.

### Fixed

- **A MIDI receive test waited for the wrong thing** - `test_input_port_receives_from_virtual_source` sends two messages, then waited for *one packet* before asserting both had arrived. CoreMIDI decides how many messages travel in a packet, so the test passed only while the two happened to be coalesced, and failed on a loaded CI runner that delivered them separately - reporting `assert b'\x82\x15 ' in [b'\x92\x15E']`. Tests now wait on a message count rather than a packet count, via a `collect_messages` helper that keeps one splitter across polls so a message spanning packets is still assembled. Confirmed by forcing the split: the old form fails with the CI error, the new one passes.

- **CI failed intermittently on a single Python version** - `MIDIClientCreate` was refused partway through a run with an undocumented status (-2 and -304 both seen), failing two tests on Python 3.12 while the other four versions passed the same commit. The system log gives the cause: MIDIServer is an on-demand daemon that exits a few seconds after its last client disconnects, and that invalidates the CoreMIDI connection inside every process still running. The client framework does not re-establish it, so from that moment every `MIDIClientCreate` in the process fails with "null connection" for the rest of the run - which is why retrying and waiting never helped, and why a fresh process always worked. The suite creates and disposes clients in bursts, so it regularly leaves a window with none alive; whether the server idled out during one was a race. A session-scoped fixture now holds one client open for the whole run. Under conditions that previously produced 84 skips, four consecutive runs are now identical at 38. Client creation also goes through `conftest.midi_or_skip`, which retries and then skips, as a backstop for machines with no MIDI at all.

- **Roughly twenty MIDI tests were silently skipping** - `tests/conftest.py` reorders MIDI modules to run before the audio tests, because CoreAudio activity makes CoreMIDI unavailable later in a run - but `test_midi_endpoints`, `test_midi_transform`, and `test_link_midi` were never added to that list, so they ran last and mostly skipped. Running each module alone hid it: `test_midi_endpoints` passed 26 of 26 in isolation while skipping 19 of 26 in a full run. With the modules ordered correctly, the full suite goes from 2186 to 2210 passing.

- **`convert()` could not write AIFF** - AIFF stores big-endian signed integer PCM, but `shortcuts.convert()` derives the output format from the source, so any WAV input described the output as little-endian and `ExtAudioFileCreateWithURL` rejected it outright with `kAudioFileStreamError_UnsupportedDataFormat`. `convert("input.wav", "output.aiff")` was advertised in the README and had never worked. The PCM output format is now re-described for the container.

- **Six documentation examples ran without exercising anything** - They defined a function and never called it, so executing them proved only that the file imported. Two were broken behind that: one passed two arguments to `AudioConverter.convert()`, which takes one, and another re-read packet 0 on every iteration of its chunked-read loop. `tests/test_examples.py` now fails an example that references none of its own definitions.

- **Duplicate key in `RECOVERY_SUGGESTIONS`** - `-43` was listed twice in the same dict literal, so the first suggestion was dead code and the second silently won. Kept one entry.

- **`ruff` was declared as a runtime dependency** - It landed in `[project] dependencies` rather than the dev group, which already carried it, so every install of a package whose README leads with "zero-dependency" would have pulled a linter. Moved to the dev group and pinned to the version the lint configuration targets.

- **Stale module paths in the developer notes** - Eight notes under `docs/dev/` name `src/coremusic/objects.py`, `utilities.py`, `audiounit_host.py`, `link_midi.py`, and `objects.pyi`, none of which survived the 0.2.3 reorganisation; `implementation_summary.md` cited `objects.py` with line ranges, as if they could still be read. They are implementation records, so each now opens with a note giving the old-to-new mapping and the current home of what it describes, rather than being rewritten as though the reorganisation had never happened.

- **Dangling demo references in the developer notes** - `docs/dev/audiounit_implementation.md`, `ableton_link.md`, and `audiounit_name_lookup.md` pointed at seven demo scripts under `tests/demos/` that were deleted in `fb50106`. They are implementation records rather than guides, so the paths are now marked as removed and annotated with where the same ground is covered today, rather than being silently dropped.

- **`extras/audio_converter.py` called a function that does not exist** - It reached for `capi.convert_audio_file`; the helper lives in `coremusic.audio`. Nothing ran the utility, so nothing noticed. `tests/test_extras.py` now runs both command-line utilities in `extras/`.

- **`docs/examples/audio_inspector.md` still used the removed flat namespace** - Two blocks called `cm.AudioFile` and `cm.AudioFileError` with no import at all, which the snippet check skipped because there was no alias to resolve. The check now rejects any `cm.` use outright, since there is no flat namespace to alias.

- **Disposing a MIDI client, port, or endpoint deadlocked with packets in flight** - `midi_client_dispose()`, `midi_port_dispose()`, and `midi_endpoint_dispose()` held the GIL across the CoreMIDI call. Those calls wait for any in-flight read proc to return, and the read proc - installed on every input port and virtual destination since 0.2.4 - blocks on `with gil:` to deliver its packet. Sending to a virtual destination and then disposing therefore hung the process, permanently and without an error. It looked intermittent because it needs a packet actually in flight: disposing with nothing pending always worked, while anything that delayed delivery, such as an enabled Ableton Link session competing for CPU, made it near-certain. The three calls now release the GIL, matching what `AudioOutputUnitStop` already does for the same reason on the audio side.

- **Note on the send path** - `MIDISend` and `MIDIReceived` deliberately keep the GIL, unlike the dispose calls above. They are non-blocking hand-offs to the MIDI server, a few microseconds each, so releasing the GIL adds a reacquisition per call. Measured against a CPU-bound Python thread, doing so took p99 send latency from 0.01ms to 6.3ms and made a 1 kHz note stream drop messages. The reasoning is recorded next to the call so it is not "optimized" later.

- **`send_data()` and `connect_source()` rejected raw endpoint ids** - Both read `.object_id` off their argument, so an integer handle from `capi` raised `AttributeError` instead of working, and a wrong argument type produced that same opaque error rather than a `MIDIError`. Both now accept either a `MIDIEndpoint` or an integer, so the two APIs interoperate as documented.

### Changed

- **flake8-bugbear (`B`) adopted in full** - 153 `raise` statements inside `except` blocks discarded the original error: the message kept its text, but `__cause__` was `None`, so Python printed "During handling of the above exception, another exception occurred" instead of linking the two. Every wrapped error in the library now chains with `from err`. The ten handlers that did not bind the exception were split by intent: a missing optional dependency chains, because a *broken* install raises a different `ImportError` than an absent one, while invalid user input uses `from None`, since "invalid literal for int()" is noise next to the message replacing it. 26 unused loop variables were renamed, and two documentation examples that claimed to process a chunk while never touching it now use it.

- **Three bugbear rules adopted (`B011`, `B017`, `B905`)** - `assert False` in 17 places, which `python -O` removes: 15 were hand-rolled expect-raise blocks in `test_coremidi.py`, now `pytest.raises(RuntimeError, match="failed")`, and two wrapped whole tests in `try/except Exception` that replaced the real error with a bare assertion failure. Four `pytest.raises(Exception)` assertions passed on any error at all, including an `AttributeError` from a typo; each now names what the call actually raises. Twenty-seven `zip()` calls had no `strict=`, so a length mismatch truncated silently: the sites where the sequences must match are now `strict=True`, and the two where truncation is intended say so with a comment. All five `strict=True` sites in the library were confirmed to execute - two by the test suite, three by running the paths directly.

- **Ruff rule set pinned explicitly** - Ruff 0.16 widened its default selection from a handful of rules to roughly 920, which turned a clean tree into 955 findings without a line of code changing. `pyproject.toml` now names the rules the project lints against (`E4`, `E7`, `E9`, `F`, `I`, `W`, `UP`, `C4`, `PIE`) so a future upgrade adds rules to ruff rather than silently redefining what "lint clean" means here. The larger opinionated families are listed in a comment with their counts, to be adopted deliberately: `BLE` alone would mean narrowing 242 `except Exception` clauses in the library, a refactor `docs/dev/error_decorator.md` records as already deferred.

- **`tests/examples/` renamed to `extras/`** - Its `daw/` and `generative/` subdirectories are experimental *modules* with 309 tests, not examples, and the name collided with both `examples/` and `demos/`. Their test modules moved to `tests/` proper, where they now add `extras/` to `sys.path` rather than their own directory. `CONTRIBUTING.md` records which of the four directories new code belongs in.

- **Documentation examples are now runnable programs, and are executed by the test suite** - Every example now lives under `examples/`, mirroring the page that uses it, and doc pages include it with `pymdownx.snippets` rather than restating it. `tests/test_examples.py` runs all 249 of them in a temp directory seeded with sample media and requires a clean exit, and refuses an example that defines code it never calls; `tests/test_readme.py` does the same for the README, whose code has to stay inline because GitHub cannot process includes; `tests/test_doc_snippets.py` checks that every include resolves and that any block still written inline only names things that exist. 342 of the 373 blocks are executed - the remainder are comparisons against other libraries, MIDI constant tables, and API signature listings. A snippet that stops working now fails the build instead of reaching a reader.

- **Import Guide rewritten** - It documented version 0.1.8: an `objects.py` module, a flat top-level namespace, and "full backward compatibility" with imports that were removed in 0.2.3. It now describes the actual package layout, with the per-domain import for each public class, and the optional-dependency flags.

- **Migration Guide covers upgrading from 0.2.2 and earlier** - A table maps every `coremusic.objects` import to its domain replacement, alongside the existing guidance for moving from pydub, soundfile, mido, AudioKit, and CoreAudio C.

- **MIDI documentation corrected** - The README, `docs/index.md`, `docs/examples/index.md`, and the MIDI Basics tutorial were written against `coremusic.objects`, a package removed in 0.2.3, and against a flat `coremusic.*` namespace that has never existed. The tutorial additionally documented a two-argument receive callback, a `port.send()` method, and a `packet_list` object, none of which are real. All of them now use `coremusic.midi` and the actual send, receive, and endpoint APIs. The architecture tree in the README was updated to the current layout.

### Removed

- **`tests/demos/`** - Thirty-six scripts that nothing ran: pytest collected no tests from the directory, so seventeen of them had quietly stopped working. `daw.py` imported `coremusic.daw`, which is not a module in the package; `effects/find_by_name.py` called `capi.find_audio_unit_by_name`, which does not exist; `visualization/spectrum.py` called a plotter method that had been renamed; seven more opened `tests/amen.wav`, which moved to `tests/data/wav/` several releases ago. Most of the rest had been superseded - by CLI commands (`coremusic audio info`, `analyze levels`, `device list`, `plugin list`, `convert file`) or by the runnable snippets now under `examples/`. Two were worth keeping and moved to `demos/`.

## [0.2.4]

### Fixed

- **Segfault when receiving MIDI input ([#1](https://github.com/shakfu/coremusic/issues/1))** - `midi_input_port_create()` and `midi_destination_create()` passed `NULL` as the CoreMIDI read proc. That parameter is not optional: the framework calls it on its own receive thread as soon as a packet arrives, so the process jumped to address 0 and died with `Segmentation fault: 11`. Any `coremusic midi receive`/`midi monitor` session crashed the moment the monitored device sent anything, and sending to a virtual destination crashed the sender. Both functions now install a real read proc. By default incoming packets are buffered in a bounded queue and drained with the new `midi_input_poll()`; passing a `callback` delivers them on the CoreMIDI thread instead, with exceptions contained so they can never propagate back into the framework. Receivers are released when the owning port, endpoint, or client is disposed.

- **`MIDITimeStamp` was declared 32-bit** - The Cython declaration had it as `UInt32` where CoreMIDI defines `UInt64`, so packet timestamps were truncated and `midi_send_data()` could not accept a real host time (its `timestamp` argument was a C `int`). Both are now 64-bit. `MIDIUniqueID` was likewise corrected from `UInt32` to the signed `SInt32` the header declares.

- **`MIDIOutputPort.send_data()` always failed** - It called `capi.midi_send()`, which does not exist; every call raised `MIDIError: module 'coremusic.capi' has no attribute 'midi_send'`. The same nonexistent function was called from the `midi send`, `midi panic`, and `sequence play` CLI commands, so all three were broken. All call sites now use `midi_send_data()`.

- **`midi receive --plugin` was broken end to end** - Three independent faults, none of which surfaced as an error. No MIDI was ever forwarded to the plugin, so it had nothing to play. The render loop called `plugin.render()`, which does not exist on `AudioUnitPlugin` (the call carried a `# type: ignore[attr-defined]` and the resulting `AttributeError` was swallowed by a `logger.debug`), so no audio was produced either. Finally the output `AudioFormat` was built with no format flags and no frame sizes, so `ExtAudioFileCreateWithURL` rejected it and `-o` silently wrote nothing. Incoming channel voice messages are now forwarded, rendering uses `capi.audio_unit_render_instrument()`, and the output format is a valid packed float32 ASBD. Live playback (without `--quiet`) previously passed raw bytes to `AudioQueue.enqueue_buffer()`, which expects an allocated `AudioBuffer`, and never started the queue; it now drives the instrument from an `AudioOutputStream` generator, which is how a host normally pulls an AudioUnit. Verified against an external sender: both `--quiet -o out.wav` and the live path render audible audio.

- **System messages were displayed with a bogus channel** - `midi receive` and `midi monitor` derived a channel from the low nibble of every status byte, so system-exclusive and real-time messages were printed as belonging to a channel. They are now labelled as system messages.

- **`AudioFormat.pcm()` produced an ASBD CoreAudio rejects** - It set `flags |= 4 | 2` under a comment claiming `kAudioFormatFlagIsPacked | kAudioFormatFlagIsSignedInteger`, but in CoreAudio `2` is `kAudioFormatFlagIsBigEndian` and `IsPacked` is `8`. The factory therefore described unpacked big-endian integer PCM, and float formats got no `IsPacked` at all, so `ExtAudioFileCreateWithURL` rejected the result for WAV and CAF with `kAudioFileStreamError_UnsupportedDataFormat`. Flags are now built from the named `LinearPCMFormatFlag` constants (`12` for signed integer, `9` for float), matching the `format_flags=12` convention already used throughout the rest of the codebase. Added a `big_endian` parameter for containers that require it, such as AIFF.

- **`AudioFormat.to_numpy_dtype()` misread the same flags** - It tested flag `2` as `kAudioFormatFlagIsSignedInteger` (that bit is `IsBigEndian`; signed is `4`) *and* inverted the result, so an explicitly signed 8-bit format decoded as `uint8`. It also ignored byte order entirely, decoding big-endian streams as little-endian. Signedness is now read from the correct flag and the dtype carries the stream's byte order. 8-bit PCM is unsigned unless flagged signed, per the CoreAudio convention; at 16 bits and above an unflagged format is read as signed, since unsigned PCM does not occur at those depths and reinterpreting real samples as unsigned would corrupt them. The two 8-bit tests that asserted the inverted meaning (one commented "kAudioFormatFlagIsSignedInteger = 2 (inverted)") were corrected.

### Added

- **`MIDIMessageSplitter` and `split_midi_messages()`** (`coremusic.midi`) - A CoreMIDI packet is not one MIDI message: the framework packs several same-timestamp events into a single packet, spreads a large system-exclusive dump over consecutive packets, and may interleave real-time bytes inside another message. The splitter reassembles a packet stream into individual messages, resolving running status and holding sysex state across packets.

- **MIDI input buffering API** - `capi.midi_input_poll()`, `midi_input_wait()`, `midi_input_pending()`, and `midi_input_dropped()`, plus `MIDIInputPort.poll()`, `.wait()`, `.pending`, and `.dropped` on the object layer. `MIDIClient.create_input_port()` accepts `callback` and `queue_size`. Overflow discards the oldest packets and is reported by `dropped` rather than passing silently; the CLI warns when it happens.

- **Host time conversion helpers** - `capi.midi_host_time_to_seconds()`, `midi_seconds_to_host_time()`, and `midi_current_host_time()` for working with MIDI packet timestamps and for scheduling sends ahead of the current time.

- **`capi.midi_received()`** - Wraps `MIDIReceived`, the counterpart of `MIDISend` for virtual sources: it distributes MIDI data to everything connected to a source created with `midi_source_create()`.

- **`AudioFormat.pcm(big_endian=...)`** - Builds a big-endian PCM description for containers that require it. AIFF is integer-only and big-endian, so `midi receive --plugin -o out.aif` now converts the instrument's float32 output to big-endian int16 rather than writing a file CoreAudio refuses to create.

### Changed

- **MIDI input documentation rewritten** - The receive recipes in `docs/cookbook/midi_processing.md` were written against an API that does not exist (`midi_packet_list_get_num_packets()`, `midi_packet_list_get_packet()`, `midi_packet_get_data()`, and a three-argument `midi_input_port_create()`), so none of them ran. They now use the real polling and callback APIs, and a new section explains that a CoreMIDI packet is not a MIDI message. The timing and thread-safety notes were rewritten around host times and the receive thread, and `capi.midi_send()` was corrected to `midi_send_data()` throughout the docs.

## [0.2.3]

### Fixed

- **Real-time stream teardown could deadlock (Ctrl-C left a tone playing)** - `AudioOutputStreamImpl` / `AudioInputStreamImpl` called `AudioOutputUnitStop`, `AudioUnitUninitialize`, and `AudioComponentInstanceDispose` while holding the GIL. Those calls block until the in-flight render/capture callback returns, but that callback (`output_stream_render_callback` / `input_stream_capture_callback`) blocks on `with gil:` -- so if teardown landed while a callback was executing, the two threads deadlocked: the process wedged and the output unit kept playing. `stop()`, `close()`, and `__dealloc__` now release the GIL (`with nogil:`) around the CoreAudio stop/uninitialize/dispose calls (the three functions are declared `nogil`), so the render/capture thread can finish and teardown completes. This is what made `demos/output_stream_tone.py` occasionally unresponsive to Ctrl-C.

- **`AudioUnitPlugin.process()` / `AudioUnitChain.process()` halved interleaved audio when `num_frames` was omitted** - The auto-detection divided the byte length by `bytes_per_frame` (which already spans all channels) and then by the channel count again, so a stereo buffer processed only its first half. It now divides by channels only in the non-interleaved (per-sample) case. Existing tests always passed `num_frames` explicitly, so the auto path was untested; added regression coverage for stereo and mono.

- **`AudioFileType.M4A` held the wrong FourCC** - It was set to `1836069990` (`'mp4f'`, i.e. `kAudioFileMPEG4Type`), identical to `AudioFileType.MPEG4`, so it did not name the M4A container. Corrected to `1832149350` (`'m4af'`, `kAudioFileM4AType`); added the missing `AudioFileType.FLAC` (`'flac'`).

- **`extended_audio_file_write()` hardcoded 2 channels** - The write buffer always claimed `mNumberChannels = 2`, mislabeling mono and multichannel writes. It now takes a `num_channels` argument (defaulting to 2), and `ExtendedAudioFile.write()` supplies the client/file format's channel count.

- **`AudioAnalyzer.calculate_loudness()` was missing, breaking `analyze_loudness()`** - `coremusic.shortcuts.analyze_loudness()` called `AudioAnalyzer.calculate_loudness()`, which did not exist, raising `AttributeError`. Implemented a proper ITU-R BS.1770 / EBU R128 loudness measurement: two-stage K-weighting (high-shelf + RLB high-pass) with per-sample-rate biquad coefficients, 400 ms gated blocks with the -70 LUFS absolute and -10 LU relative gates for integrated LUFS, plus EBU Tech 3342 loudness range (LRA). Added `calculate_loudness()`, `loudness_range()`, and `measure_loudness()` to `AudioAnalyzer`. The `analyze loudness` CLI now shares this implementation instead of its inline approximation. Verified against standard properties: dual-mono is +3.01 LU louder than mono, a signal scaled to -23 LUFS round-trips, and silence measures `-inf`.

- **`convert` produced invalid or mislabeled output for every conversion** - The CLI `convert file` command built the destination `AudioFormat` with a zero `bytes_per_frame` and no PCM format flags, producing an invalid AudioStreamBasicDescription that CoreAudio rejected; even `convert a.wav b.wav` failed. Sample-rate conversion additionally used the non-resampling buffer API. In addition, `.flac`/`.mp3` output extensions silently fell through to WAV, writing mislabeled files. Now: WAV/AIFF/CAF output is produced with a complete PCM ASBD (including AIFF big-endian), and sample-rate, channel, and bit-depth conversion all work. Extensions macOS cannot encode (`.mp3`, `.ogg`, `.opus`) are rejected with a clear, actionable error instead of a broken or mislabeled file (compressed `.m4a`/`.aac`/`.flac` output is now supported -- see the AAC/ALAC/FLAC encoding entry above). `cmd_convert` and `cmd_batch` now delegate to the shared library helper so the CLI and Python API agree.

- **Library-level `convert_audio_file()` ignored the requested container** - It hardcoded WAV output regardless of the output path extension. It now resolves the file type from the extension via the shared `resolve_output_file_type()` (raising a clear error for formats coremusic cannot write) and only takes the byte-copy fast path when the input and output containers match. `shortcuts.convert()` no longer crashes on the no-argument path (it previously passed `None` as the output format) and now computes a valid PCM frame size so resampling works.

- **`get_input_devices()` / `get_output_devices()` returned all devices** - Both ignored scope and returned every device, making `shortcuts.list_devices(input_only=...)` inaccurate. They now filter by parsing the `AudioBufferList` channel count for the requested scope (`_parse_buffer_list_channels`). `AudioDevice.get_stream_configuration()` now returns a real channel count, and `channel_count()`, `has_input()`, and `has_output()` were added.

- **`plugin render` produced silent audio** - Rendering MIDI through an instrument plugin (e.g. `DLSMusicDevice`) wrote silent files because instruments output non-interleaved float32 while `capi.audio_unit_render` supplied a single interleaved buffer, so `AudioUnitRender` returned `paramErr` on every chunk. Added `capi.audio_unit_render_instrument()`, which builds the non-interleaved `AudioBufferList` the instrument expects and advances the render timeline. The `plugin render` command now produces audible output.

- **Effect processing was broken for canonical (non-interleaved) AudioUnits** - `AudioUnitPlugin.process()`, `AudioUnitChain.process()`, and the `plugin process`/`plugin chain` CLI commands failed with `paramErr` on effects whose native output is non-interleaved float32 (most of them, e.g. `AUDelay`) -- the effect-render test was previously skipped for this reason. Added `capi.audio_unit_render_effect()`, which feeds input via an input render callback, renders through the effect's native (interleaved or non-interleaved) layout, and re-interleaves the output. It renders in sub-blocks no larger than the unit's `MaximumFramesPerSlice` (advancing the input slice and timeline so time-based effects stay correct), so arbitrarily large buffers work. All effect-processing paths now produce correct audio.

- **Three AudioUnit/device stubs implemented** - `AudioDevice.sample_rate` setter now sets the nominal sample rate via `AudioObjectSetPropertyData` (raising `AudioDeviceError` if the device rejects it); `AudioUnitChain.bypass_plugin()` sets `kAudioUnitProperty_BypassEffect` so a plugin passes audio through while remaining in the chain; and `AudioUnit.render()` performs offline effect processing via `audio_unit_render_effect()`. All three previously raised `NotImplementedError`.

### Added

- **Compressed audio encoding (AAC, M4A, ALAC, FLAC) via ExtAudioFile** - `convert_audio_file()`, `shortcuts.convert()`, and the `convert`/`batch` CLI commands now write `.m4a` (AAC by default), `.aac` (AAC/ADTS), and `.flac`, in addition to the existing lossless PCM containers. Encoding decodes the source to a canonical 16-bit interleaved PCM feed (applying any sample-rate/channel change) and hands it to an ExtAudioFile whose file format is the codec and whose client format is the PCM feed, so CoreAudio runs the encoder on write. Added `AudioFormat.aac()`, `AudioFormat.alac()`, and `AudioFormat.flac()` factories plus an `AudioFormat.is_compressed` property; ALAC output is available by passing `AudioFormat.alac(...)` (or `--format alac`) into an `.m4a` container. Sample-rate conversion and mono/stereo/multichannel encoding all work. `.mp3`, `.ogg`, and `.opus` are still rejected with a clear error (macOS AudioToolbox cannot encode them).

- **AAC target bitrate control** - `convert_audio_file(..., bitrate=...)`, `shortcuts.convert(..., bitrate=...)` (bits/sec), and the `convert file`/`convert batch` CLI `--bitrate` flag (kbps) set the AAC encode bitrate by applying `kAudioConverterEncodeBitRate` to the ExtAudioFile's converter and resetting its converter config. The value is a target/average (the encoder may use fewer bits on simpler material). Only valid for AAC output; PCM and lossless (ALAC/FLAC) reject it. Added `ExtendedAudioFile.set_encode_bitrate()` and `capi` getters for the ExtAudioFile AudioConverter and ConverterConfig properties. A VBR quality knob is tracked in TODO (the mode/quality codec properties are not reachable through the direct property path).

- **Top-level `demos/` directory** - Four small, dependency-light (stdlib + `coremusic` only) runnable scripts with a README: `host_au_chain.py` (process a WAV through an AudioUnit effect chain), `render_midi_to_wav.py` (render a MIDI file through an instrument AudioUnit), `output_stream_tone.py` (real-time sine tone via a pull-generator output stream), and `link_sequencer.py` (a step sequencer locked to the Ableton Link shared beat grid, with a console-timeline fallback when no audio device is present). The `make demos` target runs all four in sequence, writing their output to `build/demos-output/`.

- **Offline MIDI-to-audio rendering API** - `AudioUnitPlugin.render_midi(events, duration, ...)` renders scheduled MIDI events through an instrument to interleaved float32 audio; `coremusic.audio.audiounit_host.render_midi_file(plugin, midi_path, output_path, ...)` renders a MIDI file to a WAV file; and `coremusic.shortcuts.render_midi(...)` provides a one-line convenience wrapper. The `plugin render` CLI now delegates to these.

- **Real-time audio streaming (capture and playback)** - `coremusic.audio.streaming.AudioOutputStream` and `AudioInputStream` replace their previous `NotImplementedError` stubs with working Cython callbacks: `capi.AudioOutputStreamImpl` installs a render callback on a Default Output unit, and `capi.AudioInputStreamImpl` configures a HAL unit for input capture. Output pulls each buffer from a Python generator (bytes or NumPy float; mono is broadcast to the channel count; generator exceptions are contained); input delivers a NumPy `(frames, channels)` float32 array (or bytes) to registered callbacks. Capture requires macOS microphone permission; when it is not granted, `AudioUnitRender` returns `kAudioUnitErr_CannotDoInCurrentContext` and `start()` raises a clear `RuntimeError` pointing to System Settings > Privacy & Security > Microphone.

- **Lock-free ring buffer and a GIL-free audio path** - `capi.AudioRingBuffer` is a single-producer/single-consumer, wait-free ring of float32 samples (acquire/release atomics on cache-line-separated cursors; no lock, no allocation; `push_floats`/`push_bytes`/`pop_into` plus `overruns`/`underruns` counters). It keeps Python and the GIL off the CoreAudio threads: capture pushes into a ring that a drain thread reads; `create_loopback()` returns a `DirectLoopback` that shares one ring between the capture and render callbacks so audio flows input-to-output entirely in C; and `AudioProcessor` (and `StreamGraph` through it) run arbitrary `process_func` effects on a worker thread via a two-ring model, so both audio threads stay GIL-free even with a Python effect. `AudioUnitRender` is now declared `nogil` so the callbacks can pull audio without the GIL. `AudioInputStream`, `DirectLoopback`, and `AudioProcessor` expose `overruns`/`underruns` for tuning ring capacity.

- **`coremusic doctor` command** - Diagnoses the installation and environment: coremusic/Python/macOS versions, optional dependency availability (numpy, scipy, matplotlib), audio hardware access (device counts and default input/output), AudioUnit plugin counts, and CoreMIDI endpoint counts. Supports `--json`.

### Changed

- **Documentation accuracy** - `docs/index.md` now states Python 3.10+ (was 3.11+), uses the real repository URL (was a `yourusername` placeholder), and shows the current CLI syntax (`midi list` / `midi monitor` / `midi panic`, not `midi devices` / `midi input monitor` / `midi output panic`); it also documents `plugin render` and `doctor`. The README documents the `doctor` command and the `render_midi` shortcut and refreshes the convert examples. The `shortcuts.convert()` docstring no longer shows unsupported `.mp3` output.

## [0.2.2]

### Changed

- **Dual license structure** - Core project remains MIT-licensed. The optional `coremusic.link` module is GPLv2-licensed as a derivative work of Ableton Link. GPL header added to `link.pyx`.

- **Documentation migrated from Sphinx to MkDocs with Material theme** - Converted all 23 reStructuredText files to Markdown. Replaced Sphinx autodoc directives with mkdocstrings. Added Material for MkDocs theme with dark/light mode toggle, navigation tabs, code copy buttons, and search. `make docs-serve` now provides live-reload preview via `mkdocs serve`.

- **Renamed `play_async` to `play_background`** - Clarifies the function uses a background thread, not Python `async/await`. The old `play_async` name is preserved as a deprecated wrapper that emits `DeprecationWarning`.

### Added

- **Audio file metadata read/write** - `AudioFile.metadata` property reads the info dictionary as a Python dict (title, artist, genre, etc.). `AudioFile.set_metadata(tags)` writes metadata to writable formats (CAF, AIFF). File must be opened with `AudioFile(path, writable=True)` for writes. Added `capi.audio_file_read_info_dictionary()`, `capi.audio_file_write_info_dictionary()`, and `capi.audio_file_set_property()` low-level functions with proper CFDictionary-to-dict conversion.

- **CLI metadata write support** - `coremusic audio metadata <file> --set key=value [...]` writes metadata tags from the command line. Supports multiple key=value pairs in a single invocation. Works with `--json` for structured output.

- **`plugin chain` command** - `coremusic plugin chain <file> -p "AUDelay" -p "AUReverb2" -o out.wav` processes audio through multiple effect plugins sequentially. Supports per-plugin presets via `--preset`, inline parameters via colon syntax (`-p "AUDelay:Delay Time=0.5:Wet Dry Mix=50"`), and JSON/YAML config files via `--config`. Parameters are resolved by name (case-insensitive partial match).

- **`midi monitor` command** - `coremusic midi monitor` displays incoming MIDI with human-readable formatting: note names (C4, F#3), CC names (Sustain Pedal, Modulation), centered pitch bend values, and 1-indexed channels. Supports `--device` to select input source and `--json` for structured output.

- **`device monitor` command** - `coremusic device monitor` polls for audio device state changes: connect/disconnect, sample rate, volume, mute, and default device changes. Configurable poll interval via `--interval`. Supports `--json` for structured output.

### Removed

- Sphinx configuration (`docs/conf.py`, `docs/Makefile`) and build artifacts (`docs/_build/`)

- `sphinx>=7.0` dev dependency, replaced by `mkdocs>=1.6`, `mkdocs-material>=9.5`, `mkdocstrings[python]>=0.25`

- `docs-pdf` and `docs-linkcheck` Makefile targets (Sphinx-specific)

## [0.2.1]

### Fixed

- **MusicDevice segfault in `music_device_start_note`** - Two fixes for a segfault in `MusicDeviceStartNote` that crashed CI on all macOS/Python versions:

  1. Fixed `size_t` underflow in `capi.pyx` heap allocation for `MusicDeviceNoteParams` when no controls are provided. `(num_controls - 1)` with `num_controls=0` produced `-1`, which wrapped to `SIZE_MAX` when promoted to unsigned `size_t`, causing `malloc` to return an undersized buffer.

  2. Fixed `music_device_unit` test fixture to call `audio_unit_initialize()` before use. `MusicDeviceStartNote` segfaults on uninitialized audio units (unlike `MusicDeviceMIDIEvent`/`MusicDeviceSysEx` which return errors gracefully). Fixture now properly initializes/uninitializes the unit and skips if initialization fails.

- **CI timing test flakiness** - Fixed six tests that assumed precise `time.sleep()` behavior or deterministic Link state propagation, which fails on loaded CI runners:

  - `test_clock_advances_at_normal_speed`, `test_clock_advances_at_half_speed`: Replaced absolute clock-time assertions with clock-to-wall-time ratio checks, decoupling correctness from sleep precision.

  - `test_link_clock_precision`: Widened upper bound from 5ms to 50ms to accommodate CI scheduler jitter.

  - `test_link_timing_updates`: Replaced absolute beat-delta assertion with wall-time-proportional check.

  - `test_context_manager_with_operations`: Added retry loop for Link tempo propagation which can lag on CI runners.

  - `test_beat_tracking_pattern`: Replaced `beat >= 0.0` assertion with monotonicity check; `beat_at_time` can return negative values before the session timeline origin.

- **CoreMIDI segfault in `test_full_midi_workflow`** - Removed `midi_send_data` call to a virtual destination created with a NULL read proc callback. `MIDIDestinationCreate` is called with `NULL` as the read proc in `midi_destination_create`, so `MIDISend` to that endpoint invokes a NULL function pointer, causing a segfault on CI.

- **CI smoke test import path** - Updated `ci.yml` smoke test to import `audio_file_open_url` from `coremusic.capi` instead of the top-level `coremusic` namespace, which was cleaned up during the v0.2.0 restructure.

### Added

- **Python 3.10 support** - Lowered `requires-python` from `>=3.11` to `>=3.10`. No 3.11+ language features were in use (all `X | Y` annotations are guarded by `from __future__ import annotations`). Dev dependency floors lowered to 3.10-compatible versions (numpy>=2.0, scipy>=1.10, sphinx>=7.0, etc.); on 3.11+ uv still resolves the latest. Added 3.10 to CI test matrix.

## [0.2.0]

### Fixed

- **Silent exception logging** - ~15 bare `except: pass` blocks across `os_status.py`, `shortcuts.py`, `audio/devices.py`, `audio/units.py`, `cli/plugins.py`, `cli/audio.py`, `cli/midi.py` now log with `logger.debug()` instead of silently swallowing errors

- **CLI CoreAudioError handling** - Top-level exception handler in `cli/main.py` now catches `CoreAudioError` and prints user-friendly messages with `os_status.get_error_suggestion()` hints

- **`capi.pyi` type stubs** - Added ~40 missing function stubs (clock, device properties, buffer processing) and fixed signatures for `music_track_new_midi_note_event`, `music_track_new_midi_channel_event`, `music_sequence_file_load`; eliminated ~47 `type: ignore[attr-defined]` comments

- **CLI version fallback** - `cli/main.py` now falls back to `coremusic.__version__` instead of a hardcoded `"0.1.12"` string

- **`AudioPlayerHandle` typing** - `_player` parameter typed as `AudioPlayer` instead of `Any`

- **Coverage regression gate** - Added `fail_under = 60` to `[tool.coverage.report]` in pyproject.toml

- **`os_status.py` precedence documented** - Documented `ALL_ERRORS` dict merge order and all overlapping FourCC keys with their final resolution

### Added

- **`AudioFormat.from_asbd_bytes()`** - Classmethod to parse 40-byte AudioStreamBasicDescription into `AudioFormat`, replacing 3 duplicated `struct.unpack` call sites

- **`AudioFormat.pcm()` factory** - Convenience factory that computes derived ASBD fields (`bytes_per_frame`, `bytes_per_packet`, `format_flags`) from sample rate, channels, bits, and float/int selection

- **Hardware detection test fixtures** - `has_audio_output` and `has_audio_input` pytest markers in `conftest.py` for skipping hardware-dependent tests in CI

- **Exception subclass test coverage** - Added tests for `AudioDeviceError` and `AUGraphError` in `test_objects_base.py`

- **Batch Convert Progress Bar** - Added progress indicator for `coremusic convert batch`

  - Shows ASCII progress bar with file count and percentage during conversion

  - Errors collected and displayed at end instead of inline

  - Cleaner output: `[==============================] 50/50 (100%)`

- **Rhythm and Meter Module** - Added rhythm/meter representation to music theory

  - `NoteValue` enum: whole, half, quarter, eighth, sixteenth with dotted/triplet support

  - `TimeSignature` class: meter classification, beat/measure calculations, grid quantization

  - `Duration` class: note values with dots and tuplet modifiers

  - `RhythmPattern` class: rhythmic sequences with onset positions and tempo scaling

  - `COMMON_PATTERNS`: four_on_floor, eighth_notes, sixteenth_notes presets

  - Enhances MIDI quantization, tempo analysis, and sequence generation

- **Shell Completion** - Added `coremusic completion` command for shell autocompletion

  - Supports bash, zsh, and fish shells

  - Usage: `eval "$(coremusic completion bash)"` (add to shell rc file)

  - Completes commands, subcommands, and audio file extensions

- **Convenience Functions** - Added simple one-liner functions for common operations

  - `cm.play("audio.wav")` - Quick audio playback (blocking or async)

  - `cm.play_async("audio.wav")` - Non-blocking playback with control handle

  - `cm.convert("input.wav", "output.mp3")` - Quick format conversion

  - `cm.analyze_tempo("audio.wav")` - Get BPM

  - `cm.analyze_key("audio.wav")` - Get musical key

  - `cm.analyze_loudness("audio.wav")` - Get LUFS loudness

  - `cm.get_duration("audio.wav")` - Get duration in seconds

  - `cm.get_info("audio.wav")` - Get file metadata as dict

  - `cm.list_devices()` - List audio devices

  - `cm.list_plugins()` - List AudioUnit plugins

- **Improved REPL Experience** - Added `__repr__` to all major classes

  - Better introspection for AudioFile, AudioFormat, AudioQueue, MIDIClient, AudioUnit, etc.

- **Developer Documentation** - Added contributor guidelines and tooling

  - `CONTRIBUTING.md` - Build prerequisites, development setup, code style, testing guidelines

  - `.pre-commit-config.yaml` - Pre-commit hooks for ruff, mypy, and file checks

- **Optional Dependencies** - Added extras for optional features in pyproject.toml

  - `coremusic[analysis]` - NumPy and SciPy for audio analysis

  - `coremusic[visualization]` - Matplotlib for waveform/spectrogram plots

  - `coremusic[all]` - All optional dependencies

  - Documented `NUMPY_AVAILABLE` and `SCIPY_AVAILABLE` flags in README

### Changed

- **PEP 604/585 typing modernization** - All source files use modern `X | None`, `list[X]`, `dict[K, V]` syntax instead of `Optional`, `List`, `Dict` from `typing`

- **`constants.py` split into subpackage** - Split monolithic `constants.py` (489 lines) into `constants/audio.py`, `constants/audiounit.py`, `constants/queue.py`, `constants/device.py`, `constants/midi.py` with backward-compatible re-exports

- **Domain-oriented import paths (BREAKING)** - Dissolved `coremusic.objects` into domain subpackages

  - `coremusic.audio` - Audio file, format, queue, unit, graph, device, and clock classes

  - `coremusic.midi` - MIDI client/port classes and music player/sequence/track classes

  - `coremusic.exceptions` - Exception hierarchy (`CoreAudioError`, `AudioFileError`, etc.)

  - `coremusic.base` - Base classes (`CoreAudioObject`, `AudioPlayer`, `NUMPY_AVAILABLE`)

  - Example: `from coremusic.audio import AudioFile, AudioFormat`

  - Example: `from coremusic.midi import MIDIClient, MusicPlayer`

  - Example: `from coremusic.exceptions import AudioFileError`

- **MIDI CLI Restructured** - Flattened command hierarchy from 3 levels to 2 levels

  - `midi list` - Combined device/input/output listing (was `midi device list`, `midi input list`, `midi output list`)

  - `midi info <file>` - File metadata with `--events` flag for event table (was `midi file info`, `midi file dump`)

  - `midi play <file>` - Play MIDI file (was `midi file play`)

  - `midi quantize <file>` - Quantize timing (was `midi file quantize`)

  - `midi receive` - Unified MIDI input command (was `midi input monitor`, `midi input record`)

    - Display mode: `midi receive` (shows incoming MIDI)

    - Record mode: `midi receive -o file.mid` (saves to MIDI file)

    - Plugin mode: `midi receive --plugin "DLSMusicDevice"` (routes to AudioUnit, plays audio)

    - Capture mode: `midi receive --plugin X -o file.wav` (routes to plugin, saves audio)

    - Quiet mode: `midi receive --plugin X -o file.wav --quiet` (saves audio without playback)

  - `midi send` - Send MIDI with `--test` flag (was `midi output send`, `midi output test`)

  - `midi panic` - All notes off (was `midi output panic`)

  - Consistent with rest of CLI structure

- **CI/CD Enabled** - GitHub Actions workflow now triggers on push/PR to main and develop branches

  - Previously was manual dispatch only (`workflow_dispatch`)

  - Runs tests on Python 3.11, 3.12, 3.13, 3.14

- **Dynamic Version** - CLI version now uses `importlib.metadata` instead of hardcoded string

  - Fixes version mismatch between CLI and package metadata

  - Falls back gracefully during development

### Removed

- **`coremusic.objects` package** - Removed entirely (no backwards-compat shim)

  - All imports from `coremusic.objects` must be updated to the new domain paths above

  - `objects.pyi` type stub also removed

## [0.1.12]

### Changed

- **Build System Migration** - Converted from setuptools to scikit-build-core with CMake

  - Replaced `setup.py` with `CMakeLists.txt` for building Cython extensions

  - `capi` module: C extension with CoreAudio/CoreMIDI/AudioToolbox frameworks

  - `link` module: C++11 extension with Ableton Link headers

  - Cython directives (`language_level=3`, `embedsignature=True`) passed via CMake

  - macOS-only build configuration (removed Windows/Linux conditionals)

- **Makefile Improvements** - Consolidated and enhanced build targets

  - `test` now excludes slow tests by default (`-m "not slow"`)

  - `test-all` runs complete test suite including slow tests

  - `release` builds wheels for Python 3.11-3.14

  - `typecheck` now checks full `src/coremusic` directory

  - Added `isort`, `docs-clean`, `docs-serve`, `docs-pdf`, `docs-linkcheck` targets

  - Fixed `clean` to not delete `.so` files from `.venv`

  - Organized `help` output into categories

### Added

- **Comprehensive Documentation** - New tutorials, quick-start guide, and doctest examples

  - `docs/quickstart.rst` - 5-minute getting started guide

  - `docs/tutorials/audio_playback.rst` - Audio playback from simple to advanced

  - `docs/tutorials/audio_recording.rst` - Recording from input devices

  - `docs/tutorials/effects_processing.rst` - AudioUnit effects processing

  - `docs/tutorials/midi_basics.rst` - MIDI fundamentals and message handling

- **Doctest Tutorial Suite** - 84 executable doctests in `tests/tutorials/`

  - `test_audio_file_basics.py` - Audio file operations (12 tests)

  - `test_midi_basics.py` - MIDI device and message handling (13 tests)

  - `test_effects_processing.py` - AudioUnit effects (17 tests)

  - `test_music_theory.py` - Notes, scales, chords, intervals (18 tests)

  - `test_quickstart.py` - Quick reference examples (24 tests)

### Fixed

- **Test Collection Error** - Added `from __future__ import annotations` to `tests/test_audio_slicing.py` to fix `NameError` when numpy type hints were evaluated at class definition time

## [0.1.11]

### Changed

- **Audio Player Test Duration** - Reduced `test_audio_playback` from 10 seconds to 2 seconds

  - Improves test suite execution time while maintaining coverage

- **Render Callback Performance Optimization** - Optimized `audio_player_render_callback()` in `capi.pyx`

  - Replaced frame-by-frame loop with block `memcpy()` for entire buffer transfer

  - Critical path optimization: callback is invoked 44,100+ times/second at 44.1kHz

  - Expected 2-5x performance improvement for real-time audio playback

### Added

- Essential MIDI commands for debugging and device testing

  - `midi output panic` - Sends all-notes-off (CC 123) and all-sound-off (CC 120) on all 16 channels to stop stuck notes

  - `midi output test` - Sends test note (middle C, note 60) to verify MIDI connectivity

  - `midi device info <name>` - Shows detailed MIDI device info including entities, sources, and destinations

  - `midi file dump <path>` - Hex dump of raw MIDI events with time, track, channel, type, and data

- **Audio playback, MIDI monitoring, and plugin processing

  - `audio play <path>` - Plays audio file with progress bar, supports `--loop` option

  - `midi input monitor [index]` - Real-time MIDI input monitoring with message formatting (Ctrl+C to stop)

  - `midi file play <path>` - Plays MIDI file to output device with precise timing

  - `plugin process <name> <audio> -o <output>` - Applies effect plugin to audio file

  - `plugin render <name> <midi> -o <output>` - Renders MIDI through instrument plugin to audio file

- Audio manipulation, analysis, and device control

  - `convert normalize <input> <output>` - Normalizes audio to target peak or RMS level with `--target` and `--mode` options

  - `convert trim <input> <output>` - Extracts portion of audio file with `--start`, `--end`, or `--duration` options

  - `midi input record [index] -o <file>` - Records MIDI input to file with optional `--duration` and `--tempo`

  - `analyze loudness <path>` - LUFS loudness measurement with integrated LUFS, loudness range, peak, and RMS

  - `analyze onsets <path>` - Onset detection using spectral flux with configurable `--threshold` and `--min-gap`

  - `device volume <name> [level]` - Get or set device volume (0.0-1.0) with `--scope` and `--channel` options

  - `device set-default <name>` - Set default audio device with `--input` or `--output` flags

  - `device mute <name> [on|off]` - Get or set device mute state

- New low-level functions for modifying audio hardware properties (`capi.pyx`)

  - `audio_object_set_property_data()` - Sets property data on an AudioObject (volume, mute, default device)

  - `audio_object_is_property_settable()` - Checks if a property can be modified

  - `get_audio_device_property_volume_scalar()` - Volume property selector constant

  - `get_audio_device_property_mute()` - Mute property selector constant

  - `get_audio_hardware_property_default_output_device()` - Default output device property selector

  - `get_audio_hardware_property_default_input_device()` - Default input device property selector

- New methods in `AudioDevice` class (`objects.py`)

  - `get_volume(scope, channel)` - Get volume level (0.0-1.0) for a specific scope and channel

  - `set_volume(level, scope, channel)` - Set volume level with validation and settability check

  - `get_mute(scope, channel)` - Get mute state (True/False) for a specific scope and channel

  - `set_mute(muted, scope, channel)` - Set mute state with settability check

- **AudioDeviceManager Default Device Control** - New static methods (`objects.py`)

  - `set_default_output_device(device)` - Set the system default output device

  - `set_default_input_device(device)` - Set the system default input device

- Audio recording, plugin presets, and MIDI quantization

  - `audio record -o <path>` - Records audio from input device with progress bar, `--duration`, `--sample-rate`, `--channels`

  - `plugin preset list <name>` - Lists factory presets for a plugin with index numbers

  - `plugin process --preset <name|number>` - Added preset selection to effect processing

  - `plugin render --preset <name|number>` - Added preset selection to instrument rendering

  - `midi file quantize <path> -o <output>` - Quantizes MIDI note timing to grid with `--grid` (1/4, 1/8, 1/16, etc.) and `--strength` (0.0-1.0)

- **AudioRecorder Class** - New Cython class for audio input recording (`capi.pyx`)

  - Uses AudioQueue API for input capture from default device

  - Supports configurable sample rate and channel count

  - Saves recordings to WAV format via AudioFileCreateWithURL

- **Zero-Copy Audio Functions** - High-performance memoryview variants for buffer operations (`capi.pyx`)

  - `audio_file_read_packets_into()`: Reads audio packets directly into caller-provided buffer

    - Accepts bytearray or numpy array as buffer

    - Eliminates malloc/copy/free overhead in audio file read path

    - Enables buffer reuse across multiple sequential reads

  - `audio_converter_convert_buffer_into()`: Converts audio data directly into output buffer

    - Accepts memoryview input (bytes, bytearray, numpy uint8 array)

    - Writes converted data to caller-provided output buffer

    - Enables zero-allocation chained conversions in streaming scenarios

  - `extended_audio_file_read_into()`: Reads ExtAudioFile frames directly into buffer

    - Zero-copy reading for high-level audio file API

    - Configurable channel count

  - `audio_unit_render_into()`: Processes audio through AudioUnit with zero-copy I/O

    - Accepts memoryview input, writes to caller-provided output

    - Buffer size validation for both input and output

  - `audio_converter_fill_complex_buffer_into()`: Complex conversion (sample rate, codec) with zero-copy

    - Near-zero allocation for streaming conversions

    - Supports sample rate conversion, codec conversion, channel changes

  - `audio_file_stream_parse_buffer()`: Parses streaming audio data with zero-copy

    - Accepts bytes, bytearray, numpy arrays, memoryviews

    - Enables buffer pool reuse for network streaming scenarios

  - Original functions preserved for backward compatibility

  - 26 new tests in `tests/test_memoryview_optimizations.py`

- **Integration Tests** - Comprehensive end-to-end workflow tests (`tests/test_integration_workflows.py`)

  - `TestAudioProcessSaveWorkflow` (3 tests): Audio file read, effect discovery/configuration, format conversion, multi-effect chains, file output

  - `TestMIDIToAudioUnitWorkflow` (4 tests): MIDI file loading, MusicSequence creation, MusicPlayer playback, DLSMusicDevice instrument control

  - `TestLinkSessionSynchronization` (7 tests): Session sync, tempo sync, transport sync, beat quantization, AudioPlayer integration, multi-player sync, clock precision

  - `TestCombinedWorkflows` (2 tests): MIDI-to-audio file workflow, Link-synchronized MIDI playback

  - Total: 16 new integration tests covering all major subsystem interactions

### Changed

- **Refactored Repeated Patterns** - Created helper functions to reduce code duplication

  - `init_property_address()` in `capi.pyx:173` - Initializes AudioObjectPropertyAddress structs consistently

  - `dict_to_asbd()` and `asbd_to_dict()` in `capi.pyx` - Converts between dict and AudioStreamBasicDescription struct

  - `parse_audio_stream_basic_description()` in `audio/utilities.py` - Parses ASBD bytes to dictionary

  - `cfstring_to_str()` in `capi.pyx` - Converts CFStringRef to Python string

- **Eliminated Magic Numbers** - Replaced hardcoded values with named constants

  - Property address scope/element now use `ca.kAudioObjectPropertyScopeGlobal` and `ca.kAudioObjectPropertyElementMain` instead of `0, 0`

  - Buffer sizes now use `STRING_BUFFER_SIZE` constant (defined as 256) instead of hardcoded values

### Improved

- **Parameter Validation** - Added comprehensive input validation to `objects.py` methods

  - `convert_with_callback()`: Validates input_packet_count, output_packet_count, input_data type

  - `get_track()`: Validates index range with descriptive IndexError messages

  - `get_node_at_index()`: Validates index range with descriptive IndexError messages

  - `set_stream_format()`: Validates format type and element index

  - `MusicPlayer.time` setter: Validates non-negative time

  - All validation raises appropriate exception types (ValueError, TypeError, IndexError)

- **Documentation Examples** - Added usage examples to key method docstrings

  - `read_packets()`: Example showing chunked audio reading

  - `get_stream_format()`: Example showing format inspection

  - `set_stream_format()`: Example showing format configuration

  - `get_track()`: Example showing track iteration

  - `add_node()`: Example showing graph construction with effect chain

- **Declaration File Documentation** - Added deprecation notes and block type documentation to `.pxd` files

  - Deprecated API notes with replacement recommendations:

    - `AudioConverterFillBuffer` -> `AudioConverterFillComplexBuffer`

    - `AudioServicesPlayAlertSound/PlaySystemSound` -> completion block versions

    - `MIDIDeviceAddEntity` -> `MIDIDeviceNewEntity`

    - `kAudioObjectPropertyElementMaster` -> `kAudioObjectPropertyElementMain`

  - Block-based API documentation explaining Cython limitations:

    - `audiotoolbox.pxd`: AudioServices completion block APIs

    - `coremidi.pxd`: MIDINotifyBlock, MIDIReceiveBlock types

    - `coreaudio.pxd`: Property listener and IO proc block APIs

- **Test Quality Cleanup** - Removed redundant and low-value tests, consolidated existence checks

  - Removed 17 constant-verification tests (testing hardcoded values against themselves)

  - Consolidated 9 existence-check tests into 2 parameterized tests (18 classes, 28 functions)

  - Strengthened 8 weak property tests (use `pytest.skip()` instead of silent `if` blocks)

  - Final test count: 1586 passed, 78 skipped, 36 deselected

- **Documentation Examples** - Fixed broken imports in `tests/examples/` test files

  - Updated imports from removed modules (`coremusic.daw`, `coremusic.music.generative`, etc.) to local example modules

  - Fixed `docs/tutorials/music_theory.rst` to remove references to moved modules

### Removed

- **Neural Module** - Removed `coremusic.music.neural` subpackage

  - Out of scope for the core package

  - Included: `api.py`, `data.py`, `evaluation.py`, `generation.py`, `models.py`, `training.py`

  - Associated tests removed: `test_music_neural.py`, `test_neural_training_classical.py`

  - Demo scripts removed: `tests/demos/neural/`

  - Keeping the package lean and fit-for-purpose

- **Generative Algorithms** - Moved from package to examples

  - `coremusic.music.generative` module removed from package

  - `coremusic.music.markov` module removed from package

  - `coremusic.music.bayes` module removed from package

  - Code relocated to `tests/examples/` directory for reference

  - Associated tests removed: `test_music_generative.py`, `test_music_markov.py`, `test_music_bayes.py`

- **DAW Integration** - Removed `coremusic.daw` module

  - Moved to `tests/examples/` directory

  - Associated tests removed: `test_daw.py`

- **CLI Commands** - Removed generative CLI commands

  - `coremusic generate` command removed

  - `coremusic neural` command removed

### Fixed

- **MIDI Test Suite Reliability** - Fixed MIDI tests being skipped when running full test suite

  - Root cause: Long-running audio playback tests (CoreAudio) interfere with CoreMIDI services on macOS

  - Solution: Added pytest hook in `conftest.py` to run MIDI tests first, before audio playback tests

  - Result: 41 additional tests now pass (1689 passed vs 1648 previously)

## [0.1.10]

### Changed

- **Improved Code Quality** - Comprehensive lint and type checking fixes

  - Fixed all ruff/flake8 lint errors without using noqa suppressions

  - Converted star imports (`from x import *`) to explicit imports in `__init__.py` files

  - Fixed E741 ambiguous variable names (`l` -> `lyr`)

  - Fixed E721 type comparisons (use `is` instead of `==` for type checks)

  - Fixed F841 unused variable assignments

  - All modules now pass `make lint` and `make typecheck`

### Removed

- **Test Suite Cleanup** - Removed 4 low-quality/redundant test files (20 tests)

  - `test_fourcharcode.py` - Tested Python builtins, not coremusic functionality

  - `test_coverage_improvements.py` - Overly broad exception tests with no specific assertions

  - `test_audiotoolbox_audio_queue.py` - Trivial assertions (only checked `is not None`)

  - `test_audiounit.py` - No behavior verification, just type checks

  - Test count: 1699 -> 1679 (all remaining tests pass)

### Added

- **Command Line Interface** - Comprehensive CLI for audio and MIDI operations (`coremusic.cli`)

  - `coremusic audio` - Audio file operations (info, duration, metadata)

  - `coremusic devices` - Audio device management (list, default, info)

  - `coremusic plugins` - AudioUnit plugin discovery (list, find, info, params)

  - `coremusic analyze` - Audio analysis (peak, rms, silence, tempo, spectrum, key, mfcc)

  - `coremusic convert` - Format conversion (file, batch)

  - `coremusic midi` - MIDI device discovery (devices, inputs, outputs, send, file)

  - `coremusic generate` - Generative algorithms (arpeggio, euclidean, melody)

    - Transform support: `--transform`/`-t` to apply transforms (humanize, reverse, arpeggiate, quantize, velocity_scale)

    - Tempo option: `--bpm`/`-b` (renamed from `--tempo`/`-t`)

  - `coremusic sequence` - MIDI sequence operations (info, play, tracks)

  - JSON output support (`--json`) for scripting integration

- **Bit Shift Register Generator** - Gate-based sequencing with variable velocity and duration (`coremusic.music.generative`)

  - `BitShiftRegister` - Core 8-bit shift register with configurable feedback taps

    - Left/right shift operations with XOR feedback

    - Rotate operations preserving bit state

    - Clock-based operation with gate output

    - Seed configuration for reproducible sequences

  - `BitShiftRegisterConfig` - Configuration for the generator

    - Variable velocity modes: fixed, random, pattern-based

    - Variable duration modes: fixed, random, pattern-based

    - Configurable note, channel, and base parameters

  - `BitShiftRegisterGenerator` - Full MIDI event generator

    - Step-based generation with clock advancement

    - MIDI file output integration

    - Humanization and swing support

  - 47 tests covering all functionality

- **Bayesian Network MIDI Analysis** - Probabilistic modeling of note dependencies (`coremusic.music.bayes`)

  - `BayesianNetwork` - Core Bayesian network implementation

    - Configurable network structure (fixed, learned, or manual)

    - Directed acyclic graph with cycle detection

    - Conditional probability tables (CPT) with Laplace smoothing

    - Topological sampling (ancestral sampling)

  - `NetworkConfig` - Comprehensive configuration dataclass

    - Network modes: pitch-only, pitch+duration, pitch+duration+velocity, full (with IOI)

    - Structure modes: fixed, learned, manual

    - Configurable temporal order (1st, 2nd, higher)

    - Discretization bins for each variable

  - **Network Structure**

    - `add_variable()`, `add_edge()` - Build custom network structures

    - `remove_variable()`, `remove_edge()` - Modify structures

    - Automatic cycle detection

  - **Conditional Probability Tables**

    - `CPT` class for storing P(variable | parents)

    - Laplace smoothing for unseen observations

    - Sampling and entropy calculation

  - `MIDIBayesAnalyzer` - MIDI file analysis

    - `analyze_file()` - Create network from MIDI file

    - `analyze_track()` - Analyze specific track

    - `analyze_all_tracks()` - Create networks for all tracks

  - `MIDIBayesGenerator` - MIDI generation from networks

    - `generate()` - Generate new MIDI sequence

    - `generate_to_track()` - Add generated notes to existing track

    - Start pitch control for deterministic beginnings

  - **Utility Functions**

    - `analyze_and_generate()` - One-step analysis and generation

    - `merge_networks()` - Combine multiple networks

    - `network_statistics()` - Get network metrics

  - 66 comprehensive tests covering all functionality

- **Markov Chain MIDI Analysis** - Advanced MIDI file analysis and generation using Markov chains (`coremusic.music.markov`)

  - `MarkovChain` - Core Markov chain implementation

    - Configurable order (1st, 2nd, higher-order chains)

    - Training from note sequences

    - Probability matrix with transition sampling

    - JSON serialization for saving/loading models

  - `ChainConfig` - Comprehensive configuration dataclass

    - Modeling modes: pitch-only, pitch+duration, pitch+duration+velocity

    - Rhythm modes: constant, Markov-based, external generator

    - Temperature control for sampling randomness

    - Note range clamping (min/max MIDI notes)

    - Gravity weights for biasing toward specific notes

    - Probability smoothing (Laplace smoothing)

  - **Node-Edge Editing** - Granular transition manipulation

    - `set_transition_probability()` - Set specific transition weights

    - `remove_transition()` - Remove specific transitions

    - `get_transition_probability()` - Query transition weights

    - `get_transitions_from()` - Get all transitions from a state

  - **Chain-Scope Adjustments** - Global chain modifications

    - `set_temperature()` - Control sampling randomness

    - `set_note_range()` - Clamp output to MIDI range

    - `set_gravity()` - Bias toward specific notes

    - `sparsify()` - Remove low-probability transitions

  - `MIDIMarkovAnalyzer` - MIDI file analysis

    - `analyze_file()` - Create chain from MIDI file

    - `analyze_track()` - Analyze specific track

    - `analyze_all_tracks()` - Create chains for all tracks

  - `MIDIMarkovGenerator` - MIDI generation from chains

    - `generate()` - Generate new MIDI sequence

    - `generate_to_track()` - Add generated notes to existing track

    - Start pitch control for deterministic beginnings

  - **Utility Functions**

    - `analyze_and_generate()` - One-step analysis and generation

    - `merge_chains()` - Combine multiple chains

    - `chain_statistics()` - Get chain metrics (states, transitions, entropy)

  - 64 comprehensive tests covering all functionality

- **Music Theory and Generative Module** - Complete music theory foundations and MIDI-enabled generative algorithms (`coremusic.music`)

  - **Music Theory Primitives** (`src/coremusic/music/theory.py`)

    - `Note` class with MIDI number conversion, transposition, frequency calculation (A4=440Hz)

    - `Interval` class with standard intervals (unison through compound intervals)

    - `Scale` class with 25+ scale types:

      - Diatonic: major, natural/harmonic/melodic minor

      - Modes: dorian, phrygian, lydian, mixolydian, locrian

      - Pentatonic: major, minor, blues major/minor

      - Jazz: bebop major/dominant/minor, whole tone, diminished

      - World: harmonic major, double harmonic, hungarian minor, neapolitan

      - Exotic: hirajoshi, in-sen, iwato, pelog

    - `Chord` class with 35+ chord types:

      - Triads: major, minor, diminished, augmented, sus2, sus4

      - 7ths: dominant7, major7, minor7, diminished7, half-diminished7, minorMajor7

      - Extended: 9th, 11th, 13th variants

      - Altered: 7b5, 7#5, 7b9, 7#9, 7#11

      - Added tone: add9, add11, 6, minor6

    - `ChordProgression` class with Roman numeral parsing (I, ii, IV, V7, etc.)

    - Enharmonic note handling (flats normalized to sharps internally)

  - **Generative Algorithms** (`src/coremusic/music/generative.py`)

    - `Arpeggiator` with 10 pattern modes:

      - UP, DOWN, UP_DOWN, DOWN_UP, RANDOM, RANDOM_WALK

      - OUTSIDE_IN, INSIDE_OUT, CHORD, AS_PLAYED

    - `EuclideanGenerator` using Bjorklund's algorithm for mathematical rhythm patterns

      - Classic patterns: tresillo (3,8), cinquillo (5,8), rumba (7,12)

    - `MarkovGenerator` for probabilistic note sequences

      - Training from note sequences

      - Scale constraint for harmonic coherence

    - `ProbabilisticGenerator` for weighted random note selection

      - Custom note weights for biased selection

      - Rest probability for rhythmic variety

    - `SequenceGenerator` (step sequencer)

      - Per-step note, velocity, gate length

      - Step probability for variation

    - `MelodyGenerator` for rule-based melodic phrases

      - Scale-constrained motion with configurable step size

      - Rest insertion and phrase structure

    - `PolyrhythmGenerator` for layered polyrhythmic patterns

      - Multiple independent rhythmic layers

      - Cross-rhythm pattern generation

    - Common features across generators:

      - Swing timing (0.0-1.0)

      - Humanization for timing and velocity

      - Reproducible results via seed parameter

  - **MIDI Integration**

    - All generators output `MIDIEvent` objects compatible with `coremusic.midi.utilities`

    - Direct integration with `MIDITrack` and `MIDISequence` for file export

    - Utility functions: `create_arp_from_progression()`, `combine_generators()`

  - **Test Coverage**

    - 80 tests in `tests/test_music_theory.py` (Note, Interval, Scale, Chord, ChordProgression)

    - 79 tests in `tests/test_music_generative.py` (all generators + MIDI file generation)

    - 19 MIDI file generation tests creating actual `.mid` files in `build/midi_files/`

  - **Generated MIDI Files** (`build/midi_files/`)

    - Arpeggiator demos: up, up-down, random patterns

    - Euclidean rhythms: tresillo (3,8), cinquillo (5,8), 7/12

    - Markov melodies: trained and pentatonic-constrained

    - Probabilistic: Dorian scale, weighted selection

    - Sequences: drum patterns, melodic sequences

    - Melodies: major scale, blues scale

    - Polyrhythms: 3:4, 5:4:3

    - Combined: arp+drums, chord progression, full composition

  **Example Usage:**

  ```python
  import coremusic as cm
  from coremusic.music.theory import Note, Scale, ScaleType, Chord, ChordType
  from coremusic.music.generative import Arpeggiator, ArpPattern, EuclideanGenerator

  # Music Theory
  c4 = Note.from_name("C4")
  print(f"MIDI: {c4.midi}, Frequency: {c4.frequency:.2f} Hz")

  c_major = Scale(Note.from_name("C4"), ScaleType.MAJOR)
  print(f"C Major: {[str(n) for n in c_major.notes]}")

  cm_chord = Chord(Note.from_name("C4"), ChordType.MAJOR_7)
  print(f"Cmaj7: {[str(n) for n in cm_chord.notes]}")

  # Arpeggiator
  chord = Chord(Note.from_name("C4"), ChordType.MAJOR)
  arp = Arpeggiator(chord.notes, pattern=ArpPattern.UP_DOWN, note_duration=0.25)
  events = arp.generate(num_notes=8)

  # Euclidean Rhythms
  euclidean = EuclideanGenerator(
      pulses=5, steps=8,  # Cinquillo pattern
      note=Note.from_name("C4"),
      step_duration=0.125
  )
  events = euclidean.generate(num_cycles=2)

  # Export to MIDI file
  from coremusic.midi.utilities import MIDISequence, MIDITrack
  sequence = MIDISequence()
  track = sequence.create_track("Arpeggio")
  for event in events:
      if event.status == MIDIStatus.NOTE_ON:
          track.add_note(event.time, event.data1, event.data2, 0.2)
  sequence.save("arpeggio.mid")
  ```

- **matplotlib as dev dependency** - Enables audio visualization tests (`tests/test_audio_visualization.py`)

- **MIDI Transformation Pipeline** - Composable pipeline for analyzing and transforming MIDI files (`coremusic.midi.transform`)

  - **Base Classes**

    - `MIDITransformer` - Abstract base class for all transformers

    - `Pipeline` - Chain of transformers applied in sequence with fluent API

  - **Pitch Transformers**

    - `Transpose` - Shift notes by semitones (with clamping to 0-127)

    - `Invert` - Mirror melody around a pivot note (retrograde inversion)

    - `Harmonize` - Add parallel intervals (thirds, fifths, triads, etc.)

  - **Time Transformers**

    - `Quantize` - Snap timing to grid with configurable strength and swing

    - `TimeStretch` - Speed up or slow down (tempo change)

    - `TimeShift` - Move events forward/backward in time

    - `Reverse` - Retrograde (reverse note order preserving durations)

  - **Velocity Transformers**

    - `VelocityScale` - Scale by factor or compress to min/max range

    - `VelocityCurve` - Apply curves (linear, log, exp, soft, hard, custom)

    - `Humanize` - Add human-like timing and velocity variation

  - **Filter Transformers**

    - `NoteFilter` - Filter by pitch range, velocity, channel (with invert option)

    - `ScaleFilter` - Filter notes to only allow those in a given musical scale (scale mask)

    - `EventTypeFilter` - Keep or remove specific MIDI event types

  - **Track Transformers**

    - `ChannelRemap` - Remap MIDI channels

    - `TrackMerge` - Merge all tracks into a single track

    - `Arpeggiate` - Convert chords to arpeggios (up, down, up_down, random patterns)

  - **Convenience Functions**

    - `transpose()`, `quantize()`, `humanize()`, `reverse()`, `scale_velocity()`, `filter_to_scale()`

  - **Test Coverage**

    - 79 tests in `tests/test_midi_transform.py`

    - Integration tests generating MIDI files in `build/midi_files/transform_tests/`

  - **Generated MIDI Files** (`build/midi_files/transform_tests/`)

    - Pre/post pairs for each transformation: `<name>_pre.mid` and `<name>_post.mid`

    - Transformations: transposed, quantized, humanized, harmonized, arpeggiated, reversed, inverted, velocity curved, time stretched, pipeline (combined)

- **Audio Slicing File Generation Tests** - Integration tests demonstrating audio transformations (`tests/test_audio_slicing.py`)

  - **Generated Audio Files** (`build/audio_files/slicing_tests/`)

    - Pre/post pairs for each transformation: `<name>_pre.wav` and `<name>_post.wav`

    - Transformations: shuffled, reversed, pattern, repeated, filtered, sorted by duration, normalized

  - Uses `scipy.io.wavfile` for WAV file output

  **Example Usage:**

  ```python
  from coremusic.midi.utilities import MIDISequence
  from coremusic.midi.transform import Pipeline, Transpose, Quantize, Humanize, VelocityScale

  # Load MIDI file
  seq = MIDISequence.load("input.mid")

  # Create transformation pipeline
  pipeline = Pipeline([
      Transpose(semitones=5),              # Up a perfect fourth
      Quantize(grid=0.125, strength=0.8),  # Quantize to 16th notes
      VelocityScale(min_vel=40, max_vel=100),  # Compress velocity range
      Humanize(timing=0.02, velocity=10),  # Add human feel
  ])

  # Apply transformations and save
  transformed = pipeline.apply(seq)
  transformed.save("output.mid")
  ```

### Changed

- **Demo Files Reorganization** - Restructured `tests/demos/` for better organization and usability

  - **Split monolithic demos into focused single-purpose files** organized by category:

    - `analysis/` - Audio analysis (file_info, peak_rms, silence_detection)

    - `audiounit/` - AudioUnit plugins (list_plugins, plugin_info, parameter_control, factory_presets, discover_plugins, stream_format)

    - `conversion/` - Format conversion (stereo_to_mono, format_presets)

    - `devices/` - Audio device management (list_devices, default_devices, find_device)

    - `effects/` - Audio effects (create_chain, find_by_name, fourcc_reference)

    - `link/` - Ableton Link (session, beat_tracking)

    - `midi/` - MIDI (create_sequence, multi_track, routing)

    - `numpy/` - NumPy integration (read_audio, channel_analysis, format_dtypes)

    - `slicing/` - Audio slicing (onset_slicing, grid_slicing, recombine)

    - `streaming/` - Real-time streaming (input_stream, output_stream, latency_comparison)

    - `visualization/` - Audio visualization (waveform, spectrogram, spectrum)

  - **Reduced print() noise** - Demos now only print results, not verbose logging

  - **Removed sys.path.insert** - All demos now run cleanly with `uv run python`

  - **Updated README.md** - Usage examples now use `uv run python` commands

  - **Improved daw.py audio quality** - Synthesized sounds now sound musical:

    - Punchy electronic drums with pitch-swept kick, layered snare, crisp hi-hats

    - A minor bass line with warm sub-harmonic tones

    - Lush ambient pad with Am-F-C-G chord progression and detuned oscillators

    - Expressive vocal lead melody with vibrato in A minor

    - Clair de Lune-inspired piano melody in Db major

    - Jazz chord progression (Dm9-G7-Cmaj7-Am7) for MIDI instruments

    - E minor arpeggio pattern with bell-like tones for effects demo

- **Constants Export** - All constant enum classes from `coremusic.constants` are now exported directly from the main `coremusic` package for convenience:
  ```python
  # Now you can do:
  import coremusic as cm
  cm.AudioFileProperty.DATA_FORMAT
  cm.AudioFormatID.LINEAR_PCM

  # Instead of:
  from coremusic.constants import AudioFileProperty
  ```

### Deprecated

- **Legacy Constant Getter Functions** - The `get_*` functions in `coremusic.capi` (e.g., `get_audio_format_linear_pcm()`) are now deprecated in favor of the enum classes in `coremusic.constants`. The getter functions remain for backward compatibility but new code should use the enum classes:
  ```python
  # Deprecated:
  capi.get_audio_file_property_data_format()

  # Preferred:
  from coremusic import AudioFileProperty
  AudioFileProperty.DATA_FORMAT
  ```

### Fixed

- **MIDI File Save/Load Bug** - Fixed two critical bugs in MIDI file I/O (`coremusic.midi.utilities`):

  - **Track count mismatch**: Format 1 MIDI files now correctly report track count (including tempo track)

  - **Meta event parsing**: Track names and other meta events (0xFF) are now correctly parsed during load

  - Track names set via `track.name` are now properly preserved through save/load cycles

- **Docstring Typos** - Fixed 38 instances of malformed docstrings in `capi.pyx` where `"Returns:\n      f OSStatus result code"` has been corrected to `"Returns:\n        OSStatus result code (0 on success)"`.

### Added

- **CODE_REVIEW.md** - Comprehensive code review report covering architecture, API design, test coverage, and recommendations.

### Documentation

- **Enhanced Module Docstring** - The main `coremusic/__init__.py` now includes comprehensive documentation:

  - Basic usage examples with `AudioFile` context manager

  - Async/await support with `AsyncAudioFile` and `AsyncAudioQueue` examples

  - NumPy integration guide with `NUMPY_AVAILABLE` flag and usage patterns

  - Module organization overview

- **Async Classes Exported** - `AsyncAudioFile`, `AsyncAudioQueue`, `open_audio_file_async`, and `create_output_queue_async` are now included in `__all__` for better discoverability.

- **Async Audio Tutorial** (`docs/tutorials/async_audio.rst`) - Complete guide covering:

  - Async file operations with `AsyncAudioFile`

  - Streaming audio chunks asynchronously

  - Concurrent file processing with `asyncio.gather()`

  - Producer-consumer patterns for streaming

  - Integration with web frameworks (FastAPI example)

- **API Quickstart Guide** (`docs/api/quickstart.rst`) - Rapid introduction covering:

  - Import patterns for OO and functional APIs

  - Audio file, AudioUnit, and MIDI operations

  - Constants usage with enum classes

  - Async operations and error handling

  - NumPy integration and quick reference table

- **Common Patterns Cookbook** (`docs/cookbook/common_patterns.rst`) - Essential patterns including:

  - Resource management (context managers, multiple resources)

  - Error handling (graceful recovery, retry patterns)

  - Format handling (detection, validation, conversion pipelines)

  - Streaming patterns (generators, progress tracking)

  - Caching patterns (LRU cache, file hash cache)

  - Batch processing (parallel and sequential)

## [0.1.9]

### Added

- **Performance Optimizations Suite** - Complete infrastructure for high-performance audio processing (January 2025)

  - **Memory-Mapped File Access** (`src/coremusic/audio/mmap_file.py`)

    - `MMapAudioFile` class for fast random access to large audio files without loading into RAM

    - Support for WAV and AIFF format parsing with zero-copy access

    - NumPy integration with zero-copy when possible via `read_as_numpy()`

    - Array-like indexing support (`file[100:200]`) for intuitive frame access

    - Properties: `format`, `frame_count`, `duration`, `sample_rate`, `channels`

    - Context manager support for automatic resource cleanup

    - Lazy format parsing - only reads metadata when needed

    - 19 comprehensive tests in `tests/test_mmap_file.py` (100% passing)

  - **Buffer Pooling System** (`src/coremusic/audio/buffer_pool.py`)

    - `BufferPool` class for thread-safe buffer reuse to reduce allocation overhead

    - `PooledBuffer` context manager for automatic buffer acquisition and release

    - Statistics tracking (cache hits, misses, hit rate, outstanding buffers)

    - Global pool management with `get_global_pool()` and `reset_global_pool()`

    - Configurable max buffers per size with LRU eviction

    - `BufferPoolStats` class for detailed performance monitoring

    - Fixed critical deadlock bugs in stats property and summary method

    - 23 comprehensive tests in `tests/test_buffer_pool.py` (100% passing)

  - **Cython Performance Optimizations** (consolidated into `src/coremusic/capi.pyx`)

    - High-performance audio operations with typed memoryviews (`float32_t[:, ::1]`)

    - GIL release with `nogil` for parallel processing capabilities

    - Zero-overhead inline utility functions (`clip_float32`, `db_to_linear`, `linear_to_db`)

    - Compiler directives for maximum performance (`boundscheck=False`, `wraparound=False`, `cdivision=True`)

    - **Normalization Functions**:

      - `normalize_audio()` / `normalize_audio_float32()` - Peak normalization with target level

    - **Gain Functions**:

      - `apply_gain()` / `apply_gain_float32()` - dB-based gain adjustment

    - **Signal Analysis**:

      - `calculate_rms()` / `calculate_rms_float32()` - RMS level calculation

      - `calculate_peak()` / `calculate_peak_float32()` - Peak amplitude detection

    - **Format Conversions**:

      - `convert_float32_to_int16()` - Float to 16-bit integer with clipping

      - `convert_int16_to_float32()` - Integer to float normalization

      - `stereo_to_mono_float32()` - Stereo to mono downmixing (average)

      - `mono_to_stereo_float32()` - Mono to stereo upmixing (duplicate)

    - **Audio Mixing**:

      - `mix_audio_float32()` - Mix two audio signals with configurable ratio

    - **Fade Effects**:

      - `apply_fade_in_float32()` - Linear fade-in with configurable duration

      - `apply_fade_out_float32()` - Linear fade-out with configurable duration

    - 22 comprehensive tests in `tests/test_cython_ops.py` (100% passing)

    - Performance test verifies < 100ms for 10 seconds of 44.1kHz stereo audio

  - **Benchmarking Suite** (`benchmarks/bench_performance.py`)

    - Comprehensive benchmark infrastructure for performance measurement

    - Benchmarks for AudioFile, MMapAudioFile, BufferPool, and Cython operations

    - Statistics collection (mean, median, standard deviation)

    - Warmup runs to stabilize measurements

    - Multiple iterations with outlier detection

    - Configurable file paths and iteration counts

  - **Integration with Audio Package**

    - All Cython optimizations exported from `coremusic.audio` module

    - `CYTHON_OPS_AVAILABLE` flag for runtime feature detection

    - Backward compatible - existing code continues to work

    - Zero-copy operations when possible for maximum performance

  - **Total Test Count**: 1234 tests passing (1170 existing + 64 new performance tests)

  - **Zero Test Regressions**: All existing functionality preserved

  **Performance Benefits:**
  - Memory-mapped files: Fast random access without loading entire file into memory

  - Buffer pooling: Reduced allocation overhead through buffer reuse

  - Cython optimizations: 10-100x speedup for common audio operations vs pure Python

  - GIL release: Enables parallel processing and concurrent operations

  **Example Usage:**

  ```python
  import coremusic as cm
  import numpy as np

  # Memory-mapped file access (fast random access)
  with cm.MMapAudioFile("large_file.wav") as mmap_file:
      # Fast random frame access without loading entire file
      chunk = mmap_file[1000:2000]  # Read frames 1000-2000

      # Zero-copy NumPy access when possible
      audio_np = mmap_file.read_as_numpy(start_frame=0, num_frames=44100)

      print(f"Duration: {mmap_file.duration:.2f}s")
      print(f"Format: {mmap_file.format}")

  # Buffer pooling (efficient memory management)
  from coremusic.audio import BufferPool, get_global_pool

  # Use global pool
  with get_global_pool().acquire(size=4096) as buffer:
      # Use buffer for audio processing
      # Automatically returned to pool when done
      pass

  # Or create custom pool
  pool = BufferPool(max_buffers_per_size=10)
  with pool.acquire(size=8192) as buffer:
      process_audio(buffer)

  # Check pool statistics
  stats = pool.stats
  print(f"Hit rate: {stats['hit_rate']:.1%}")
  print(f"Outstanding: {stats['outstanding']}")

  # Cython-optimized operations (10-100x faster)
  audio = np.random.randn(44100, 2).astype(np.float32)

  # Normalize audio (very fast)
  normalized = cm.normalize_audio(audio, target_peak=0.9)

  # Apply gain in dB
  gained = cm.apply_gain(audio, gain_db=6.0)

  # Calculate signal metrics
  rms = cm.calculate_rms(audio)
  peak = cm.calculate_peak(audio)

  # Mix two signals
  output = np.zeros_like(audio)
  cm.mix_audio_float32(output, audio, other_audio, mix_ratio=0.5)

  # Apply fades
  cm.apply_fade_in_float32(audio, fade_frames=2205)  # 50ms at 44.1kHz
  cm.apply_fade_out_float32(audio, fade_frames=2205)

  # Format conversions
  int16_data = np.zeros((44100, 2), dtype=np.int16)
  cm.convert_float32_to_int16(audio, int16_data)

  # Channel conversions
  mono = np.zeros(44100, dtype=np.float32)
  cm.stereo_to_mono_float32(audio, mono)
  ```

  **Use Cases:**
  - High-performance audio applications requiring fast I/O

  - Real-time audio processing with low latency requirements

  - Large audio file manipulation without memory constraints

  - Batch processing workflows with buffer reuse

  - Audio analysis and DSP requiring maximum performance

  - Professional audio software with strict performance requirements

- **MIDI and AudioUnit Plugin Support for DAW Module** - Complete MIDI sequencing and plugin integration (October 2025)

  - **MIDINote and MIDIClip Classes** (`src/coremusic/daw.py`)

    - `MIDINote` dataclass for individual MIDI notes with pitch, velocity, timing, duration, and channel

    - `MIDIClip` class for MIDI note containers with sorting and time-range queries

    - `add_note()` method for adding MIDI notes with automatic sorting

    - `get_notes_in_range()` for querying notes within time ranges

  - **Enhanced Clip Class** with MIDI support

    - Added `clip_type` parameter ('audio' or 'midi')

    - `is_midi` property for type checking

    - Support for `MIDIClip` as source data

    - Unified API for both audio and MIDI clips

  - **AudioUnitPlugin Class** - Complete wrapper for AudioUnit instruments and effects

    - Automatic AudioUnit initialization with sample rate configuration

    - `send_midi()` method for sending MIDI events to instruments

    - `process_audio()` method for audio effects processing

    - Support for both 4-character codes and full plugin names

    - Proper resource management with `dispose()` and `__del__` cleanup

    - Works with both instrument (`aumu`) and effect (`aufx`) plugins

  - **Enhanced Track Class** plugin support

    - Updated `add_plugin()` creates `AudioUnitPlugin` instances

    - New `set_instrument()` method for MIDI track instruments

    - Plugin chain management (instruments first, then effects)

    - Support for both audio processing and MIDI-driven instruments

  - **Comprehensive Demo Enhancements** (`tests/demos/demo_daw.py`)

    - **MIDI Rendering Functions**:

      - `render_midi_to_audio()` - Convert MIDI notes to audio with synthesized instruments

      - Support for piano, synth, and bass instrument types

      - MIDI note-to-frequency conversion (440Hz = MIDI note 69)

      - Velocity-sensitive rendering with harmonic generation

    - **Audio Effects Functions**:

      - `apply_delay_effect()` - Delay/echo with configurable feedback and mix

      - `apply_reverb_effect()` - Comb filter-based reverb (Freeverb-inspired)

      - Effect chaining support for complex processing

    - **New Demo Functions**:

      - `demo_midi_clip()` - C major scale with piano rendering (creates `midi_piano_melody.wav`)

      - `demo_midi_instruments()` - Chord progression (Am-F-C-G) with 3 instruments (piano/synth/bass)

      - `demo_audio_effects()` - Delay, reverb, and combined effects demonstration

    - **13 Audio Files Generated**:

      - 4 MIDI demonstration files (piano melody + 3 chord variations)

      - 4 effects demonstration files (original + delay + reverb + combo)

      - 5 DAW workflow files (full mix + 4 track stems)

  - **Test Coverage**: All demos run successfully with audio output verification

  - **Total Test Count**: 1074 tests passing (DAW module fully functional)

  **Example Usage:**

  ```python
  import coremusic as cm
  from coremusic.daw import MIDIClip, Clip, Timeline

  # Create MIDI clip with notes
  midi_clip = MIDIClip()
  midi_clip.add_note(note=60, velocity=100, start_time=0.0, duration=0.5)  # C4
  midi_clip.add_note(note=64, velocity=90, start_time=0.5, duration=0.5)   # E4
  midi_clip.add_note(note=67, velocity=95, start_time=1.0, duration=0.5)   # G4

  # Add to MIDI track with instrument
  timeline = Timeline(sample_rate=48000, tempo=120.0)
  piano_track = timeline.add_track("Piano", "midi")
  piano_track.set_instrument("dls ")  # DLSMusicDevice (Apple GM synth)

  # Add MIDI clip to track
  clip = Clip(midi_clip, clip_type="midi")
  clip.duration = 2.0
  piano_track.add_clip(clip, start_time=0.0)

  # Add audio effects to track
  piano_track.add_plugin("AUDelay", plugin_type="effect")
  piano_track.add_plugin("AUReverb", plugin_type="effect")

  # Audio track with effects
  guitar_track = timeline.add_track("Guitar", "audio")
  guitar_track.add_plugin("AUHighpass", plugin_type="effect")
  guitar_track.add_plugin("AUDelay", plugin_type="effect")
  ```

  **Use Cases:**
  - MIDI sequencing and composition

  - Virtual instrument playback (software synths)

  - Audio effects processing chains

  - Complete DAW-style production workflows

  - Live MIDI performance with effects

  - Music production and arrangement

## [0.1.8]

### Added

- **DAW (Digital Audio Workstation) Essentials Module** - Complete DAW building blocks for multi-track applications (January 2025)

  - **New Module**: `coremusic.daw` provides high-level DAW abstractions

  - **Timeline Class** - Multi-track timeline with transport control

    - Sample rate and tempo configuration

    - Multi-track audio and MIDI support

    - Transport control: play, pause, stop, record

    - Playhead position management

    - Timeline duration calculation

    - Ableton Link synchronization support

    - Session state tracking (playing, recording)

  - **Track Class** - Individual audio or MIDI track

    - Audio and MIDI track types

    - Clip management (add, remove, query by time)

    - Volume, pan, mute, solo controls

    - Recording arm state

    - AudioUnit plugin chain integration

    - Parameter automation lanes

    - Automatic clip organization

  - **Clip Class** - Audio/MIDI clip representation

    - Audio file or MIDI sequence source

    - Trim functionality with offset and duration

    - Fade in/out support

    - Gain control (linear multiplier)

    - Method chaining for fluent API

    - Automatic duration detection from audio files

    - Timeline positioning (start time, end time)

  - **AutomationLane Class** - Parameter automation

    - Time-based automation points

    - Three interpolation modes:

      - Linear interpolation (smooth transitions)

      - Step interpolation (instant changes)

      - Cubic interpolation (smooth curves)

    - Automatic point sorting by time

    - Value interpolation at any time point

    - Point management (add, remove, clear)

  - **TimelineMarker Class** - Markers and cue points

    - Position-based markers (seconds)

    - Named markers with optional colors

    - Automatic sorting by position

    - Range-based marker queries

  - **TimeRange Class** - Time range representation

    - Start/end time with duration calculation

    - Containment checking

    - Loop region support

  - **Integration Features**:

    - AudioUnit plugin loading and configuration

    - Ableton Link tempo synchronization

    - Automatic clip duration from AudioFile

    - Transport control with state management

  - **Comprehensive Test Coverage**: 52 tests in `tests/test_daw.py` (100% passing)

  - **Interactive Demo**: `tests/demos/demo_daw.py` with 10 examples

  - **Total Test Count**: 1074 tests passing, 46 skipped (up from 1022 passed)

  **Example Usage:**

  ```python
  import coremusic as cm

  # Create DAW session
  timeline = cm.Timeline(sample_rate=48000, tempo=128.0)

  # Add tracks
  drums = timeline.add_track("Drums", "audio")
  vocals = timeline.add_track("Vocals", "audio")

  # Add clips with trimming and fades
  drums.add_clip(cm.Clip("drums.wav"), start_time=0.0)
  vocals.add_clip(
      cm.Clip("vocals.wav").trim(2.0, 26.0).set_fades(0.5, 1.0),
      start_time=8.0
  )

  # Add automation
  volume_auto = vocals.automate("volume")
  volume_auto.add_point(8.0, 0.0)   # Fade in
  volume_auto.add_point(10.0, 1.0)  # Full volume

  # Add markers and loop region
  timeline.add_marker(0.0, "Intro")
  timeline.add_marker(16.0, "Chorus")
  timeline.set_loop_region(16.0, 32.0)

  # Transport control
  timeline.play()
  timeline.pause()
  timeline.stop()

  # Recording
  vocals.record_enable(True)
  timeline.record()
  ```

  **Use Cases:**
  - Multi-track audio/MIDI recording applications

  - DAW-like timeline interfaces

  - Music production software

  - Live performance tools with transport control

  - Automated mixing and mastering workflows

  - Educational DAW implementations

  - Audio post-production tools

- **Audio Analysis and Feature Extraction** - Comprehensive audio analysis framework for music information retrieval (October 2025)

  - **New Module**: `coremusic.audio.analysis` provides advanced audio analysis capabilities

  - **AudioAnalyzer Class** for comprehensive audio feature extraction

    - **Beat Detection**: Onset-based beat detection with tempo estimation

      - Spectral flux onset detection

      - Autocorrelation-based tempo estimation

      - Downbeat detection for bar tracking

      - Confidence scoring for detection quality

    - **Pitch Detection**: Autocorrelation-based pitch tracking

      - Fundamental frequency detection

      - MIDI note number conversion

      - Cents offset calculation for tuning analysis

      - Confidence scoring per frame

    - **Spectral Analysis**: Frequency domain feature extraction

      - Spectral centroid (brightness measure)

      - Spectral rolloff (frequency content boundary)

      - Peak detection in frequency spectrum

      - FFT-based spectrum analysis at any time point

    - **MFCC Extraction**: Mel-Frequency Cepstral Coefficients

      - Configurable coefficient count (default 13)

      - Mel filterbank implementation

      - DCT transformation for cepstral features

      - Frame-by-frame MFCC matrices

    - **Key Detection**: Musical key and mode estimation

      - Chromagram computation (12 pitch classes)

      - Krumhansl-Schmuckler key profiles

      - Major/minor mode detection

      - Time-averaged chroma analysis

    - **Audio Fingerprinting**: Unique audio identification

      - Spectral peak extraction

      - Peak constellation mapping

      - Hash-based fingerprint generation

      - Content-based audio matching

  - **LivePitchDetector Class** for real-time pitch tracking

    - Streaming pitch detection for live audio

    - Autocorrelation-based algorithm

    - Configurable buffer size and sample rate

    - Returns PitchInfo with frequency, MIDI note, and confidence

  - **Data Classes** for structured results

    - **BeatInfo**: tempo, beats, downbeats, confidence

    - **PitchInfo**: frequency, midi_note, cents_offset, confidence

  - **SciPy Integration**: Leverages scipy.signal for DSP operations

  - **Optional Dependencies**: Requires NumPy and SciPy with graceful fallback

  - **Comprehensive Test Coverage**: 42 tests in `tests/test_audio_analysis.py` (100% passing)

  - **Interactive Demo**: `tests/demos/demo_audio_analysis.py` with 8 examples

  - **Total Test Count**: 942 tests passing, 33 skipped (up from 900 passed)

  **Example Usage:**

  ```python
  import coremusic as cm

  # Beat detection and tempo estimation
  analyzer = cm.AudioAnalyzer("song.wav")
  beat_info = analyzer.detect_beats()
  print(f"Tempo: {beat_info.tempo:.1f} BPM")
  print(f"Beats: {beat_info.beats[:5]}")  # First 5 beat times
  print(f"Downbeats: {beat_info.downbeats}")

  # Pitch detection and tracking
  pitch_info = analyzer.detect_pitch()
  print(f"Frequency: {pitch_info.frequency:.2f} Hz")
  print(f"MIDI Note: {pitch_info.midi_note}")
  print(f"Cents: {pitch_info.cents_offset:+.1f}")

  # Spectral analysis at specific time
  spectrum = analyzer.analyze_spectrum(time=1.0, window_size=0.1)
  print(f"Centroid: {spectrum['centroid']:.1f} Hz")
  print(f"Rolloff: {spectrum['rolloff']:.1f} Hz")
  print(f"Peaks: {spectrum['peaks'][:3]}")  # Top 3 peaks

  # MFCC extraction for timbre analysis
  mfcc = analyzer.extract_mfcc(n_mfcc=13)
  print(f"MFCC shape: {mfcc.shape}")  # (13, n_frames)

  # Key detection
  key, mode = analyzer.detect_key()
  print(f"Key: {key} {mode}")  # e.g., "C major"

  # Audio fingerprinting
  fingerprint = analyzer.get_audio_fingerprint()
  print(f"Fingerprint: {fingerprint[:64]}...")  # First 64 chars

  # Real-time pitch detection
  live_detector = cm.LivePitchDetector(sample_rate=44100.0, buffer_size=2048)
  for audio_chunk in stream:
      pitch_info = live_detector.process(audio_chunk)
      if pitch_info and pitch_info.confidence > 0.8:
          print(f"Pitch: {pitch_info.frequency:.2f} Hz")
  ```

  **Use Cases:**
  - Music information retrieval and analysis

  - Beat tracking for DJ software and auto-sync

  - Pitch detection for tuning and vocal analysis

  - Automatic key detection for harmonic mixing

  - Audio fingerprinting for content identification

  - MFCC extraction for machine learning features

  - Real-time pitch tracking for live performance

  - Spectral analysis for sound design and synthesis

- **Audio Slicing and Recombination** - Complete audio slicing framework for creative sample manipulation (October 2025)

  - **New Module**: `coremusic.audio.slicing` provides comprehensive audio slicing and recombination tools

  - **Slicing Methods**: 5 different slicing algorithms for various use cases

    - **Onset Detection**: Spectral flux-based onset detection for rhythmic material

    - **Transient Detection**: Envelope analysis with dB thresholding for dynamic changes

    - **Zero-Crossing Detection**: Glitch-free slicing at zero crossings

    - **Grid-Based Slicing**: Regular equal-duration divisions with optional beat alignment

    - **Manual Slicing**: User-specified time points for precise control

  - **Slice Dataclass** with properties for duration and sample count

  - **AudioSlicer Class** for detecting and extracting audio slices

    - Configurable sensitivity parameter (0.0-1.0)

    - Optional maximum slice count limiting

    - Minimum slice duration filtering

    - Export slices as individual audio files

  - **SliceCollection Class** with fluent API for slice manipulation

    - `shuffle()` - Randomize slice order

    - `reverse()` - Reverse slice sequence

    - `repeat(times)` - Duplicate slices

    - `filter(predicate)` - Filter slices by condition

    - `sort_by_duration()` - Sort by slice length

    - `select(indices)` - Select specific slices

    - `apply_pattern(pattern)` - Apply custom patterns

    - Method chaining support for complex operations

  - **SliceRecombinator Class** with 5 recombination strategies

    - **Original**: Maintain original order with crossfading

    - **Random**: Random selection and ordering

    - **Reverse**: Reversed order

    - **Pattern**: Custom index-based patterns

    - **Custom**: User-defined ordering functions

    - Crossfading algorithm for smooth transitions (configurable duration)

    - Optional normalization of output audio

  - **Comprehensive Test Coverage**: 50 tests in `tests/test_audio_slicing.py` (100% passing)

  - **Interactive Demo**: `tests/demos/demo_audio_slicing.py` with 9 examples

  - **Total Test Count**: 942 tests passing, 33 skipped (up from 905 passed)

  **Example Usage:**

  ```python
  import coremusic as cm

  # Slice using onset detection
  slicer = cm.AudioSlicer("drums.wav", method="onset", sensitivity=0.6)
  slices = slicer.detect_slices(min_slice_duration=0.05, max_slices=16)

  # Manipulate slices with fluent API
  collection = cm.SliceCollection(slices)
  shuffled = collection.filter(lambda s: s.duration > 0.1).shuffle().repeat(2)

  # Recombine with crossfading
  recombinator = cm.SliceRecombinator(shuffled)
  output = recombinator.recombine(method="random", crossfade_duration=0.01)
  recombinator.export("output.wav", method="pattern", pattern=[0, 2, 1, 3])

  # Grid slicing with beat alignment
  grid_slicer = cm.AudioSlicer("audio.wav", method="grid")
  slices = grid_slicer.detect_slices(divisions=16, tempo=120.0)

  # Zero-crossing for glitch-free slicing
  zc_slicer = cm.AudioSlicer("audio.wav", method="zero_crossing")
  slices = zc_slicer.detect_slices(target_slices=8, snap_to_zero=True)
  ```

  **Use Cases:**
  - Beat slicing for drum loops and rhythm manipulation

  - Creative sample recombination and glitch effects

  - Automatic audio segmentation for music analysis

  - Live performance sample triggering

  - Audio collage and mashup creation

- **Audio Visualization** - Comprehensive visualization tools for audio analysis (October 2025)

  - **New Module**: `coremusic.audio.visualization` provides matplotlib-based audio visualization

  - **WaveformPlotter Class** for waveform visualization

    - Basic waveform plotting with time axis

    - Optional RMS envelope overlay (configurable window size)

    - Optional peak envelope overlay

    - Time range zooming (plot specific sections)

    - Custom figure sizes and titles

    - Save to file (PNG, PDF, etc.) with configurable DPI

  - **SpectrogramPlotter Class** for time-frequency analysis

    - STFT-based spectrogram generation

    - Configurable window size and hop size

    - Multiple colormap support (viridis, magma, plasma, inferno)

    - Window function selection (hann, hamming, blackman)

    - dB scale with configurable min/max values

    - Save spectrograms with high quality

  - **FrequencySpectrumPlotter Class** for spectral analysis

    - Instant spectrum at specific time points

    - Average spectrum over time ranges

    - Logarithmic frequency scale

    - Configurable FFT window sizes (2048, 4096, 8192)

    - Frequency range filtering (min/max Hz)

    - Multiple window function support

  - **matplotlib Integration**: High-quality publication-ready plots

  - **Optional Dependency**: Gracefully handles missing matplotlib

  - **Comprehensive Test Coverage**: 37 tests in `tests/test_audio_visualization.py` (100% passing)

  - **Interactive Demo**: `tests/demos/demo_audio_visualization.py` with 11 examples

  - **Total Test Count**: 942 tests passing, 33 skipped (up from 905 passed)

  **Example Usage:**

  ```python
  import coremusic as cm

  # Plot waveform with envelopes
  plotter = cm.WaveformPlotter("audio.wav")
  fig, ax = plotter.plot(show_rms=True, show_peaks=True)
  plotter.save("waveform.png", dpi=150)

  # Generate spectrogram
  spec = cm.SpectrogramPlotter("audio.wav")
  fig, ax = spec.plot(window_size=2048, cmap="magma", min_db=-80)
  spec.save("spectrogram.png")

  # Frequency spectrum analysis
  spectrum = cm.FrequencySpectrumPlotter("audio.wav")

  # At specific time
  fig, ax = spectrum.plot(time=1.0, window_size=4096)

  # Averaged over time range
  fig, ax = spectrum.plot_average(time_range=(0, 5), hop_size=1024)
  spectrum.save("spectrum.png")

  # Complete workflow
  waveform = cm.WaveformPlotter("audio.wav")
  waveform.plot(time_range=(0.5, 1.5), show_rms=True)  # Zoom to specific range

  spec = cm.SpectrogramPlotter("audio.wav")
  spec.plot(window_size=1024, hop_size=256, cmap="plasma")
  ```

  **Use Cases:**
  - Audio analysis and debugging

  - Music production visualization

  - Scientific audio research

  - Educational demonstrations

  - Publication-quality figures

  - Real-time audio monitoring (with matplotlib animation)

- **OSStatus Error Translation** - Human-readable error messages with recovery suggestions (October 2025)

  - **New Module**: `coremusic.os_status` provides comprehensive OSStatus error code translation

  - **Error Code Coverage**: 100+ error codes from all CoreAudio frameworks

    - AudioHardware errors (13 codes): device, stream, property errors

    - AudioFile errors (14 codes): file I/O, format, permissions errors

    - AudioFormat errors (6 codes): format validation errors

    - AudioFileStream errors (12 codes): streaming parser errors

    - AudioCodec errors (9 codes): codec operation errors

    - AudioUnit errors (20 codes): plugin lifecycle and configuration errors

    - AudioQueue errors (23 codes): queue management errors

    - System errors (4 codes): parameter, memory, permission errors

  - **Translation Functions**:

    - `os_status_to_string(status)` - Convert OSStatus to "ErrorName: Description"

    - `get_error_suggestion(status)` - Get recovery suggestion for error

    - `format_os_status_error(status, operation)` - Complete formatted error message

    - `get_error_info(status)` - Get (name, description, suggestion) tuple

  - **Recovery Suggestions**: 30+ actionable suggestions for common errors

    - File errors: Check permissions, verify path exists, check file format

    - Hardware errors: Check device connection, wait for ready state

    - AudioUnit errors: Verify initialization, check format compatibility

    - Parameter errors: Validate ranges, check format parameters

  - **Enhanced Exception Classes**:

    - Added `CoreAudioError.from_os_status()` class method

    - Automatically formats error with name, description, and suggestion

    - Works with all exception subclasses (AudioFileError, AudioQueueError, etc.)

  - **FourCC Support**: Translates both integer codes and four-character codes

  - **Comprehensive Test Coverage**: 31 new tests in `tests/test_os_status.py` (100% passing)

  - **Zero Dependencies**: Pure Python implementation using only stdlib

  - **Total Test Count**: 735 tests passing, 45 skipped (up from 681 passed, 32 skipped)

    - Added 31 os_status tests

    - Enabled 4 previously skipped AudioQueue tests

    - Improved 19 test assertions across multiple test files

  **Example Usage:**

  ```python
  import coremusic as cm
  from coremusic import os_status

  # Translate error codes
  print(os_status.os_status_to_string(-43))
  # Output: kAudioFileFileNotFoundError: File not found

  # Get recovery suggestion
  suggestion = os_status.get_error_suggestion(-43)
  # Output: Verify the file path exists and is spelled correctly

  # Complete formatted error
  msg = os_status.format_os_status_error(-43, "open audio file")
  # Output: Failed to open audio file: kAudioFileFileNotFoundError (File not found)
  #         Suggestion: Verify the file path exists and is spelled correctly

  # Use with exceptions
  exc = cm.AudioFileError.from_os_status(-43, "load file")
  raise exc
  # AudioFileError: Failed to load file: kAudioFileFileNotFoundError: File not found.
  #                 Verify the file path exists and is spelled correctly
  ```

  **Before (cryptic numeric codes):**
  ```
  RuntimeError: AudioFileOpenURL failed with status: -43
  ```

  **After (human-readable with suggestion):**
  ```
  AudioFileError: Failed to open audio file: kAudioFileFileNotFoundError: File not found.
  Verify the file path exists and is spelled correctly
  ```

  **Impact:**
  - **Developers**: Much easier debugging with clear error names and actionable suggestions

  - **Users**: Better error messages guide them to fix issues themselves

  - **Support**: Reduced support burden with self-explanatory error messages

  - **Documentation**: Error codes now self-documenting

  **Implementation Details:**
  - **Capi Layer Integration**: 150+ error locations updated across `src/coremusic/capi.pyx`

    - Added `format_osstatus_error()` helper function

    - Integrated with `coremusic.log` module for structured error logging

    - All `RuntimeError` messages now include human-readable translations

  - **Objects Layer**: Enhanced `CoreAudioError.from_os_status()` class method

    - Automatically formats errors with name, description, and suggestion

    - Preserves status_code attribute for programmatic access

  - **Test Updates**: Comprehensive test suite updates to work with new error format

    - Updated `tests/test_objects_audio_queue.py`: Changed error detection from `"status: -50"` to `e.status_code == -50 or "paramErr" in str(e)`

    - Updated `tests/test_objects_comprehensive.py`: Similar paramErr error detection pattern

    - Updated `tests/test_coremidi.py`: Changed 18 assertions from `"failed with status"` to `"failed"`

    - Updated `tests/test_audiotoolbox_music_device.py`: Enhanced fixture error detection for userCanceledErr and InvalidFile

    - All tests now verify errors exist without depending on exact message format

  - **Logging Integration**: Structured logging with extra context

    - Error logs include status_code, operation, and suggestion fields

    - Controlled via DEBUG environment variable (DEBUG=0 disables logging)

  - **Demo Application**: `tests/demos/demo_os_status_errors.py` with 7 comprehensive examples

    - Basic error translation demonstration

    - Recovery suggestions showcase

    - Complete formatted error messages

    - Enhanced exception classes usage

    - Structured error information

    - Real-world scenario (file not found)

    - Error categories overview

  - **Backward Compatibility**: Status codes still preserved in exception attributes

### Changed

- **Unified AudioAnalyzer class** - Merged basic utility methods into comprehensive analysis class (October 2025)

  - **Issue**: Two separate `AudioAnalyzer` classes existed with naming conflict

    - `coremusic.audio.analysis.AudioAnalyzer` - Advanced music analysis (beat/pitch detection, MFCC, key detection)

    - `coremusic.audio.utilities.AudioAnalyzer` - Basic metrics (silence detection, peak, RMS)

  - **Solution**: Merged utility methods into analysis class as static methods

  - **New unified API**:

    - **Instance Methods** (advanced analysis, requires SciPy):

      - `detect_beats()` - Beat detection and tempo estimation

      - `detect_pitch()` - Pitch tracking over time

      - `analyze_spectrum()` - Spectral analysis

      - `extract_mfcc()` - MFCC extraction

      - `detect_key()` - Musical key detection

      - `get_audio_fingerprint()` - Audio fingerprinting

    - **Static Methods** (basic metrics, NumPy only):

      - `detect_silence()` - Find quiet regions in audio

      - `get_peak_amplitude()` - Maximum amplitude

      - `calculate_rms()` - RMS level calculation

      - `get_file_info()` - Comprehensive file metadata

  - **Benefits**:

    - Single import for all audio analysis: `from coremusic.audio.analysis import AudioAnalyzer`

    - Naming conflict resolved - one AudioAnalyzer class

    - Flexible API - choose static or instance methods based on needs

    - No breaking changes - existing code continues to work

  - **Example Usage**:
    ```python
    # Static API (no initialization, lightweight)
    silence = AudioAnalyzer.detect_silence("audio.wav", threshold_db=-40)
    peak = AudioAnalyzer.get_peak_amplitude("audio.wav")

    # Instance API (advanced analysis)
    analyzer = AudioAnalyzer("song.wav")
    beat_info = analyzer.detect_beats()
    key, mode = analyzer.detect_key()
    ```
  - **Migration**: Tests updated to import from `coremusic.audio.analysis`

  - **Verification**: All 1022 tests passing, type checking successful

- **Reorganized utilities module** - Moved `coremusic.utilities` to `coremusic.audio.utilities` (October 2025)

  - **Change**: Relocated utilities module for better package organization

    - From: `coremusic.utilities`

    - To: `coremusic.audio.utilities`

  - **Reason**: Utilities are audio-specific and belong in audio subpackage

  - **Updated imports**:

    - Main package: `from .audio.utilities import *` in `coremusic/__init__.py`

    - Audio package: Added utilities exports to `coremusic.audio.__init__.py`

    - Fixed all relative imports within utilities.py (`.` → `..`)

  - **Exports remain accessible**: All utilities still available via `import coremusic as cm`

  - **No breaking changes**: Existing user code continues to work

  - **Verification**: All 1022 tests passing, type checking successful

- **Improved test coverage for AudioQueue OO API** - Selective skipping instead of blanket test exclusion (October 2025)

  - **Issue**: All 16 tests in `test_objects_audio_queue.py` were skipped due to module-level `pytestmark`

  - **Root Cause**: Overly conservative assumption that all AudioQueue tests require audio hardware

  - **Fix**: Removed blanket skip marker and implemented selective skipping using fixture-based hardware detection

  - **Result**: 4 tests now passing (25% → 100% execution for non-hardware tests), 12 tests properly skip when hardware unavailable

  - **Tests now running without hardware**:

    - `test_audio_buffer_creation` - Pure Python object creation

    - `test_audio_buffer_properties` - Property access testing

    - `test_audio_queue_creation_with_format` - Object initialization

    - `test_audio_queue_error_handling` - Error handling for invalid formats

  - **Hardware-dependent tests** gracefully skip with clear message: "Audio hardware not available"

  - **Impact**: Better CI/headless environment coverage while preserving hardware functionality tests

  - **Verification**: All 681 tests passing, 32 skipped (no regressions)

  - **Updated error detection**: Changed from checking for string "status: -50" to checking status_code attribute or "paramErr" keyword

  - **Better compatibility**: Tests now work with new human-readable error format

- **Enhanced documentation for bytes parameters** - Added comprehensive usage examples to method docstrings (October 2025)

  - **Analysis**: Identified 6 methods accepting `bytes` parameters representing binary audio/MIDI data

  - **Confirmed**: All `bytes` parameters are correctly typed (binary data, not text):

    - `AudioFileStream.parse_bytes()` - Raw audio file format data (WAV/MP3/AAC headers)

    - `AudioConverter.convert()` - Raw PCM audio samples

    - `AudioConverter.convert_with_callback()` - Raw audio samples (already had example)

    - `AudioConverter.set_property()` - Binary property data (structs, ints)

    - `ExtendedAudioFile.write()` - Raw audio frame data

    - `MIDIOutputPort.send_data()` - MIDI protocol messages

  - **Documentation improvements**:

    - Added practical usage examples to 5 methods (1 already had examples)

    - Clarified binary nature of data with inline comments

    - Showed proper `struct.pack()` usage for creating binary data

    - Demonstrated MIDI protocol byte construction

    - Included context managers and realistic workflows

  - **Consistency**: All examples follow existing pattern from `convert_with_callback()`

  - **Verification**: All 681 tests passing (documentation-only changes, no functional impact)

### Fixed

- **Music device test fixture errors** - Enhanced error detection for security restrictions and invalid components (October 2025)

  - **Issue**: Fixture in `test_audiotoolbox_music_device.py` was showing ERROR instead of SKIPPED for unavailable/broken plugins

  - **Root Cause**: Error detection was checking for numeric code "-128" but new OSStatus translation changed format to "userCanceledErr"

  - **Fix**: Updated error detection to check for keyword "userCanceledErr" or "security restriction" instead of numeric codes

  - **Also handles**: kAudioUnitErr_InvalidFile (-10863) for broken third-party plugins

  - **Impact**: All 4 test errors converted to proper skips with clear messages

  - **Result**: Tests now gracefully skip when encountering:

    - Security-restricted components (userCanceledErr -128)

    - Invalid/broken plugin files (kAudioUnitErr_InvalidFile -10863)

    - Other component instantiation failures

  - **Final test count**: 735 passed, 45 skipped (up from 735 passed, 41 skipped, 4 errors)

- **AudioUnit factory presets crash** - Fixed critical bug in `audio_unit_get_factory_presets()` (src/coremusic/capi.pyx:1802)

  - **Root Cause**: Code was incorrectly treating `kAudioUnitProperty_FactoryPresets` return value as a CFArray of CFDictionaries

  - **Correct Implementation**: According to [Apple TechNote TN2157](https://developer.apple.com/library/archive/technotes/tn2157/_index.html), the property returns a CFArray of `AUPreset` structs

  - **Fix**: Changed implementation to cast array elements directly to `AUPreset*` pointers and access struct fields (`presetNumber`, `presetName`) instead of creating CFString keys and using `CFDictionaryGetValue()`

  - **Impact**: Eliminated crashes when querying factory presets from AudioUnits like AUDynamicsProcessor, AUDistortion, etc.

  - **Test Results**: Successfully discovered factory presets from 9 Apple plugins (48 total tested, 18.8% have presets)

  - Added missing `preset_name` variable declaration that was causing undefined behavior

- **Music device test fixture errors** - Improved error handling in `test_audiotoolbox_music_device.py`

  - **Issue**: Test fixture was encountering broken/unavailable third-party music device plugins returning error -10863 (`kAudioUnitErr_InvalidFile`)

  - **Fix**: Added -10863 to the list of errors that trigger test skip (alongside existing -128 handling)

  - **Impact**: Tests now properly skip instead of erroring when encountering incompatible plugins

  - **Result**: All 677 tests passing, 37 skipped, 0 errors

### Added

- **AudioUnit Host Enhancements** - Advanced audio format support, preset management, and plugin chaining

  - **AudioFormat Class** (`src/coremusic/audiounit_host.py:18-93`)

    - Support for multiple sample formats: `float32`, `float64`, `int16`, `int32`

    - Interleaved and non-interleaved buffer layout support

    - Properties: `bytes_per_sample`, `bytes_per_frame`

    - Format comparison and dictionary serialization

    - Type-safe format specification with string constants

  - **AudioFormatConverter Class** (`src/coremusic/audiounit_host.py:94-243`)

    - Automatic format conversion between any supported formats

    - Two-stage conversion pipeline: source → float32 interleaved → destination

    - Proper audio normalization to [-1.0, 1.0] range

    - Support for all format combinations (format, bit depth, channel layout)

    - Symmetric rounding for integer formats

  - **PresetManager Class** (`src/coremusic/audiounit_host.py:341-535`)

    - Complete preset lifecycle management (save/load/export/import)

    - JSON-based preset storage in `~/Library/Audio/Presets/coremusic/`

    - Preset metadata: name, description, plugin info, timestamp

    - Parameter state capture and restoration

    - Preset validation and compatibility checking

    - List, delete, export, and import operations

  - **AudioUnitChain Class** (`src/coremusic/audiounit_host.py:1169-1438`)

    - Sequential plugin processing with automatic routing

    - Dynamic chain building: add, insert, remove plugins

    - Automatic format conversion between plugins

    - Wet/dry mixing support (blend processed and original signals)

    - Plugin configuration by index

    - Context manager support for automatic cleanup

    - Method chaining for fluent API

  - **Enhanced AudioUnitPlugin** (`src/coremusic/audiounit_host.py:565-930`)

    - `set_audio_format()` - Configure plugin audio format

    - `process()` enhanced with format parameter for automatic conversion

    - `save_preset()`, `load_preset()`, `list_user_presets()` - Preset management

    - `delete_preset()`, `export_preset()`, `import_preset()` - Preset operations

    - `audio_format` property for format queries

  - **Comprehensive test coverage** - 37 new tests in `tests/test_audiounit_host_enhancements.py`

    - 5 AudioFormat tests (creation, properties, equality, serialization)

    - 7 AudioFormatConverter tests (all format combinations, interleaved/non-interleaved)

    - 6 PresetManager tests (save, load, list, delete, export, import)

    - 3 Plugin enhancement tests (format setting, conversion, integration)

    - 14 AudioUnitChain tests (creation, operations, processing, context manager)

    - 1 full workflow integration test

    - 27 tests passing, 10 skipped (plugins not available)

  - **Updated exports** - New classes exported from `coremusic` module

    - `AudioFormat`, `AudioFormatConverter`, `AudioUnitChain`, `PresetManager`

    - All classes available via `import coremusic as cm`

  - **Documentation updates**

    - `TODO.md` updated with completed features and usage examples

    - Test count updated to **736 tests passing** (100% success rate)

  - **Zero test regressions** - All 736 tests passing after enhancements

  **Example Usage:**

  ```python
  import coremusic as cm

  # Audio Format Support
  fmt = cm.PluginAudioFormat(44100.0, 2, cm.PluginAudioFormat.INT16, interleaved=True)
  plugin.set_audio_format(fmt)
  output = plugin.process(input_data, num_frames, fmt)

  # User Preset Management
  plugin.save_preset("My Reverb", "Large hall with 3s decay")
  plugin.load_preset("My Reverb")
  presets = plugin.list_user_presets()
  plugin.export_preset("My Reverb", "/path/to/export.json")
  plugin.import_preset("/path/to/preset.json")

  # AudioUnit Chain
  chain = cm.AudioUnitChain()
  chain.add_plugin("AUHighpass")
  chain.add_plugin("AUDelay")
  chain.add_plugin("AUReverb")
  chain.configure_plugin(0, {'Cutoff Frequency': 200.0})
  chain.configure_plugin(1, {'Delay Time': 0.5})
  output = chain.process(input_audio, num_frames, wet_dry_mix=0.8)

  # Or use context manager
  with cm.AudioUnitChain() as chain:
      chain.add_plugin("AUDelay")
      output = chain.process(input_data)
  ```

- **AudioUnit MIDI Support** - Complete MIDI control for AudioUnit instrument plugins

  - **MIDI Methods in AudioUnitPlugin class** (`src/coremusic/audiounit_host.py`)

    - `send_midi()` - Send raw MIDI messages to instrument plugins

    - `note_on()` - Send MIDI Note On with channel, note, velocity, and optional offset frames

    - `note_off()` - Send MIDI Note Off with channel, note, and optional velocity/offset

    - `control_change()` - Send MIDI Control Change (volume, pan, expression, etc.)

    - `program_change()` - Send MIDI Program Change for instrument selection (General MIDI)

    - `pitch_bend()` - Send MIDI Pitch Bend with 14-bit precision (0-16383)

    - `all_notes_off()` - Emergency stop all notes on a channel (MIDI CC 123)

    - Type checking ensures MIDI methods only work on instrument plugins (`aumu` type)

    - Sample-accurate MIDI scheduling with `offset_frames` parameter

  - **Full MIDI Specification Support**

    - All 128 MIDI notes (0-127)

    - All 128 velocity levels (0-127)

    - All 128 MIDI controllers (CC 0-127)

    - All 128 General MIDI programs (0-127)

    - 14-bit pitch bend precision (0-16383, center = 8192)

    - All 16 MIDI channels (0-15)

    - Sample-accurate timing for tight rhythmic patterns

  - **Comprehensive test coverage** - 19 new tests in `tests/test_audiounit_midi.py`

    - Basic MIDI operations (note on/off, chords, scales)

    - Velocity and note range testing across MIDI spec

    - Control Change messages (volume, pan, expression)

    - Program Change for instrument selection

    - Pitch Bend messages with smooth modulation

    - All Notes Off command

    - Multi-channel MIDI (all 16 channels)

    - Raw MIDI message sending

    - Sample-accurate scheduling with offset frames

    - Type checking (MIDI rejected on effect plugins)

    - Error handling and validation

    - Rapid note sequences (arpeggiator patterns)

    - Multi-channel orchestration

  - **Interactive demo application** - `tests/demos/audiounit_instrument_demo.py`

    - 8 comprehensive demonstrations of MIDI functionality

    - Plugin discovery (62 instrument plugins found)

    - Basic MIDI control (notes, chords, C major scale)

    - Instrument selection via General MIDI program changes

    - MIDI controller automation (volume fade, pan sweep)

    - Pitch bend demonstrations (smooth pitch modulation)

    - Multi-channel performance (4-channel orchestration example)

    - Arpeggiator patterns (rapid note sequences)

    - Interactive keyboard mapping demo

    - Integration with Apple DLSMusicDevice (built-in General MIDI synth)

  - **Updated documentation**

    - `docs/dev/audiounit_implementation.md` updated with MIDI sections and examples

    - MIDI instrument control examples

    - Multi-channel MIDI orchestration examples

    - Sample-accurate MIDI scheduling examples

    - Updated test coverage and demo information

  - **All 662 tests passing** (643 existing + 19 new MIDI tests)

  - **62 instrument plugins** discovered and working with MIDI control

  **Example Usage:**

  ```python
  import coremusic as cm
  import time

  # Load a General MIDI synthesizer
  with cm.AudioUnitPlugin.from_name("DLSMusicDevice", component_type='aumu') as synth:
      # Play a note
      synth.note_on(channel=0, note=60, velocity=100)  # Middle C
      time.sleep(1.0)
      synth.note_off(channel=0, note=60)

      # Play a chord
      notes = [60, 64, 67]  # C major (C, E, G)
      for note in notes:
          synth.note_on(channel=0, note=note, velocity=90)
      time.sleep(1.5)
      synth.all_notes_off(channel=0)

      # Change instrument (General MIDI)
      synth.program_change(channel=0, program=0)   # Acoustic Grand Piano
      synth.program_change(channel=0, program=40)  # Violin

      # Control volume with MIDI CC
      synth.control_change(channel=0, controller=7, value=100)  # Full volume
      synth.control_change(channel=0, controller=7, value=50)   # Half volume

      # Pitch bend
      synth.note_on(channel=0, note=60, velocity=100)
      synth.pitch_bend(channel=0, value=8192)   # Center (no bend)
      synth.pitch_bend(channel=0, value=12288)  # Bend up
      synth.pitch_bend(channel=0, value=8192)   # Back to center
      synth.note_off(channel=0, note=60)

  # Multi-channel orchestration
  with cm.AudioUnitPlugin.from_name("DLSMusicDevice", component_type='aumu') as synth:
      # Setup different instruments on different channels
      synth.program_change(channel=0, program=0)   # Piano
      synth.program_change(channel=1, program=48)  # Strings
      synth.program_change(channel=2, program=56)  # Trumpet

      # Play multi-channel arrangement
      synth.note_on(channel=0, note=60, velocity=90)  # Piano
      synth.note_on(channel=1, note=64, velocity=70)  # Strings
      synth.note_on(channel=2, note=72, velocity=80)  # Trumpet
      time.sleep(1.0)

      # Clean stop all channels
      for ch in range(3):
          synth.all_notes_off(channel=ch)
  ```

- **AudioPlayer.play() method** - Added `play()` as an intuitive alias for `start()` method

  - Both `player.play()` and `player.start()` now work identically

  - Improves API ergonomics and developer experience

  - Backward compatible - existing code using `start()` continues to work

- **Ableton Link Integration** - Complete tempo synchronization and beat grid support

  - **Link Cython wrapper** (`src/coremusic/link.pyx` and `link.pxd`)

    - `Clock` class - Platform-specific clock for Link timing

      - `micros()` - Get current time in microseconds

      - `ticks()` - Get current time in system ticks (mach_absolute_time)

      - `ticks_to_micros()` - Convert system ticks to microseconds

      - `micros_to_ticks()` - Convert microseconds to system ticks

    - `SessionState` class - Link timeline and transport state snapshot

      - Properties: `tempo`, `is_playing`

      - Beat/phase queries: `beat_at_time()`, `phase_at_time()`, `time_at_beat()`

      - Beat mapping: `request_beat_at_time()`, `force_beat_at_time()`

      - Transport control: `set_tempo()`, `set_is_playing()`, `time_for_is_playing()`

      - Convenience methods: `request_beat_at_start_playing_time()`, `set_is_playing_and_request_beat_at_time()`

    - `LinkSession` class - Main Link session for tempo synchronization

      - Properties: `enabled`, `num_peers`, `start_stop_sync_enabled`, `clock`

      - Session state capture: `capture_audio_session_state()`, `capture_app_session_state()`

      - Session state commit: `commit_audio_session_state()`, `commit_app_session_state()`

      - Realtime-safe audio thread operations with `nogil`

  - **AudioPlayer Link Integration** (`src/coremusic/capi.pyx`)

    - AudioPlayer now accepts optional `link_session` parameter

    - `link_session` property to access attached Link session

    - `get_link_timing(quantum)` method returns timing info dict (tempo, beat, phase, is_playing)

    - Python-layer timing queries for synchronized playback control

    - Link session reference kept alive to prevent garbage collection

  - **C++ Build Integration** (`setup.py`)

    - Link extension compiled with C++11 support

    - Include paths for Link library and ASIO standalone

    - LINK_PLATFORM_MACOSX define for macOS platform

  - **Comprehensive test coverage**

    - `test_link.py` - 25 tests covering all Link functionality

      - Clock operations (time queries, conversions, round-trip)

      - Session management (enable/disable, peers, transport sync)

      - State capture and commit (audio/app thread)

      - Tempo control and beat/phase calculations

      - Transport state management

      - Two-session synchronization tests

    - `test_link_audio_integration.py` - 9 tests for AudioPlayer integration

      - Player creation with/without Link session

      - Timing queries and updates

      - Tempo and transport state visibility

      - Multiple players sharing Link session

      - Reference lifecycle management

    - All 575 tests passing (566 existing + 9 new)

  - **Demo application** (`tests/demos/link_audio_demo.py`)

    - Complete Link + AudioPlayer workflow demonstration

    - Real-time beat/tempo monitoring during playback

    - Visual beat indicators and progress tracking

    - Example of synchronized audio playback

  - **High-Level Python API** (Phase 3 enhancements)

    - Context manager support for `LinkSession` - automatic enable/disable

    - `__enter__` and `__exit__` methods for `with` statement support

    - Exported from main `coremusic` package via `cm.link` module

    - Fully Pythonic API with properties, named arguments, informative `__repr__`

    - 19 additional tests for high-level API patterns

    - High-level demo (`tests/demos/link_high_level_demo.py`) with 6 examples

    - All 594 tests passing (566 existing + 9 AudioPlayer + 19 high-level API)

  - **Link + CoreMIDI Integration** (`src/coremusic/link_midi.py`)

    - `LinkMIDIClock` class - MIDI Clock messages synchronized to Link tempo

      - Sends 24 clock messages per quarter note per MIDI spec

      - Automatic tempo tracking when Link tempo changes

      - Sends MIDI Start/Stop messages

      - Runs in separate thread for realtime performance

    - `LinkMIDISequencer` class - Beat-accurate MIDI event scheduling

      - Schedule MIDI events at specific Link beat positions

      - `schedule_note()` - Schedule notes with automatic note-off

      - `schedule_cc()` - Schedule MIDI CC messages

      - `schedule_event()` - Schedule arbitrary MIDI messages

      - Events kept sorted by beat position

      - Thread-safe event scheduling

    - Time conversion utilities

      - `link_beat_to_host_time()` - Convert Link beats to mach_absolute_time

      - `host_time_to_link_beat()` - Convert host time to Link beats

      - Round-trip conversion with < 0.01 beat accuracy

    - MIDI constants (MIDI_CLOCK, MIDI_START, MIDI_STOP, MIDI_CLOCKS_PER_QUARTER_NOTE)

    - 20 comprehensive tests covering all functionality

    - Interactive demo (`tests/demos/link_midi_demo.py`) with 3 examples

    - All 614 tests passing (594 existing + 20 Link+MIDI integration)

  **Example Usage:**

  ```python
  import coremusic as cm

  # Basic Link usage with context manager
  with cm.link.LinkSession(bpm=120.0) as session:
      state = session.capture_app_session_state()
      print(f"Tempo: {state.tempo:.1f} BPM, Peers: {session.num_peers}")

  # AudioPlayer + Link integration
  with cm.link.LinkSession(bpm=120.0) as session:
      player = cm.AudioPlayer(link_session=session)
      player.load_file("audio.wav")
      player.setup_output()

      # Query Link timing
      timing = player.get_link_timing(quantum=4.0)
      print(f"Beat: {timing['beat']:.2f}, Tempo: {timing['tempo']:.1f} BPM")

      player.play()
      player.start()

  # MIDI Clock synchronized to Link
  from coremusic import link_midi

  client = cm.capi.midi_client_create("MIDI Clock")
  port = cm.capi.midi_output_port_create(client, "Clock Out")
  dest = cm.capi.midi_get_destination(0)

  with cm.link.LinkSession(bpm=120.0) as session:
      clock = link_midi.LinkMIDIClock(session, port, dest)
      clock.start()  # Sends MIDI clock messages
      time.sleep(10)
      clock.stop()

  # Beat-accurate MIDI sequencer
  with cm.link.LinkSession(bpm=120.0) as session:
      seq = link_midi.LinkMIDISequencer(session, port, dest)

      # Schedule notes at Link beat positions
      seq.schedule_note(beat=0.0, channel=0, note=60, velocity=100, duration=0.9)
      seq.schedule_note(beat=1.0, channel=0, note=64, velocity=100, duration=0.9)

      seq.start()
      time.sleep(5)
      seq.stop()
  ```

## [0.1.7]

### Added

- **CoreAudioClock API** - Complete audio/MIDI synchronization and timing services

  - **Low-level C API wrappers** in `capi.pyx`

    - `ca_clock_new()` - Create new clock instances

    - `ca_clock_dispose()` - Resource cleanup

    - `ca_clock_start()` / `ca_clock_stop()` - Playback control

    - `ca_clock_get_play_rate()` / `ca_clock_set_play_rate()` - Speed control

    - `ca_clock_get_current_time()` - Time queries with format support

    - Time format getter functions for seconds, beats, samples, host time

  - **High-level AudioClock class** with context manager support

    - Properties: `play_rate`, `is_running`, `is_disposed`

    - Methods: `start()`, `stop()`, `get_time_seconds()`, `get_time_beats()`, `get_time_samples()`, `get_time_host()`

    - Automatic resource management with `__enter__` and `__exit__`

  - **ClockTimeFormat constants** for time format specifications

    - `HOST_TIME` - mach_absolute_time()

    - `SAMPLES` - Audio sample count

    - `BEATS` - Musical beats

    - `SECONDS` - Seconds

    - `SMPTE_TIME` - SMPTE timecode

  - **Comprehensive test coverage** - 21 tests covering all functionality

    - Low-level API tests (create/dispose, start/stop, play rate, time formats)

    - High-level API tests (context manager, properties, time getters)

    - Timing accuracy verification (normal and half-speed)

    - Error handling and multiple simultaneous clocks

  - **Complete documentation**

    - Sphinx API reference with autodoc integration

    - Code examples in main index and getting started guide

    - Detailed docstrings with RST formatting

  - **Use cases**: DAWs, sequencers, MIDI sync, tempo control, audio/MIDI alignment

  **Example Usage:**

  ```python
  import coremusic as cm

  # High-level API
  with cm.AudioClock() as clock:
      clock.play_rate = 1.0  # Normal speed
      clock.start()

      # Get time in different formats
      seconds = clock.get_time_seconds()
      beats = clock.get_time_beats()
      samples = clock.get_time_samples()

      # Change speed for tempo sync
      clock.play_rate = 0.5  # Half speed

      clock.stop()

  # Low-level API
  import coremusic.capi as capi

  clock_id = capi.ca_clock_new()
  capi.ca_clock_start(clock_id)
  # ... operations ...
  capi.ca_clock_dispose(clock_id)
  ```

- **Full mypy type checking support**

  - Added comprehensive type hints across entire Python codebase

  - Configured strict mypy settings in `pyproject.toml`

  - Fixed all type errors in `scipy_utils.py`, `utilities.py`, `async_io.py`

  - Added `make typecheck` target to Makefile

  - All 516 tests passing with full type safety

- **AudioStreamBasicDescription parsing utility**

  - Added `parse_audio_stream_basic_description()` function to `utilities` module

  - Parses 40-byte ASBD structure from CoreAudio APIs into Python dictionary

  - Returns all format fields: sample_rate, format_id, channels, bit depth, etc.

  - Comprehensive documentation with structure layout and usage examples

  - 3 test cases verifying parsing, validation, and compatibility with OO API

  - Useful for functional API users who need to parse raw format data

  **Example Usage:**

  ```python
  import coremusic as cm
  import coremusic.capi as capi

  file_id = capi.audio_file_open_url("audio.wav")
  format_data = capi.audio_file_get_property(
      file_id,
      capi.get_audio_file_property_data_format()
  )
  asbd = cm.parse_audio_stream_basic_description(format_data)
  print(f"{asbd['sample_rate']} Hz, {asbd['channels_per_frame']} channels")
  capi.audio_file_close(file_id)
  ```

### Fixed

- **Sphinx documentation build warnings** - Eliminated all 41 warnings in documentation build

  - Fixed AudioClock docstring RST formatting (changed markdown code blocks to RST format)

  - Removed autofunction directives for non-exported capi functions

  - Updated API reference to guide users to `coremusic.capi` module for low-level functions

  - Updated audio file documentation examples to use correct import patterns

  - Fixed Makefile documentation targets to properly delegate to docs/Makefile

  - Documentation now builds cleanly with 0 warnings, 0 errors

### Changed

- **Pure Cython Audio Player Implementation** - Replaced C audio player with native Cython implementation

  - **Removed C dependencies**: Eliminated `audio_player.c`, `audio_player.h`, and `audio_player.pxd` files

  - **Simplified build process**: No separate C compilation needed, all audio playback in Cython

  - **Cleaner architecture**: Consistent with existing callback patterns in the codebase

  - **Same functionality**: All `AudioPlayer` methods work identically with same API

  - **Pure Cython render callback**: `audio_player_render_callback()` implemented as `cdef` function with `noexcept nogil`

  - **ExtAudioFile-based loading**: Uses already-wrapped ExtAudioFile APIs for audio file loading

  - **AudioUnit integration**: Native AudioUnit setup and control entirely in Cython

  - **Better maintainability**: All code in one language, easier to understand and extend

  - **Proven pattern**: Follows same approach as existing `audio_queue_output_callback` and `audio_converter_input_callback`

  - **Zero test regressions**: All 516 tests passing after migration

  - **Fixed build configuration**: Updated `setup.py` and `pyproject.toml` for pure Cython build

  **Technical Details:**
  - Render callback handles real-time audio rendering, looping, and playback state

  - Automatic format conversion to 44.1kHz stereo float32

  - Sample-rate conversion and chunked reading for large files

  - Full AudioUnit lifecycle management (initialize, start, stop, cleanup)

  - Proper memory management with automatic buffer cleanup

  **Impact:**
  - **Users**: No API changes - `AudioPlayer` works exactly the same

  - **Developers**: Simpler codebase with better maintainability

  - **Build**: Faster compilation without separate C sources

## [0.1.6]

### Changed

- **Namespace Refactoring** - Separated object-oriented API from functional C API for cleaner, more Pythonic interface

  - **Object-Oriented API is now the primary interface** - All high-level classes available directly from `import coremusic as cm`

  - **Functional C API moved to explicit namespace** - Low-level C functions now require `import coremusic.capi as capi`

  - **Cleaner main namespace** - `coremusic.*` now contains only Pythonic object-oriented classes and utilities

  - **Advanced users retain full access** - Complete functional API still available via `capi` submodule

  - **Re-exported base classes** - `CoreAudioObject` and `AudioPlayer` properly exported from main namespace

  - **Comprehensive migration** - 1,126 functional API calls migrated across 27 files (tests, demos, scripts)

  - **Zero test regressions** - All 516 tests passing after migration

  **Before (intermingled APIs):**

  ```python
  import coremusic as cm

  # Mix of OO and functional APIs in same namespace
  file = cm.AudioFile("audio.wav")  # OO class
  file_id = cm.audio_file_open_url("audio.wav")  # functional C API
  ```

  **After (clean separation):**

  ```python
  import coremusic as cm
  import coremusic.capi as capi

  # Object-oriented API (primary interface)
  file = cm.AudioFile("audio.wav")

  # Functional C API (advanced usage)
  file_id = capi.audio_file_open_url("audio.wav")
  ```

  **Impact:**
  - **Most users** - No changes needed if using OO API (`AudioFile`, `AudioQueue`, `AudioUnit`, etc.)

  - **Advanced users** - Add `import coremusic.capi as capi` and prefix functional calls with `capi.`

  - **SciPy utilities** - Already required explicit import: `import coremusic.scipy_utils as spu`

- Removed auto-import of scipy utilities in `__init__.py`

## [0.1.5]

- First pypi release for python 3.11 - 3.14 inclusive.

### Added

- sphinx docs, tutorials and examples.

- **SciPy Signal Processing Integration** - Seamless integration with SciPy for advanced audio DSP workflows

  - **Filter Design** (`scipy_utils.py`)

    - `design_butterworth_filter()` - Design Butterworth filters (lowpass, highpass, bandpass, bandstop)

    - `design_chebyshev_filter()` - Design Chebyshev Type I filters with configurable ripple

    - Support for all standard filter types with customizable order

  - **Filter Application**

    - `apply_filter()` - Generic filter application with zero-phase filtering option

    - `apply_scipy_filter()` - **NEW** Convenience wrapper accepting scipy.signal filter output directly

    - `apply_lowpass_filter()` - Convenient lowpass filtering

    - `apply_highpass_filter()` - Convenient highpass filtering

    - `apply_bandpass_filter()` - Convenient bandpass filtering

    - Automatic handling of mono and stereo audio

  - **Resampling**

    - `resample_audio()` - High-quality resampling using SciPy

    - Support for both FFT and polyphase methods

    - Automatic multi-channel handling

  - **Spectral Analysis**

    - `compute_spectrum()` - Power spectral density using Welch's method

    - `compute_fft()` - Fast Fourier Transform with windowing

    - `compute_spectrogram()` - Time-frequency analysis

    - Configurable window functions (hann, hamming, blackman, etc.)

  - **AudioSignalProcessor Class** - High-level interface for DSP workflows

    - Method chaining for fluent API (e.g., `.lowpass(1000).normalize().get_audio()`)

    - Built-in methods: `lowpass()`, `highpass()`, `bandpass()`, `resample()`, `normalize()`

    - Integrated spectral analysis: `spectrum()`, `fft()`, `spectrogram()`

    - `reset()` method to restore original audio

  - **SCIPY_AVAILABLE** flag for feature detection

  - **42 comprehensive tests** covering all SciPy functionality (including 7 tests for convenience API)

  - **Demo script** (`tests/demos/demo_scipy_integration.py`) with 6 detailed examples

  - **Complete NumPy/SciPy ecosystem integration** for scientific audio processing

  **Example Usage:**

  ```python
  import coremusic as cm
  import coremusic.scipy_utils as spu

  # Load and process audio
  with cm.AudioFile("audio.wav") as af:
      audio = af.read_as_numpy()
      sr = af.format.sample_rate

  # Use AudioSignalProcessor for chained operations
  processor = spu.AudioSignalProcessor(audio, sr)
  processed = (processor
              .highpass(50)      # Remove rumble
              .lowpass(15000)    # Remove ultrasonic
              .normalize(0.9)    # Normalize
              .get_audio())

  # Or use individual functions
  filtered = spu.apply_lowpass_filter(audio, cutoff=2000, sample_rate=sr)
  resampled = spu.resample_audio(audio, original_rate=sr, target_rate=48000)
  freqs, spectrum = spu.compute_spectrum(audio, sample_rate=sr)

  # Or use scipy.signal filters directly with convenience wrapper
  import scipy.signal
  filtered = spu.apply_scipy_filter(audio, scipy.signal.butter(5, 1000, 'low', fs=sr))
  ```

- **Complex Audio Conversion Support** - Full callback-based AudioConverter API for advanced audio format conversions

  - **Callback Infrastructure** in Cython layer (`src/coremusic/capi.pyx`)

    - `AudioConverterCallbackData` struct for passing data between Python and C callback

    - `audio_converter_input_callback()` - C callback function with `nogil` and `noexcept` for providing input data on demand

    - `audio_converter_fill_complex_buffer()` - Python wrapper for Apple's `AudioConverterFillComplexBuffer` API

    - Proper GIL management for thread-safe operation

    - Safe memory allocation/deallocation with automatic cleanup

  - **Enhanced AudioConverter class** (`src/coremusic/objects.py`)

    - `convert_with_callback()` method supporting all conversion types:

      - Sample rate changes (e.g., 44.1kHz → 48kHz, 48kHz → 96kHz)

      - Bit depth changes (e.g., 16-bit → 24-bit)

      - Channel count changes (stereo ↔ mono)

      - Combined conversions (e.g., 44.1kHz stereo → 48kHz mono)

    - Auto-calculation of output packet count based on sample rate ratio

    - Comprehensive documentation with usage examples

  - **Updated utilities** (`src/coremusic/utilities.py`)

    - `convert_audio_file()` now supports ALL conversion types (previously only channel count)

    - Automatically chooses between simple buffer API and callback API based on conversion type

    - Added `_formats_match()` helper function for format comparison

    - Removed NotImplementedError for complex conversions

  - **Comprehensive test coverage**

    - 6 new tests in `test_objects_audio_converter.py`:

      - Sample rate conversion (44.1kHz ↔ 48kHz)

      - Real file sample rate conversion with verification

      - Combined sample rate and channel conversion

      - Auto output packet count calculation

    - 3 previously skipped tests now enabled in `test_utilities.py`:

      - `test_convert_audio_file_sample_rate`

      - `test_convert_audio_file_bit_depth`

      - `test_convert_audio_file_combined_conversions`

    - All tests passing (474 passed, 36 skipped, 0 failures)

    - Duration preservation verified (< 0.000003s error for 2.743s audio)

  - **Documentation** in `docs/COMPLEX_AUDIO_CONVERSION.md`

    - Complete implementation guide with code examples

    - Technical details on callback mechanism and memory management

    - Usage examples and best practices

    - Implementation status updated

## [0.1.4]

### Added

- **Async I/O Support** - Complete async/await support for non-blocking audio operations

  - `AsyncAudioFile` class for asynchronous file reading with chunk streaming

  - `AsyncAudioQueue` class for non-blocking audio queue operations

  - Async context manager support (`async with`) for automatic resource cleanup

  - Async chunk streaming via `read_chunks_async()` - yields audio data without blocking event loop

  - Async packet reading via `read_packets_async()` for fine-grained control

  - NumPy integration with `read_as_numpy_async()` and `read_chunks_numpy_async()`

  - Executor-based approach using `asyncio.to_thread()` for CPU-bound operations

  - Convenience functions: `open_audio_file_async()`, `create_output_queue_async()`

  - Full backward compatibility - existing synchronous API completely untouched

  - Enables concurrent file processing and integration with modern async frameworks (FastAPI, aiohttp, etc.)

- **Comprehensive async test coverage**

  - `test_async_io.py` - 22 async tests covering all async functionality

  - Tests for async file operations (open, close, context managers)

  - Tests for async packet reading and chunk streaming

  - Tests for concurrent file access and processing pipelines

  - Tests for AudioQueue lifecycle management with async operations

  - Tests for NumPy integration with async streaming

  - Real-world async processing pipeline examples

  - 100% pass rate (22/22 tests passing when NumPy available)

- **Demo script for async I/O** (`demo_async_io.py`)

  - 6 comprehensive examples demonstrating async capabilities

  - Basic async file reading with format inspection

  - Streaming large files in chunks without blocking

  - Async AudioQueue creation and playback control

  - Concurrent file processing (batch operations)

  - Real-world processing pipeline (Read → Analyze → Save)

  - NumPy integration for signal processing workflows

- **High-Level Audio Processing Utilities** - Convenient utilities for common audio tasks

  - `AudioAnalyzer` class for audio analysis operations

    - `detect_silence()` - Detect silence regions in audio files with configurable threshold and duration

    - `get_peak_amplitude()` - Extract peak amplitude from audio files

    - `calculate_rms()` - Calculate RMS (Root Mean Square) amplitude

    - `get_file_info()` - Extract comprehensive file metadata (format, duration, sample rate, etc.)

    - All methods support both file paths and AudioFile objects

    - NumPy integration for efficient audio data processing

  - `AudioFormatPresets` class with common audio format presets

    - `wav_44100_stereo()` - CD quality WAV (44.1kHz, 16-bit, stereo)

    - `wav_44100_mono()` - Mono WAV (44.1kHz, 16-bit, mono)

    - `wav_48000_stereo()` - Pro audio WAV (48kHz, 16-bit, stereo)

    - `wav_96000_stereo()` - High-res WAV (96kHz, 24-bit, stereo)

  - `convert_audio_file()` - Simple file format conversion

    - Supports stereo ↔ mono conversion at same sample rate and bit depth

    - Automatic file copy for exact format matches

    - Raises NotImplementedError for complex conversions (guides users to AudioConverter)

  - `batch_convert()` - Batch convert multiple files with glob patterns

    - Supports custom output directory and file extension

    - Optional progress callback for UI integration

    - Automatic directory creation and file overwrite control

  - `trim_audio()` - Extract time ranges from audio files

    - Supports start and end time specification

    - Preserves audio format during trimming

  - `AudioEffectsChain` class for high-level AUGraph management

    - Pythonic wrapper for audio processing graphs with automatic resource management

    - Methods: `add_effect()`, `add_output()`, `connect()`, `open()`, `initialize()`, `start()`, `stop()`

    - Support for method chaining (e.g., `chain.open().initialize().start()`)

    - Context manager support for automatic cleanup

    - Node management with FourCC-based AudioUnit identification

  - `create_simple_effect_chain()` - Convenience function for quick effect chain creation

  - Comprehensive test coverage with 35 tests (28 passing, 7 skipped)

  - Demo script (`tests/demos/demo_utilities.py`) with 10 working examples

- **AudioUnit Name-Based Discovery** - Find and load AudioUnits by name instead of FourCC codes

  - `find_audio_unit_by_name()` - Search for AudioUnits by name (e.g., 'AUDelay')

    - Returns `AudioComponent` object (can create instances directly)

    - Case-insensitive substring matching by default

    - Returns `None` if no matching AudioUnit found

    - Iterates through all available AudioComponents using CoreAudio's `AudioComponentFindNext`

    - Example: `component = cm.find_audio_unit_by_name('AUDelay')`

  - `list_available_audio_units()` - List all available AudioUnits on the system

    - Returns list of dicts with 'name', 'type', 'subtype', 'manufacturer', 'flags'

    - Optional filtering by FourCC type code (e.g., 'aufx' for audio effects)

    - Discovers 676 AudioUnits on typical macOS system

    - Example: `units = cm.list_available_audio_units(filter_type='aufx')`

  - `get_audiounit_names()` - Get simple list of AudioUnit names

    - Returns list of strings (names only, lightweight)

    - Optional filtering by FourCC type code

    - Example: `names = cm.get_audiounit_names()`

  - `AudioEffectsChain.add_effect_by_name()` - Add effects to chain by name

    - Convenience method that automatically finds and adds AudioUnits

    - Example: `delay_node = chain.add_effect_by_name('AUDelay')`

  - Low-level C API wrappers in `src/coremusic/capi.pyx`:

    - `audio_component_copy_name()` - Get human-readable AudioComponent name

    - `audio_component_get_description()` - Get AudioComponentDescription

    - Updated `audio_component_find_next()` with iteration support

  - Proper CoreFoundation memory management with CFRelease for CFStringRef

  - Comprehensive test coverage with 11 tests (100% passing)

  - Documentation in `docs/audiounit_name_lookup.md` with usage examples

  - Demo examples in `tests/demos/demo_utilities.py` (Example 10)

### Fixed

- **Music device test fixture** - Improved error handling for component instantiation

  - Added graceful skip when `AudioComponentInstanceNew` returns status -128

  - Status -128 indicates macOS security restrictions preventing instantiation

  - Tests now properly skip instead of erroring when components cannot be instantiated

  - Improved test robustness across different macOS security configurations

  - Affects `test_audiotoolbox_music_device.py` fixture for music device unit tests

## [0.1.3]

### Added

- **AudioConverter API** - Complete audio format conversion framework

  - Functional API with 13 wrapper functions for AudioConverter operations

  - `audio_converter_new()`, `audio_converter_dispose()`, `audio_converter_convert_buffer()`

  - `audio_converter_get_property()`, `audio_converter_set_property()`, `audio_converter_reset()`

  - 6 property ID getter functions for converter configuration

  - Object-oriented `AudioConverter` class with automatic resource management

  - Context manager support for safe resource cleanup

  - Support for stereo↔mono conversion, bit depth changes, and format conversions

- **ExtendedAudioFile API** - High-level audio file I/O with automatic format conversion

  - Functional API with 14 wrapper functions for ExtendedAudioFile operations

  - `extended_audio_file_open_url()`, `extended_audio_file_create_with_url()`

  - `extended_audio_file_read()`, `extended_audio_file_write()`, `extended_audio_file_dispose()`

  - `extended_audio_file_get_property()`, `extended_audio_file_set_property()`

  - 7 property ID getter functions for file format access

  - Object-oriented `ExtendedAudioFile` class with context manager support

  - Automatic format conversion on read/write via client format property

  - Simplified file I/O compared to lower-level AudioFile API

- **AUGraph API** - Audio Unit graph framework for managing and connecting multiple AudioUnits

  - Functional API with 21 wrapper functions for AUGraph operations

  - `au_graph_new()`, `au_graph_dispose()`, `au_graph_open()`, `au_graph_close()`

  - `au_graph_initialize()`, `au_graph_uninitialize()`, `au_graph_start()`, `au_graph_stop()`

  - `au_graph_add_node()`, `au_graph_remove_node()`, `au_graph_get_node_count()`

  - `au_graph_connect_node_input()`, `au_graph_disconnect_node_input()`, `au_graph_update()`

  - 3 state query functions: `au_graph_is_open()`, `au_graph_is_initialized()`, `au_graph_is_running()`

  - CPU load monitoring: `au_graph_get_cpu_load()`, `au_graph_get_max_cpu_load()`

  - 5 error code getter functions for AUGraph-specific errors

  - Object-oriented `AUGraph` class with automatic resource management

  - Context manager support for safe graph lifecycle management

  - Node management with `AudioComponentDescription` integration

  - Connection management for building audio processing graphs

  - Method chaining support for fluent API (e.g., `graph.open().initialize()`)

  - Properties for state queries: `is_open`, `is_initialized`, `is_running`, `cpu_load`, `node_count`

- **Comprehensive test coverage** for new APIs

  - `test_audiotoolbox_audio_converter.py` - 12 functional API tests

  - `test_audiotoolbox_extended_audio_file.py` - 14 functional API tests

  - `test_objects_audio_converter.py` - 29 object-oriented wrapper tests

  - `test_augraph.py` - 16 AUGraph tests (4 functional, 11 OO, 1 integration)

  - Tests cover creation, conversion, I/O operations, property access, error handling

  - Real-world testing with actual audio files

- **Exception hierarchy** expanded

  - Added `AudioConverterError` for converter-specific exceptions

  - Added `AUGraphError` for graph operation exceptions

  - Proper error propagation with detailed error messages

### Changed

- Enhanced `AudioFormat` class integration with converter APIs

- Improved error handling consistency across audio conversion operations

### Fixed

- **Critical fix for AudioDevice string properties** - Added proper CFStringRef handling

  - Previously, `audio_object_get_property_data()` returned raw CFStringRef pointers instead of actual string content

  - Added new `audio_object_get_property_string()` function that properly dereferences CFStringRef using CoreFoundation APIs

  - Device names, UIDs, and manufacturer strings now correctly use CFStringGetCString for stable, proper string extraction

  - Fixes unstable device name/UID issues where properties returned random garbage on each read

  - All AudioDevice string properties (name, uid, manufacturer, model_uid) now work correctly

- Fixed UID string handling in `AudioDevice._get_property_string()` to strip both leading and trailing null bytes (changed from `.rstrip('\x00')` to `.strip('\x00')`)

- Improved `test_audio_device_manager_find_by_uid` test resilience to handle devices with inconsistent UID encoding

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

- Enhanced package structure with proper **init**.py imports

- Comprehensive test coverage for object-oriented APIs

  - Added tests for AudioFile, AudioUnit, AudioQueue OO classes

  - Added MIDI object-oriented API tests

  - Added comprehensive integration tests

### Changed

- Updated README with dual API examples and migration guide

- Enhanced project description to reflect comprehensive framework coverage

- Improved developer experience documentation

### Fixed

- Resource management issues with automatic cleanup via Cython **dealloc**

- Memory leaks in audio operations through proper disposal patterns

---

## [0.1.0] - Previous Release

### Added

- Added namespaces to cimports

- Added a bunch of tests

- Renamed project from `cycoreaudio` to `coremusic`

- Added CoreMIDI wrapper

- Added CoreAudio wrapper
