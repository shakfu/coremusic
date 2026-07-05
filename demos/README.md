# CoreMusic demos

Small, self-contained scripts that each demonstrate one capability of the
library. They use only the standard library plus `coremusic` (no NumPy
required) and are meant to be read as much as run.

Run them from the repository root after installing the package (`make sync`):

```bash
python demos/host_au_chain.py
python demos/render_midi_to_wav.py
python demos/output_stream_tone.py
python demos/link_sequencer.py
```

Or run all four in sequence, writing their output to `build/demos-output/`:

```bash
make demos
```

All are macOS-only (they rely on CoreAudio / AudioUnits / Ableton Link) and
accept `-h/--help` for options.

| Script | What it shows | Needs |
|--------|---------------|-------|
| `host_au_chain.py` | Process a WAV through a chain of AudioUnit effects (AUDelay -> AUMatrixReverb) and write the result. | Input WAV; Apple effect AudioUnits (built in) |
| `render_midi_to_wav.py` | Render a MIDI file to WAV through an instrument AudioUnit (`DLSMusicDevice`). | A MIDI file; an instrument AudioUnit |
| `output_stream_tone.py` | Play a generated sine tone in real time via a pull-generator output stream. | An audio output device |
| `link_sequencer.py` | A step sequencer whose timing is locked to the Ableton Link shared beat grid. | Output device (falls back to printing the timeline) |

## Notes

- **Effect / instrument names** are matched case-insensitively as substrings.
  List what is installed with:

  ```python
  from coremusic.audio.utilities import get_audiounit_names
  print(get_audiounit_names(filter_type="aufx"))  # effects
  print(get_audiounit_names(filter_type="aumu"))  # instruments
  ```

  or run `coremusic doctor`.

- **`host_au_chain.py`** and **`render_midi_to_wav.py`** produce output files
  (`out_chain.wav`, `out_midi.wav`) and need no audio hardware, so they work in
  headless/offline environments.

- **`link_sequencer.py`** tempo- and phase-syncs with any other Link app on the
  local network. Start Ableton Live (Link enabled) or a second copy of the
  script to watch them lock together. Use `--print-only` to see the live Link
  timeline without playing audio.

- Default input assets come from `tests/data/` (`wav/amen.wav`,
  `midi/demo.mid`); pass your own paths as positional arguments.
