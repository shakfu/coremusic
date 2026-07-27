# Documentation Examples

Every code example in the published documentation lives here as a runnable
program. The doc pages do not contain the code; they include it:

```markdown
--8<-- "examples/tutorials/midi_basics/send_note.py:example"
```

The point is that a snippet cannot be published unless it runs.
`tests/test_examples.py` executes every file in this directory and requires a
clean exit, so an example that references a renamed class or a method that no
longer exists fails the test suite rather than reaching a reader.

## Layout

One directory per doc page, mirroring `docs/`:

```
examples/
  quickstart/            # docs/quickstart.md
  getting_started/       # docs/getting_started.md
  guides/<page>/         # docs/guides/<page>.md
  tutorials/<page>/      # docs/tutorials/<page>.md
  cookbook/<page>/       # docs/cookbook/<page>.md
  api/<page>/            # docs/api/<page>.md
  readme/                # README.md
```

## Writing an example

An example is an ordinary script that runs top to bottom and exits 0:

```python
#!/usr/bin/env python3
"""One line saying what this shows."""

# --8<-- [start:example]
from coremusic.audio import AudioFile

with AudioFile("audio.wav") as f:
    print(f"{f.duration:.2f}s at {f.format.sample_rate}Hz")
# --8<-- [end:example]
```

- The `--8<-- [start:...]` / `[end:...]` markers delimit what the doc shows.
  The shebang, the docstring, and any harness-only code stay out of the
  published snippet. A file may define several named sections when a page
  walks through one program in stages.
- Refer to input files by plain relative name: `audio.wav`, `input.wav`,
  `drums.wav`, `song.mid`, `input.mid`. The test harness runs each example in
  a temporary directory seeded with those files, so the snippet stays free of
  fixture plumbing. Write output files into the working directory.
- Never block. No `while True`, no unbounded `input()`, no sleeping for more
  than a moment. Where a real program would loop forever, loop for a fixed
  number of iterations or a short deadline and say so in a comment.
- Degrade gracefully. Hardware may be absent: no MIDI destination, no audio
  input, no plugin of a given name. Check, report, and exit 0 rather than
  raising - and prefer checks that a reader should be writing anyway.

## Running

```bash
# One example
python examples/tutorials/midi_basics/send_note.py

# All of them, as the test suite does
pytest tests/test_examples.py
```

Examples run against the installed package. After changing Cython sources,
rebuild with `make build` first.
