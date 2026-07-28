"""Execute every documentation example.

The snippets in `docs/` are not written inline: they are included from runnable
programs under `examples/`, and this module runs each of them. A snippet that
cannot be executed cannot be published, which is what keeps the documentation
from drifting away from the API again.

Each example runs in its own temporary directory seeded with the sample assets
the examples refer to (`audio.wav`, `song.mid`, ...), so the code shown to a
reader is the code that runs - no fixture plumbing leaks into the snippet.
"""

import ast
import os
import re
import shutil
import subprocess
import sys

import pytest
from conftest import AMEN_WAV_PATH, DEMO_MID_PATH

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")

# Assets an example may open by plain relative name.
ASSETS = {
    "audio.wav": AMEN_WAV_PATH,
    "song.wav": AMEN_WAV_PATH,
    "input.wav": AMEN_WAV_PATH,
    "drums.wav": AMEN_WAV_PATH,
    "song.mid": DEMO_MID_PATH,
    "input.mid": DEMO_MID_PATH,
}

# Examples need a moment for CoreAudio/CoreMIDI round trips; none should be
# anywhere near this slow.
TIMEOUT = 120

# The MIDI server intermittently refuses to hand out a client under load. The
# rest of the suite skips on this too; matching narrowly keeps it from hiding
# a real failure.
MIDI_UNAVAILABLE = re.compile(
    r"MIDIClientCreate failed|MIDI services (?:not available|unavailable)"
)


def find_examples():
    """Every runnable example, as repo-relative paths."""
    found = []
    for dirpath, _dirnames, filenames in os.walk(EXAMPLES_DIR):
        for filename in sorted(filenames):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue
            path = os.path.join(dirpath, filename)
            found.append(os.path.relpath(path, REPO_ROOT))
    return sorted(found)


EXAMPLES = find_examples()


def test_examples_exist():
    """Guard against the discovery glob silently matching nothing."""
    assert EXAMPLES, "no examples found under examples/"


@pytest.mark.parametrize("example", EXAMPLES)
def test_example_exercises_what_it_defines(example):
    """An example that only defines a function proves nothing by running.

    Five examples originally passed `test_example_runs` while executing none
    of the code they showed, and two of those turned out to be broken - one
    passed the wrong arity to `AudioConverter.convert()`, another re-read
    packet 0 on every loop iteration. Running has to mean running.
    """
    tree = ast.parse(open(os.path.join(REPO_ROOT, example)).read())
    top_level_defs = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not top_level_defs:
        return  # straight-line script: running it is the exercise

    referenced = {
        node.id
        for sibling in tree.body
        if sibling not in top_level_defs
        for node in ast.walk(sibling)
        if isinstance(node, ast.Name)
    }

    unused = [d.name for d in top_level_defs if d.name not in referenced]
    assert len(unused) < len(top_level_defs), (
        f"{example} defines {', '.join(unused)} but never calls anything it "
        f"defines, so running it does not exercise the snippet"
    )


@pytest.mark.slow
@pytest.mark.parametrize("example", EXAMPLES)
def test_example_runs(example, tmp_path):
    """Run an example end to end and require a clean exit.

    Marked slow: the examples play, record, and wait on MIDI in real time, so
    the whole set takes minutes. `make test` skips them and relies on
    `test_doc_snippets.py` to catch stale names; `make test-all` runs them.
    """
    for name, source in ASSETS.items():
        if os.path.exists(source):
            shutil.copy(source, tmp_path / name)

    env = dict(os.environ)
    env["PYTHONPATH"] = (
        os.path.join(REPO_ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    )

    def run():
        return subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, example)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )

    result = run()
    if result.returncode != 0:
        # Running a couple of hundred processes back to back, each creating
        # and disposing MIDI clients, occasionally trips a transient failure
        # from the MIDI server. Retry once: a real breakage fails twice.
        result = run()

    if result.returncode != 0:
        if MIDI_UNAVAILABLE.search(result.stderr):
            pytest.skip(f"MIDI services unavailable: {result.stderr.strip()[-200:]}")
        pytest.fail(
            f"{example} exited {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
