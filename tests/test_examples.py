"""Execute every documentation example.

The snippets in `docs/` are not written inline: they are included from runnable
programs under `examples/`, and this module runs each of them. A snippet that
cannot be executed cannot be published, which is what keeps the documentation
from drifting away from the API again.

Each example runs in its own temporary directory seeded with the sample assets
the examples refer to (`audio.wav`, `song.mid`, ...), so the code shown to a
reader is the code that runs - no fixture plumbing leaks into the snippet.
"""

import os
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
    "input.wav": AMEN_WAV_PATH,
    "drums.wav": AMEN_WAV_PATH,
    "song.mid": DEMO_MID_PATH,
    "input.mid": DEMO_MID_PATH,
}

# Examples need a moment for CoreAudio/CoreMIDI round trips; none should be
# anywhere near this slow.
TIMEOUT = 120


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
    env["PYTHONPATH"] = os.path.join(REPO_ROOT, "src") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )

    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, example)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )

    if result.returncode != 0:
        pytest.fail(
            f"{example} exited {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
