"""Execute the code blocks in README.md.

The README is rendered by GitHub, which does not understand the `--8<--`
includes the doc site uses, so its code has to stay inline. Rather than keep a
copy under `examples/` that would drift, this runs the README's own text: each
python block is extracted and executed as a script, in a temporary directory
seeded with the sample media the blocks refer to.

Blocks are therefore expected to stand on their own - if one needs a variable,
it has to open the file itself.
"""

import os
import re
import subprocess
import sys

import pytest
from conftest import AMEN_WAV_PATH, DEMO_MID_PATH
from test_examples import MIDI_UNAVAILABLE

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(REPO_ROOT, "README.md")

PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)

ASSETS = {
    "audio.wav": AMEN_WAV_PATH,
    "song.wav": AMEN_WAV_PATH,
    "input.wav": AMEN_WAV_PATH,
    "song.mid": DEMO_MID_PATH,
    "input.mid": DEMO_MID_PATH,
}

# Hardware the block needs may simply not be there, on a CI runner or a
# headless machine. Matching narrowly keeps this from hiding a real failure.
NO_HARDWARE = re.compile(
    r"No (?:audio )?(?:output|input) device|no default output device"
    r"|Failed to (?:setup|set up) output|Plugin '[^']*' not found",
    re.IGNORECASE,
)

TIMEOUT = 120


def readme_blocks():
    text = open(README).read()
    blocks = []
    for match in PYTHON_BLOCK.finditer(text):
        line = text[: match.start()].count("\n") + 1
        blocks.append((line, match.group(1)))
    return blocks


BLOCKS = readme_blocks()


def test_readme_has_blocks():
    """Guard against the block regex silently matching nothing."""
    assert BLOCKS, "no python blocks found in README.md"


@pytest.mark.slow
@pytest.mark.parametrize("line,source", BLOCKS, ids=[f"L{n}" for n, _ in BLOCKS])
def test_readme_block_runs(line, source, tmp_path):
    """Run one README block and require a clean exit."""
    for name, asset in ASSETS.items():
        if os.path.exists(asset):
            (tmp_path / name).write_bytes(open(asset, "rb").read())

    script = tmp_path / f"readme_L{line}.py"
    script.write_text(source)

    env = dict(os.environ)
    env["PYTHONPATH"] = (
        os.path.join(REPO_ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    )

    def run():
        return subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )

    result = run()
    if result.returncode != 0:
        result = run()  # transient MIDI/CoreAudio failures; see test_examples

    if result.returncode != 0:
        if MIDI_UNAVAILABLE.search(result.stderr) or NO_HARDWARE.search(result.stderr):
            pytest.skip(f"hardware unavailable: {result.stderr.strip()[-200:]}")
        pytest.fail(
            f"README.md block at line {line} exited {result.returncode}\n"
            f"--- source ---\n{source}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
