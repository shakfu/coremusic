"""Check that every demo still loads and parses its arguments.

`tests/demos/` used to hold three dozen scripts that nothing ran. Seventeen of
them had quietly stopped working - importing a module that no longer existed,
opening `tests/amen.wav` after the test data moved, calling a plotter method
that had been renamed. They were deleted, and the handful worth keeping moved
into `demos/`.

Running each demo for real belongs in `make demos`, which needs audio hardware
and takes a while. This is the cheap half: `--help` imports the module and
exercises its argument parser, which is enough to catch the rot above.
"""

import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMOS_DIR = os.path.join(REPO_ROOT, "demos")

DEMOS = sorted(
    f for f in os.listdir(DEMOS_DIR) if f.endswith(".py") and not f.startswith("_")
)


def test_demos_exist():
    """Guard against the listing silently matching nothing."""
    assert DEMOS, "no demos found under demos/"


@pytest.mark.parametrize("demo", DEMOS)
def test_demo_help(demo):
    """`--help` must work: it imports the module and builds the parser."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO_ROOT, "src") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )

    result = subprocess.run(
        [sys.executable, os.path.join(DEMOS_DIR, demo), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"demos/{demo} --help exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "usage" in result.stdout.lower()


@pytest.mark.parametrize("demo", DEMOS)
def test_demo_is_documented(demo):
    """Every demo is listed in demos/README.md, so none goes unmentioned."""
    readme = open(os.path.join(DEMOS_DIR, "README.md")).read()
    assert demo in readme, f"demos/{demo} is not mentioned in demos/README.md"
