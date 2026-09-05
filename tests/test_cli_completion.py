#!/usr/bin/env python3
"""Tests for the `coremusic completion` shell script generators.

These tests exist to catch drift between the argparse registrations and the
names advertised by the generated completion scripts.
"""

import argparse
import re
import shutil
import subprocess

import pytest

from coremusic.cli import completion
from coremusic.cli.main import build_parser

GENERATORS = {
    "bash": completion.get_bash_completion,
    "zsh": completion.get_zsh_completion,
    "fish": completion.get_fish_completion,
}


def registered_commands() -> dict[str, list[str]]:
    """Read command and subcommand names straight from the argparse parser."""
    parser = build_parser()
    action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    tree: dict[str, list[str]] = {}
    for name, subparser in action.choices.items():
        sub_action = next(
            (
                a
                for a in subparser._actions
                if isinstance(a, argparse._SubParsersAction)
            ),
            None,
        )
        tree[name] = list(sub_action.choices) if sub_action else []
    return tree


def test_command_tree_matches_parser():
    tree = completion.get_command_tree()
    expected = registered_commands()
    assert list(tree) == list(expected)
    for name, subs in expected.items():
        assert list(tree[name][1]) == subs


def test_command_tree_has_help_text():
    for name, (help_text, subs) in completion.get_command_tree().items():
        assert help_text, f"command {name} has no help text"
        for sub, sub_help in subs.items():
            assert sub_help, f"subcommand {name} {sub} has no help text"


def test_registry_covers_known_commands():
    """Guard the specific names that previously drifted out of the scripts."""
    tree = registered_commands()
    assert "device" in tree
    assert "devices" not in tree
    assert {"doctor", "completion"} <= set(tree)
    assert {"default", "monitor"} <= set(tree["device"])
    assert {"list", "quantize", "panic"} <= set(tree["midi"])


@pytest.mark.parametrize("shell", sorted(GENERATORS))
def test_every_command_appears_in_script(shell):
    script = GENERATORS[shell]()
    for name in registered_commands():
        assert re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", script), (
            f"{shell} completion omits command {name!r}"
        )


def test_bash_lists_commands_and_subcommands():
    script = completion.get_bash_completion()
    tree = registered_commands()
    assert f'local commands="{" ".join(tree)}"' in script
    for name, subs in tree.items():
        if subs:
            assert f'{name}) COMPREPLY=($(compgen -W "{" ".join(subs)}"' in script


def test_zsh_lists_commands_and_subcommands():
    script = completion.get_zsh_completion()
    for name, subs in registered_commands().items():
        assert f'"{name}:' in script
        for sub in subs:
            assert f"'{sub}[" in script


def test_fish_lists_commands_and_subcommands():
    script = completion.get_fish_completion()
    for name, subs in registered_commands().items():
        assert f"-n '__fish_use_subcommand' -a {name} -d '" in script
        for sub in subs:
            assert f"-n '__fish_seen_subcommand_from {name}' -a {sub} -d '" in script


def test_file_completion_commands_are_registered():
    tree = registered_commands()
    assert set(completion.FILE_COMMANDS) <= set(tree)
    assert set(completion.MIDI_FILE_SUBCOMMANDS) <= set(tree["midi"])


def test_bash_uses_extglob_alternation_for_patterns():
    script = completion.get_bash_completion()
    assert f"'@({'|'.join(completion.FILE_PATTERNS)})'" in script
    assert "," not in re.search(r"_filedir '@\(([^)]*)\)'", script).group(1)


def test_zsh_uses_brace_expansion_for_patterns():
    script = completion.get_zsh_completion()
    assert f"'*.{{{','.join(completion.FILE_PATTERNS)}}}:audio-files" in script


def test_escaping_neutralises_shell_metacharacters():
    assert completion._escape_zsh("a:b]c") == "a\\:b\\]c"
    assert completion._escape_fish("it's") == "it\\'s"


@pytest.mark.parametrize("shell,binary", [("bash", "bash"), ("zsh", "zsh")])
def test_generated_script_parses(shell, binary, tmp_path):
    executable = shutil.which(binary)
    if executable is None:
        pytest.skip(f"{binary} not installed")
    path = tmp_path / f"completion.{shell}"
    path.write_text(GENERATORS[shell]())
    result = subprocess.run(
        [executable, "-n", str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("shell", sorted(GENERATORS))
def test_handle_completion_prints_script(shell, capsys):
    args = argparse.Namespace(shell=shell)
    assert completion.handle_completion(args) == 0
    assert capsys.readouterr().out.strip() == GENERATORS[shell]().strip()
