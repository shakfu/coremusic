"""Shell completion support for coremusic CLI.

This module provides shell completion scripts for bash, zsh, and fish.

Command and subcommand names are read from the argparse parser built by
``coremusic.cli.main.build_parser`` so the generated scripts cannot drift from
the registered CLI.

Usage:
    # Bash (add to ~/.bashrc)
    eval "$(coremusic completion bash)"

    # Zsh (add to ~/.zshrc)
    eval "$(coremusic completion zsh)"

    # Fish (add to ~/.config/fish/config.fish)
    coremusic completion fish | source

    # Or save to a file:
    coremusic completion bash > /etc/bash_completion.d/coremusic
    coremusic completion zsh > ~/.zfunc/_coremusic
    coremusic completion fish > ~/.config/fish/completions/coremusic.fish
"""

from __future__ import annotations

import argparse

# Common options
GLOBAL_OPTIONS = ["--help", "--version", "--json"]

# Top-level commands whose positional arguments are audio or MIDI file paths
FILE_COMMANDS = ["audio", "analyze", "convert", "sequence"]

# Subcommands of `midi` that take a MIDI file path
MIDI_FILE_SUBCOMMANDS = ["info", "play", "quantize"]

FILE_PATTERNS = ["wav", "aiff", "aif", "mp3", "m4a", "caf", "flac", "mid", "midi"]


def _subparser_action(
    parser: argparse.ArgumentParser,
) -> argparse._SubParsersAction[argparse.ArgumentParser] | None:
    """Return the subparser action of a parser, or None if it has no subcommands."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _choice_help(
    action: argparse._SubParsersAction[argparse.ArgumentParser], name: str
) -> str:
    """Return the help text argparse recorded for one subparser choice."""
    for choice_action in action._choices_actions:
        if choice_action.dest == name:
            return choice_action.help or ""
    return ""


def get_command_tree() -> dict[str, tuple[str, dict[str, str]]]:
    """Introspect the CLI parser for command names, help text, and subcommands.

    Returns a mapping of command name to a ``(help, subcommands)`` pair, where
    ``subcommands`` maps subcommand name to its help text.
    """
    from .main import build_parser

    action = _subparser_action(build_parser())
    if action is None:  # pragma: no cover - the CLI always has subcommands
        return {}

    tree: dict[str, tuple[str, dict[str, str]]] = {}
    for name, subparser in action.choices.items():
        sub_action = _subparser_action(subparser)
        subs: dict[str, str] = {}
        if sub_action is not None:
            subs = {
                sub_name: _choice_help(sub_action, sub_name)
                for sub_name in sub_action.choices
            }
        tree[name] = (_choice_help(action, name), subs)
    return tree


def _escape_zsh(text: str) -> str:
    """Escape text for use inside a zsh completion description."""
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("]", "\\]")


def _escape_fish(text: str) -> str:
    """Escape text for use inside a fish single-quoted description."""
    return text.replace("\\", "\\\\").replace("'", "\\'")


def get_bash_completion() -> str:
    """Generate bash completion script."""
    tree = get_command_tree()
    commands = " ".join(tree)
    subcommands = "\n".join(
        f'                {cmd}) COMPREPLY=($(compgen -W "{" ".join(subs)}" -- "$cur")) ;;'
        for cmd, (_, subs) in tree.items()
        if subs
    )
    file_commands = "|".join(FILE_COMMANDS)
    midi_file_subs = " ".join(MIDI_FILE_SUBCOMMANDS)
    # bash extglob alternation uses "|"; zsh brace expansion uses ","
    patterns = "|".join(FILE_PATTERNS)

    return f"""# Bash completion for coremusic
# Add to ~/.bashrc: eval "$(coremusic completion bash)"

_coremusic_completion() {{
    local cur prev words cword
    _init_completion || return

    local commands="{commands}"
    local global_opts="{" ".join(GLOBAL_OPTIONS)}"

    case ${{cword}} in
        1)
            COMPREPLY=($(compgen -W "$commands $global_opts" -- "$cur"))
            ;;
        2)
            case "${{words[1]}}" in
{subcommands}
                *) COMPREPLY=() ;;
            esac
            ;;
        *)
            # File completion for audio/midi file arguments
            case "${{words[1]}}" in
                {file_commands})
                    _filedir '@({patterns})'
                    ;;
                midi)
                    if [[ " {midi_file_subs} " == *" ${{words[2]}} "* ]]; then
                        _filedir '@(mid|midi)'
                    fi
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            ;;
    esac
}}

complete -F _coremusic_completion coremusic
"""


def get_zsh_completion() -> str:
    """Generate zsh completion script."""
    tree = get_command_tree()
    command_list = "\n".join(
        f'                "{cmd}:{_escape_zsh(help_text)}"'
        for cmd, (help_text, _) in tree.items()
    )
    cmd_cases = "\n".join(
        f'            {cmd}) _values "subcommand" {values} ;;'
        for cmd, values in (
            (
                cmd,
                " ".join(
                    f"'{sub}[{_escape_zsh(sub_help)}]'"
                    for sub, sub_help in subs.items()
                ),
            )
            for cmd, (_, subs) in tree.items()
            if subs
        )
    )
    patterns = ",".join(FILE_PATTERNS)

    return f"""#compdef coremusic
# Zsh completion for coremusic
# Add to ~/.zshrc: eval "$(coremusic completion zsh)"
# Or save to ~/.zfunc/_coremusic and add: fpath=(~/.zfunc $fpath); autoload -Uz compinit; compinit

_coremusic() {{
    local line state

    _arguments -C \\
        "--help[Show help message]" \\
        "--version[Show version]" \\
        "--json[Output in JSON format]" \\
        "1: :->command" \\
        "*::arg:->args"

    case "$state" in
        command)
            local commands=(
{command_list}
            )
            _describe "command" commands
            ;;
        args)
            case ${{line[1]}} in
{cmd_cases}
            esac
            ;;
    esac
}}

# Audio file patterns for completion
zstyle ':completion:*:*:coremusic:*' file-patterns \\
    '*.{{{patterns}}}:audio-files:audio files' \\
    '*(-/):directories:directories'

_coremusic "$@"
"""


def get_fish_completion() -> str:
    """Generate fish completion script."""
    tree = get_command_tree()

    lines = [
        "# Fish completion for coremusic",
        "# Add to ~/.config/fish/config.fish: coremusic completion fish | source",
        "# Or save to ~/.config/fish/completions/coremusic.fish",
        "",
        "# Disable file completion by default",
        "complete -c coremusic -f",
        "",
        "# Global options",
        "complete -c coremusic -s h -l help -d 'Show help message'",
        "complete -c coremusic -l version -d 'Show version'",
        "complete -c coremusic -l json -d 'Output in JSON format'",
        "",
        "# Main commands",
    ]

    for cmd, (help_text, _) in tree.items():
        lines.append(
            f"complete -c coremusic -n '__fish_use_subcommand' "
            f"-a {cmd} -d '{_escape_fish(help_text)}'"
        )

    lines.append("")
    lines.append("# Subcommands")

    for cmd, (_, subs) in tree.items():
        for sub, sub_help in subs.items():
            lines.append(
                f"complete -c coremusic -n '__fish_seen_subcommand_from {cmd}' "
                f"-a {sub} -d '{_escape_fish(sub_help)}'"
            )

    lines.append("")
    lines.append("# File completion for specific commands")
    lines.append(
        f"complete -c coremusic -n '__fish_seen_subcommand_from {' '.join(FILE_COMMANDS)}' "
        "-F -d 'Audio file'"
    )
    lines.append(
        f"complete -c coremusic -n '__fish_seen_subcommand_from {' '.join(MIDI_FILE_SUBCOMMANDS)}' "
        "-F -d 'MIDI file'"
    )

    return "\n".join(lines)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register completion command."""
    parser = subparsers.add_parser(
        "completion",
        help="Generate shell completion scripts",
        description="Generate shell completion scripts for bash, zsh, or fish.",
    )
    parser.add_argument(
        "shell",
        choices=["bash", "zsh", "fish"],
        help="Shell type (bash, zsh, or fish)",
    )
    parser.set_defaults(func=handle_completion)


def handle_completion(args: argparse.Namespace) -> int:
    """Handle completion command."""
    generators = {
        "bash": get_bash_completion,
        "zsh": get_zsh_completion,
        "fish": get_fish_completion,
    }

    script = generators[args.shell]()
    print(script)
    return 0
