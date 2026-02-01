"""Shell completion support for coremusic CLI.

This module provides shell completion scripts for bash, zsh, and fish.

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

# Commands and their subcommands for completion
COMMANDS = {
    "audio": ["play", "record", "info", "duration", "metadata"],
    "devices": ["list", "info", "volume", "mute", "set-default"],
    "plugin": ["list", "find", "info", "params", "process", "render", "preset"],
    "analyze": ["tempo", "key", "spectrum", "loudness", "onsets", "peak"],
    "convert": ["file", "batch", "normalize", "trim"],
    "midi": ["devices", "input", "output", "file", "record"],
    "sequence": ["info", "play", "tracks"],
}

# Common options
GLOBAL_OPTIONS = ["--help", "--version", "--json"]


def get_bash_completion() -> str:
    """Generate bash completion script."""
    commands = " ".join(COMMANDS.keys())
    subcommands = "\n".join(
        f'        {cmd}) COMPREPLY=($(compgen -W "{" ".join(subs)}" -- "$cur")) ;;'
        for cmd, subs in COMMANDS.items()
    )

    return f'''# Bash completion for coremusic
# Add to ~/.bashrc: eval "$(coremusic completion bash)"

_coremusic_completion() {{
    local cur prev words cword
    _init_completion || return

    local commands="{commands}"
    local global_opts="--help --version --json"

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
                audio|analyze|convert|sequence)
                    _filedir '@(wav|aiff|aif|mp3|m4a|caf|flac|mid|midi)'
                    ;;
                midi)
                    if [[ "${{words[2]}}" == "file" ]]; then
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
'''


def get_zsh_completion() -> str:
    """Generate zsh completion script."""
    cmd_cases = "\n".join(
        f'            {cmd}) _values "subcommand" {" ".join(subs)} ;;'
        for cmd, subs in COMMANDS.items()
    )

    return f'''#compdef coremusic
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
                "audio:Audio file operations"
                "devices:Audio device management"
                "plugin:AudioUnit plugin operations"
                "analyze:Audio analysis commands"
                "convert:Audio conversion commands"
                "midi:MIDI operations"
                "sequence:MIDI sequence operations"
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
    '*.{{wav,aiff,aif,mp3,m4a,caf,flac,mid,midi}}:audio-files:audio files' \\
    '*(-/):directories:directories'

_coremusic "$@"
'''


def get_fish_completion() -> str:
    """Generate fish completion script."""
    lines = [
        "# Fish completion for coremusic",
        '# Add to ~/.config/fish/config.fish: coremusic completion fish | source',
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

    cmd_descriptions = {
        "audio": "Audio file operations",
        "devices": "Audio device management",
        "plugin": "AudioUnit plugin operations",
        "analyze": "Audio analysis commands",
        "convert": "Audio conversion commands",
        "midi": "MIDI operations",
        "sequence": "MIDI sequence operations",
    }

    for cmd, desc in cmd_descriptions.items():
        lines.append(
            f"complete -c coremusic -n '__fish_use_subcommand' -a {cmd} -d '{desc}'"
        )

    lines.append("")
    lines.append("# Subcommands")

    subcmd_descriptions = {
        "audio": {
            "play": "Play an audio file",
            "record": "Record audio",
            "info": "Show file info",
            "duration": "Get file duration",
            "metadata": "Show file metadata",
        },
        "devices": {
            "list": "List audio devices",
            "info": "Show device info",
            "volume": "Get/set volume",
            "mute": "Get/set mute state",
            "set-default": "Set default device",
        },
        "plugin": {
            "list": "List plugins",
            "find": "Find plugin by name",
            "info": "Show plugin info",
            "params": "List parameters",
            "process": "Process audio",
            "render": "Render MIDI",
            "preset": "Preset operations",
        },
        "analyze": {
            "tempo": "Detect tempo",
            "key": "Detect key",
            "spectrum": "Analyze spectrum",
            "loudness": "Measure loudness",
            "onsets": "Detect onsets",
            "peak": "Get peak level",
        },
        "convert": {
            "file": "Convert a file",
            "batch": "Batch convert",
            "normalize": "Normalize audio",
            "trim": "Trim audio",
        },
        "midi": {
            "devices": "List MIDI devices",
            "input": "MIDI input operations",
            "output": "MIDI output operations",
            "file": "MIDI file operations",
            "record": "Record MIDI",
        },
        "sequence": {
            "info": "Show sequence info",
            "play": "Play sequence",
            "tracks": "List tracks",
        },
    }

    for cmd, subs in subcmd_descriptions.items():
        for sub, desc in subs.items():
            lines.append(
                f"complete -c coremusic -n '__fish_seen_subcommand_from {cmd}' "
                f"-a {sub} -d '{desc}'"
            )

    lines.append("")
    lines.append("# File completion for specific commands")
    lines.append(
        "complete -c coremusic -n '__fish_seen_subcommand_from audio analyze convert sequence' "
        "-F -d 'Audio file'"
    )

    return "\n".join(lines)


def register(subparsers: argparse._SubParsersAction) -> None:
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
