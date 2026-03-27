# coremusic Documentation

This directory contains the documentation for coremusic, built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunnel.github.io/mkdocs-material/).

## Building Documentation

### Prerequisites

Install documentation dependencies (included in dev group):

```bash
uv sync
```

### Serve Locally (with live reload)

```bash
make docs-serve
```

Visit <http://localhost:8000> -- documentation rebuilds automatically when files change.

### Build Static Site

```bash
make docs
```

The HTML site will be built in `site/`.

### Clean Build

```bash
make docs-clean
```

## Documentation Structure

```text
docs/
  index.md                  Main documentation index
  quickstart.md             Quick start guide
  getting_started.md        Installation and setup
  link_integration.md       Ableton Link integration
  api/                      API reference (mkdocstrings)
  guides/                   CLI, imports, performance, migration
  tutorials/                Step-by-step tutorials
  cookbook/                  Ready-to-use recipes
  examples/                 Working examples
  dev/                      Internal development docs
```

## Writing Documentation

All documentation is written in Markdown. See the [MkDocs Material reference](https://squidfunnel.github.io/mkdocs-material/reference/) for formatting options.

### Adding a Page

1. Create a `.md` file in the appropriate directory
2. Add it to the `nav` section in `mkdocs.yml`
3. Run `make docs-serve` to preview

### API Documentation

API docs are auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/):

```markdown
::: coremusic.AudioFile
    options:
      members: true
      show_inheritance: true
```

### Admonitions

```markdown
!!! note
    This is a note.

!!! warning
    This is a warning.
```

## Publishing

Documentation is published to GitHub Pages via `mkdocs gh-deploy`:

```bash
uv run mkdocs gh-deploy
```

## See Also

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunnel.github.io/mkdocs-material/)
- [mkdocstrings](https://mkdocstrings.github.io/)
