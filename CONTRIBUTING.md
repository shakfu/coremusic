# Contributing to CoreMusic

Thank you for your interest in contributing to CoreMusic! This document provides guidelines and instructions for contributing.

## Prerequisites

### System Requirements

- **macOS** (required - CoreMusic uses Apple's CoreAudio/CoreMIDI frameworks)
- **Python 3.11+** (supports 3.11, 3.12, 3.13, 3.14)
- **Xcode Command Line Tools** (for C/C++ compilation)

```bash
xcode-select --install
```

### Development Tools

- **uv** - Fast Python package manager (recommended)
- **CMake** - Build system for Cython extensions
- **Git** - Version control

```bash
# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv cmake
```

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/shakfu/coremusic.git
cd coremusic
```

### 2. Create Virtual Environment and Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using make
make sync
```

### 3. Build the Package

```bash
# Build Cython extensions
make build

# Or directly with uv
uv build
```

### 4. Verify Installation

```bash
# Run tests
make test

# Check import works
uv run python -c "import coremusic; print(coremusic.__version__)"
```

## Development Workflow

### Running Tests

```bash
make test          # Fast tests (excludes slow tests)
make test-all      # All tests including slow ones
make coverage      # Run with coverage report
```

### Code Quality

```bash
make lint          # Run ruff linter
make format        # Format code with ruff
make typecheck     # Run mypy type checker
make qa            # Run all quality checks (lint, format, typecheck, test)
```

### Building Documentation

```bash
make docs          # Build HTML documentation
make docs-serve    # Serve docs locally at http://localhost:8000
```

## Code Style

### Python

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for all public functions and methods
- Maximum line length: 88 characters (ruff default)
- Use double quotes for strings

### Formatting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check formatting
uv run ruff format --check src/ tests/

# Auto-format
uv run ruff format src/ tests/

# Lint
uv run ruff check src/ tests/

# Lint and auto-fix
uv run ruff check --fix src/ tests/
```

### Type Checking

We use [mypy](https://mypy.readthedocs.io/) for static type checking:

```bash
uv run mypy src/coremusic
```

## Pre-commit Hooks

We recommend using pre-commit hooks to automatically check code before commits:

```bash
# Install pre-commit
uv add --dev pre-commit

# Install hooks
uv run pre-commit install

# Run on all files (optional)
uv run pre-commit run --all-files
```

## Project Structure

```
coremusic/
  src/coremusic/
    __init__.py          # Package entry point
    capi.pyx             # Cython bindings to CoreAudio/CoreMIDI
    objects/             # Object-oriented API
      audio.py           # AudioFile, AudioFormat, etc.
      audiounit.py       # AudioUnit classes
      midi.py            # MIDI classes
      devices.py         # Audio device classes
      exceptions.py      # Exception hierarchy
    audio/               # Audio processing utilities
    midi/                # MIDI utilities
    music/               # Music theory
    cli/                 # Command-line interface
  tests/                 # Test suite
  docs/                  # Sphinx documentation
```

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clear, concise commit messages
- Add tests for new functionality
- Update documentation if needed

### 3. Run Quality Checks

```bash
make qa
```

### 4. Submit a Pull Request

- Push your branch to your fork
- Open a pull request against `main`
- Fill out the PR template
- Wait for CI checks to pass

## Testing Guidelines

### Writing Tests

- Use pytest for all tests
- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`

### Test Categories

- **Fast tests**: Run by default with `make test`
- **Slow tests**: Marked with `@pytest.mark.slow`, run with `make test-all`

```python
import pytest

def test_basic_functionality():
    """Fast test - runs by default."""
    assert True

@pytest.mark.slow
def test_slow_operation():
    """Slow test - only runs with make test-all."""
    import time
    time.sleep(2)
    assert True
```

### Test Fixtures

Common fixtures are defined in `tests/conftest.py`:

```python
def test_with_audio_file(test_audio_file):
    """Use the test_audio_file fixture for audio tests."""
    with cm.AudioFile(test_audio_file) as f:
        assert f.duration > 0
```

## Reporting Issues

When reporting issues, please include:

1. **macOS version** (`sw_vers`)
2. **Python version** (`python --version`)
3. **CoreMusic version** (`coremusic --version`)
4. **Steps to reproduce**
5. **Expected vs actual behavior**
6. **Error messages/tracebacks**

## License

By contributing to CoreMusic, you agree that your contributions will be licensed under the MIT License.
