# Makefile frontend for scikit-build-core project
# coremusic: coreaudio/coremidi/ableton-link in cython
#
# This Makefile wraps common build commands for convenience.
# The actual build is handled by scikit-build-core via pyproject.toml

.PHONY: all sync build rebuild test test-all test-clean-install lint isort format typecheck qa \
        clean distclean wheel sdist dist check publish-test publish upgrade \
        coverage coverage-html docs docs-clean docs-serve docs-pdf docs-linkcheck \
        release help

# Default target
all: build

# Sync environment (initial setup, installs dependencies + package)
sync:
	@uv sync

# Build/rebuild the extension after code changes
build:
	@uv sync --reinstall-package coremusic

# Alias for build
rebuild: build

# Run tests (excludes slow tests)
test:
	@uv run pytest -m "not slow"

# Run all tests including slow tests
test-all:
	@uv run pytest

# Test clean installation without optional dependencies
test-clean-install:
	@echo "Testing clean installation without optional dependencies..."
	@./scripts/test_clean_install.sh

# Lint with ruff
lint:
	@uv run ruff check --fix src/ tests/

# Sort imports with isort
isort:
	@uv run isort src/coremusic

# Format with ruff
format:
	@uv run ruff format src/ tests/

# Type check with mypy
typecheck:
	@uv run mypy src/coremusic

# Run a full quality assurance check
qa: lint isort format typecheck test

# Build wheel
wheel:
	@uv build --wheel

# Build source distribution
sdist:
	@uv build --sdist

# Check distributions with twine
check:
	@uv run twine check dist/*

# Build both wheel and sdist
dist: wheel sdist check

# Publish to TestPyPI
publish-test: check
	@uv run twine upload --repository testpypi dist/*

# Publish to PyPI
publish: check
	@uv run twine upload dist/*

# Upgrade all dependencies
upgrade:
	@uv lock --upgrade
	@uv sync

# Run tests with coverage (excludes slow tests)
coverage:
	@uv run pytest -m "not slow" --cov=coremusic --cov-report=term-missing

# Generate HTML coverage report
coverage-html:
	@uv run pytest -m "not slow" --cov=coremusic --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"
	@open htmlcov/index.html 2>/dev/null || echo "Open htmlcov/index.html in your browser"

# Build HTML documentation
docs:
	@echo "Building HTML documentation..."
	@cd docs && $(MAKE) html
	@echo "Documentation built in docs/_build/html/index.html"

# Clean documentation build
docs-clean:
	@echo "Cleaning documentation build..."
	@cd docs && $(MAKE) clean

# Serve documentation locally
docs-serve:
	@echo "Starting documentation server at http://localhost:8000"
	@cd docs/_build/html && python3 -m http.server 8000

# Build PDF documentation
docs-pdf:
	@echo "Building PDF documentation..."
	@cd docs && $(MAKE) latexpdf
	@echo "PDF documentation built in docs/_build/latex/coremusic.pdf"

# Check documentation links
docs-linkcheck:
	@echo "Checking documentation links..."
	@cd docs && $(MAKE) linkcheck

# Build release wheels for multiple Python versions
release:
	@rm -rf dist
	@uv build --sdist
	@uv build --wheel --python 3.11
	@uv build --wheel --python 3.12
	@uv build --wheel --python 3.13
	@uv build --wheel --python 3.14

# Clean build artifacts
clean:
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf src/*.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf htmlcov/
	@rm -f .coverage
	@find src -name "*.so" -delete 2>/dev/null || true
	@find src -name "*.pyd" -delete 2>/dev/null || true
	@find . -path ./.venv -prune -o -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Clean everything including CMake cache
distclean: clean
	@rm -rf CMakeCache.txt CMakeFiles/

# Show help
help:
	@echo "Available targets:"
	@echo ""
	@echo "  Build:"
	@echo "    all            - Build/rebuild the extension (default)"
	@echo "    sync           - Sync environment (initial setup)"
	@echo "    build          - Rebuild extension after code changes"
	@echo "    rebuild        - Alias for build"
	@echo ""
	@echo "  Test:"
	@echo "    test           - Run tests (excludes slow tests)"
	@echo "    test-all       - Run all tests including slow tests"
	@echo "    test-clean-install - Test clean installation"
	@echo "    coverage       - Run tests with coverage"
	@echo "    coverage-html  - Generate HTML coverage report"
	@echo ""
	@echo "  Quality:"
	@echo "    lint           - Lint with ruff"
	@echo "    isort          - Sort imports with isort"
	@echo "    format         - Format with ruff"
	@echo "    typecheck      - Type check with mypy"
	@echo "    qa             - Run full QA (lint, isort, format, typecheck, test)"
	@echo ""
	@echo "  Distribution:"
	@echo "    wheel          - Build wheel distribution"
	@echo "    sdist          - Build source distribution"
	@echo "    dist           - Build both wheel and sdist"
	@echo "    check          - Check distributions with twine"
	@echo "    release        - Build wheels for Python 3.11-3.14"
	@echo "    publish-test   - Publish to TestPyPI"
	@echo "    publish        - Publish to PyPI"
	@echo ""
	@echo "  Documentation:"
	@echo "    docs           - Build HTML documentation"
	@echo "    docs-clean     - Clean documentation build"
	@echo "    docs-serve     - Serve documentation locally"
	@echo "    docs-pdf       - Build PDF documentation"
	@echo "    docs-linkcheck - Check documentation links"
	@echo ""
	@echo "  Maintenance:"
	@echo "    upgrade        - Upgrade all dependencies"
	@echo "    snap           - Quick git commit and push"
	@echo "    clean          - Remove build artifacts"
	@echo "    distclean      - Remove all generated files"
	@echo "    help           - Show this help message"
