.PHONY: all sync build test test-all test-clean-install coverage coverage-html clean typecheck docs docs-clean docs-serve docs-pdf  \
		release check publish lint isort

all: build

sync:
	@uv sync

build:
	@uv sync --reinstall-package coremusic

wheel:
	@uv build --wheel

test:
	@uv run pytest -m "not slow"

test-all:
	@uv run pytest

test-clean-install:
	@echo "Testing clean installation without optional dependencies..."
	@./scripts/test_clean_install.sh

coverage:
	@uv run pytest -m "not slow" --cov=coremusic --cov-report=term-missing

coverage-html:
	@uv run pytest -m "not slow" --cov=coremusic --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"
	@open htmlcov/index.html 2>/dev/null || echo "Open htmlcov/index.html in your browser"

lint:
	@uv run ruff check --fix src/coremusic

isort:
	@uv run isort src/coremusic

typecheck:
	@uv run mypy src/coremusic

clean:
	@rm -rf build src/*.egg-info htmlcov .coverage
	@rm -f *.so

release:
	@rm -rf dist
	@uv build --sdist
	@uv build --wheel --python 3.11
	@uv build --wheel --python 3.12
	@uv build --wheel --python 3.13
	@uv build --wheel --python 3.14

check:
	@uv run twine check dist/*

publish:
	@uv run twine upload dist/*

snap:
	@git add --all . && git commit -m 'snap' && git push

# Documentation targets
docs:
	@echo "Building HTML documentation..."
	@cd docs && $(MAKE) html
	@echo "Documentation built in docs/_build/html/index.html"

docs-clean:
	@echo "Cleaning documentation build..."
	@cd docs && $(MAKE) clean

docs-serve:
	@echo "Starting documentation server at http://localhost:8000"
	@cd docs/_build/html && python3 -m http.server 8000

docs-pdf:
	@echo "Building PDF documentation..."
	@cd docs && $(MAKE) latexpdf
	@echo "PDF documentation built in docs/_build/latex/coremusic.pdf"

docs-linkcheck:
	@echo "Checking documentation links..."
	@cd docs && $(MAKE) linkcheck
