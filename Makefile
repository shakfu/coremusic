.PHONY: all build test clean docs docs-clean docs-serve docs-pdf check \
		deploy publish

all: build

build:
	@uv sync --reinstall-package coremusic
# 	@uv run python setup.py build_ext --inplace
# 	@rm -rf ./build ./src/coremusic/capi.c

# wheel:
# 	@uv run python setup.py bdist_wheel

wheel:
	@uv build

test:
	@uv run pytest

clean:
	@rm -rf build src/*.egg-info
	@rm -f *.so

check: wheel
	@uv run twine check dist/*

publish: wheel
	@uv run twine upload dist/*

# Documentation targets
docs:
	@echo "Building HTML documentation..."
	@cd docs && uv run sphinx-build -b html . _build/html
	@echo "Documentation built in docs/_build/html/index.html"

docs-clean:
	@echo "Cleaning documentation build..."
	@rm -rf docs/_build

docs-serve:
	@echo "Starting documentation server at http://localhost:8000"
	@cd docs/_build/html && uv run python -m http.server 8000

docs-pdf:
	@echo "Building PDF documentation..."
	@cd docs && uv run sphinx-build -b latex . _build/latex
	@cd docs/_build/latex && make
	@echo "PDF documentation built in docs/_build/latex/coremusic.pdf"

docs-linkcheck:
	@echo "Checking documentation links..."
	@cd docs && uv run sphinx-build -b linkcheck . _build/linkcheck
