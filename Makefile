.PHONY: all build test clean

all: build

build:
	@uv run python setup.py build_ext --inplace	
	@rm -rf ./build ./src/coremusic/capi.c

wheel:
	@uv run python setup.py bdist_wheel

test:
	@uv run pytest

clean:
	@rm -rf build src/*.egg-info
	@rm -f *.so
