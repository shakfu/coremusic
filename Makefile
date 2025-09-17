.PHONY: all build test clean

all: build

build:
	@python3 setup.py build_ext --inplace	
	@rm -rf ./build ./src/coremusic/capi.c

wheel:
	@python3 setup.py bdist_wheel

test:
	@pytest

clean:
	@rm -rf build src/*.egg-info
	@rm -f *.so
