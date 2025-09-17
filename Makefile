.PHONY: all build test clean

all: build

build:
	@python3 setup.py build_ext --inplace	
	@rm -rf ./build ./src/coremusic/capi.c

test:
	@pytest

clean:
	@rm -rf build
	@rm -f *.so
