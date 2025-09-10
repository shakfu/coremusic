.PHONY: all coreaudio test clean

all: coreaudio

coreaudio:
	@python3 setup.py build_ext --inplace	
	@rm -rf ./build ./coreaudio.c

test:
	@pytest

clean:
	@rm -rf build
	@rm -f *.so
