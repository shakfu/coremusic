

all: coreaudio

coreaudio:
	@python3 setup.py build_ext --inplace	
	@rm -rf ./build ./coreaudio.c


.PHONY: test clean


test:
	@python3 tests/test_coreaudio.py

clean:
	@rm -rf build
	@rm -f *.so
