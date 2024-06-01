
.PHONY: all build clean

all: build


build: test_sin test_sin2 test_audio


test_sin:
	@clang -o ./test_sin test_sin.c -framework AudioUnit

test_sin2:
	@clang -o ./test_sin2 test_sin2.c -framework AudioUnit


test_audio:
	@clang -o test_audio -std=c99 -lc -framework AudioToolbox test_audio.c

clean:
	@rm -f test_sin test_sin2 test_audio

