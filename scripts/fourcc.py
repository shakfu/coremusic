#!/usr/bin/env python3

"""fourcc.py: convert to and from four character code and ints"""

from __future__ import annotations

# see also: https://github.com/talonvoice/appscript


def from_fourcharcode(code):
    return (
        (ord(code[0]) << 24) | (ord(code[1]) << 16) | (ord(code[2]) << 8) | ord(code[3])
    )


def from_fourcharcode2(code):
    return int.from_bytes(code.encode("utf8"), "big")


def int_to_fourchar(n):
    return (
        chr((n >> 24) & 255) + chr((n >> 16) & 255) + chr((n >> 8) & 255) + chr(n & 255)
    )


assert int.from_bytes(b".mp3", byteorder="big") == 778924083
assert from_fourcharcode(".mp3") == 778924083
assert from_fourcharcode2(".mp3") == 778924083
assert int_to_fourchar(778924083) == ".mp3"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--from-code", help="from fourcc code to int", type=str)
    parser.add_argument("-i", "--from-int", help="from int to fourcc code", type=int)
    args = parser.parse_args()

    if args.from_code:
        print(from_fourcharcode(args.from_code))
    elif args.from_int:
        print(int_to_fourchar(args.from_int))
