def make_number_from_char_code(code):
    return (
        (ord(code[0]) << 24) | (ord(code[1]) << 16) | (ord(code[2]) << 8) | ord(code[3])
    )


def test_number_from_char_code():
    value = 1633837924
    assert make_number_from_char_code("abcd") == value

    a = ord("a") << 24
    b = ord("b") << 16
    c = ord("c") << 8
    d = ord("d")

    assert a | b | c | d == value

    assert chr(a >> 24) == "a"
    assert chr(b >> 16) == "b"
    assert chr(c >> 8) == "c"
    assert chr(d) == "d"
