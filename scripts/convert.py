
# see also: https://github.com/talonvoice/appscript

def from_fourcharcode(code):
    return (ord(code[0]) << 24) | (ord(code[1]) << 16) | (ord(code[2]) << 8) | ord(code[3])


def from_fourcharcode2(code):
    return int.from_bytes(code.encode('utf8'), 'big')

assert int.from_bytes(b'.mp3', byteorder='big') == 778924083

assert from_fourcharcode('.mp3') == 778924083