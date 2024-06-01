def makeNumberFromCharCode(code):
  return ((ord(code[0]) << 24) | 
          (ord(code[1]) << 16) |
          (ord(code[2]) << 8)  | 
           ord(code[3]))

assert makeNumberFromCharCode('abcd') == 1633837924

a = ord('a') << 24
b = ord('b') << 16
c = ord('c') << 8
d = ord('d')

assert a | b | c | d == 1633837924


assert chr(a >> 24) == 'a'
assert chr(b >> 16) == 'b'
assert chr(c >> 8) == 'c'
assert chr(d) == 'd'
