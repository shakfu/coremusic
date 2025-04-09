
def fc(code: str) -> int:
   """Convert fourcc chars to an int

   >>> fourchar_to_int('TEXT')
   1413830740
   """
   assert len(code) == 4, "should be four characters only"
   return ((ord(code[0]) << 24) | (ord(code[1]) << 16) |
           (ord(code[2]) << 8)  | ord(code[3]))



s = """
    kAudioAggregateDevicePropertyFullSubDeviceList = 'grup',
    kAudioAggregateDevicePropertyActiveSubDeviceList = 'agrp',
    kAudioAggregateDevicePropertyComposition = 'acom',
    kAudioAggregateDevicePropertyMainSubDevice = 'amst',
    kAudioAggregateDevicePropertyClockDevice = 'apcd',
    kAudioAggregateDevicePropertyTapList = 'tap#',
    kAudioAggregateDevicePropertySubTapList = 'atap',
"""

s = s.replace(',','').replace("'", "")
s.strip()
lines = [line.strip().lstrip() for line in s.splitlines()]

for line in lines:
	# print(line)
	try:
		k, v = line.split(' = ')
		print(f"{k} = {fc(v)}")
		# print((k, fc(v)))
	except ValueError:
		continue

