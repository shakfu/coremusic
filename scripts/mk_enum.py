def fc(code: str) -> int:
    """Convert fourcc chars to an int

    >>> fourchar_to_int('TEXT')
    1413830740
    """
    # print(repr(code))
    assert len(code) == 4, "should be four characters only"
    return (
        (ord(code[0]) << 24) | (ord(code[1]) << 16) | (ord(code[2]) << 8) | ord(code[3])
    )


s = """
kAudioHardwarePropertyDevices = 'dev#',
kAudioHardwarePropertyDefaultInputDevice = 'dInx',
kAudioHardwarePropertyDefaultOutputDevice = 'dOut',
kAudioHardwarePropertyDefaultSystemOutputDevice = 'sOut',
kAudioHardwarePropertyTranslateUIDToDevice = 'uidd',
kAudioHardwarePropertyMixStereoToMono = 'stmo',
kAudioHardwarePropertyPlugInList = 'plg#',
kAudioHardwarePropertyTranslateBundleIDToPlugIn = 'bidp',
kAudioHardwarePropertyTransportManagerList = 'tmg#',
kAudioHardwarePropertyTranslateBundleIDToTransportManager = 'tmbi',
kAudioHardwarePropertyBoxList = 'box#',
kAudioHardwarePropertyTranslateUIDToBox = 'uidb',
kAudioHardwarePropertyClockDeviceList = 'clk#',
kAudioHardwarePropertyTranslateUIDToClockDevice = 'uidc',
kAudioHardwarePropertyProcessIsMain = 'main',
kAudioHardwarePropertyIsInitingOrExiting = 'inot',
kAudioHardwarePropertyUserIDChanged = 'euid',
kAudioHardwarePropertyProcessInputMute = 'pmin',
kAudioHardwarePropertyProcessIsAudible = 'pmut',
kAudioHardwarePropertySleepingIsAllowed = 'slep',
kAudioHardwarePropertyUnloadingIsAllowed = 'unld',
kAudioHardwarePropertyHogModeIsAllowed = 'hogr',
kAudioHardwarePropertyUserSessionIsActiveOrHeadless = 'user',
kAudioHardwarePropertyServiceRestarted = 'srst',
kAudioHardwarePropertyPowerHint = 'powh',
kAudioHardwarePropertyProcessObjectList = 'prs#',
kAudioHardwarePropertyTranslatePIDToProcessObject = 'id2p',
kAudioHardwarePropertyTapList = 'tps#',
kAudioHardwarePropertyTranslateUIDToTap = 'uidt',
"""

s = s.replace(",", "").replace("'", "")
s.strip()
lines = [line.strip().lstrip() for line in s.splitlines()]

for line in lines:
    # print(line)
    try:
        k, v = line.split(" = ")
        print(f"{k} = {fc(v)}")
        # print((k, fc(v)))
    except ValueError:
        continue
