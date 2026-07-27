#!/usr/bin/env python3
"""Automate Parameters."""

# --8<-- [start:example]
from coremusic.audio.audiounit_host import AudioUnitPlugin

import time

with AudioUnitPlugin.from_name("AUDelay") as plugin:
    # Fade delay time from 0 to 1 second
    for i in range(100):
        delay_time = i / 100.0
        plugin['Delay Time'] = delay_time
        time.sleep(0.05)  # 50ms steps
# --8<-- [end:example]
