#!/usr/bin/env python3
"""Read the CoreAudio clock in several time formats."""

# --8<-- [start:example]
from coremusic.audio import AudioClock

# Use AudioClock for precise timing
with AudioClock() as clock:
    clock.play_rate = 1.0  # Normal speed
    clock.start()

    # Get time in different formats
    seconds = clock.get_time_seconds()
    beats = clock.get_time_beats()
    samples = clock.get_time_samples()
    print(f"{seconds:.3f}s / {beats:.3f} beats / {samples:.0f} samples")

    # Change speed (for tempo sync)
    clock.play_rate = 0.5  # Half speed

    clock.stop()
# --8<-- [end:example]
