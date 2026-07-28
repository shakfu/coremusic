#!/usr/bin/env python3
"""Watch the CoreAudio clock, then halve its play rate."""

# --8<-- [start:example]
import time

from coremusic.audio import AudioClock


def clock_demo():
    """Demonstrate AudioClock timing and synchronization."""
    with AudioClock() as clock:
        # Set playback rate
        clock.play_rate = 1.0  # Normal speed

        # Start the clock
        clock.start()
        print("Clock started at normal speed")

        # Monitor time in different formats
        for _i in range(5):
            seconds = clock.get_time_seconds()
            beats = clock.get_time_beats()
            samples = clock.get_time_samples()
            print(f"Time: {seconds:.3f}s, {beats:.2f} beats, {samples:.0f} samples")
            time.sleep(0.2)

        # Change playback rate
        clock.play_rate = 0.5  # Half speed
        print("\nChanged to half speed")

        start_time = clock.get_time_seconds()
        time.sleep(1.0)
        end_time = clock.get_time_seconds()

        print("Real time elapsed: 1.0s")
        print(f"Clock time elapsed: {end_time - start_time:.3f}s")

        clock.stop()
        print("Clock stopped")


if __name__ == "__main__":
    clock_demo()
# --8<-- [end:example]
