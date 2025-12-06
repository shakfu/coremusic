#!/usr/bin/env python3
"""Track beats using Link.

Usage:
    python beat_tracking.py [bpm] [duration_seconds]
"""

import sys
import time
import coremusic as cm


def main():
    bpm = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

    with cm.link.LinkSession(bpm=bpm) as session:
        clock = session.clock
        end_time = time.time() + duration

        print(f"Tracking beats at {bpm} BPM for {duration}s...")
        while time.time() < end_time:
            state = session.capture_app_session_state()
            current_time = clock.micros()
            beat = state.beat_at_time(current_time, quantum=4.0)
            phase = state.phase_at_time(current_time, quantum=4.0)
            indicator = "*" if int(beat) % 4 == 0 else "."
            print(f"  {indicator} Beat: {beat:7.2f} | Phase: {phase:4.2f}/4", end='\r')
            time.sleep(0.1)

        print("\nDone")


if __name__ == "__main__":
    main()
