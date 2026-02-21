#!/usr/bin/env python3
"""Create a Link session.

Usage:
    python session.py [bpm]
"""

import sys
from coremusic import link


def main():
    bpm = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0

    with link.LinkSession(bpm=bpm) as session:
        state = session.capture_app_session_state()
        print("Link Session:")
        print(f"  Tempo: {state.tempo:.1f} BPM")
        print(f"  Peers: {session.num_peers}")
        print(f"  Enabled: {session.enabled}")


if __name__ == "__main__":
    main()
