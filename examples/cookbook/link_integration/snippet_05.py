#!/usr/bin/env python3
"""Beat-Accurate Playback Start."""

# --8<-- [start:example]
import time

from coremusic import link
from coremusic.base import AudioPlayer

with link.LinkSession(bpm=120.0) as session:
    player = AudioPlayer(link_session=session)
    player.load_file("audio.wav")
    player.setup_output()

    # Wait for start of next bar (beat 0)
    state = session.capture_app_session_state()
    current_time = session.clock.micros()
    current_beat = state.beat_at_time(current_time, 4.0)

    # Calculate time to next bar
    next_bar_beat = (int(current_beat / 4) + 1) * 4
    next_bar_time = state.time_at_beat(next_bar_beat, 4.0)

    # Wait until next bar
    wait_micros = next_bar_time - current_time
    time.sleep(wait_micros / 1000000.0)

    # Start playback on the beat
    player.play()
    print(f"Started playback on beat {next_bar_beat}")

    time.sleep(5.0)
    player.stop()
# --8<-- [end:example]
