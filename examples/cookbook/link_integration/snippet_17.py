#!/usr/bin/env python3
"""Thread Safety."""

# --8<-- [start:example]
import threading

from coremusic import link

with link.LinkSession(bpm=120.0) as session:
    def audio_thread():
        """This runs on audio thread"""
        # Capture state on audio thread for low latency
        state = session.capture_audio_session_state()
        current_time = session.clock.micros()
        beat = state.beat_at_time(current_time, 4.0)
        # Process audio...

    def ui_thread():
        """This runs on UI thread"""
        # Capture state on UI thread for UI updates
        state = session.capture_app_session_state()
        tempo = state.tempo
        # Update UI...
# --8<-- [end:example]
