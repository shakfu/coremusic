#!/usr/bin/env python3
"""Play a file with the errors that can actually occur handled."""

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_output_devices():
    print("No audio output device available.")
    raise SystemExit(0)

# --8<-- [start:example]
import time
from pathlib import Path

from coremusic.base import AudioPlayer
from coremusic.exceptions import AudioFileError, AudioQueueError, CoreAudioError


def safe_play(filepath):
    """Play audio with comprehensive error handling."""
    # Check file exists
    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        return False

    try:
        player = AudioPlayer()
        player.load_file(filepath)
        player.setup_output()

        player.play()

        while player.is_playing():
            time.sleep(0.1)

        return True

    except AudioFileError as e:
        print(f"Audio file error: {e}")
        return False
    except AudioQueueError as e:
        print(f"Audio queue error: {e}")
        return False
    except CoreAudioError as e:
        # Every coremusic error derives from this one
        print(f"CoreAudio error: {e}")
        return False


# Use with error handling
success = safe_play("audio.wav")
print(f"Playback {'succeeded' if success else 'failed'}")
# --8<-- [end:example]
