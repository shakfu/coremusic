#!/usr/bin/env python3
"""Record with the failure modes handled."""

# --8<-- [start:example]
from pathlib import Path

from coremusic.audio import AudioDeviceManager
from coremusic.capi import AudioRecorder
from coremusic.exceptions import AudioDeviceError, AudioQueueError


def safe_record(output_path, duration):
    """Record with comprehensive error handling."""
    # Check output directory exists
    output_dir = Path(output_path).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # An absent input device is the most common failure, and is worth
    # reporting before touching CoreAudio
    if not AudioDeviceManager.get_input_devices():
        print("No audio input device available")
        return False

    try:
        recorder = AudioRecorder(sample_rate=44100.0, channels=2)
        recorder.setup_input(duration=duration)

        print(f"Recording to: {output_path}")
        recorder.start()
        while recorder.is_recording():
            recorder.run_loop(0.1)
        recorder.stop()

        if not recorder.has_audio_content():
            # Silence usually means macOS denied microphone access
            print("Warning: recording is silent - check microphone permissions")

        recorder.save_to_file(output_path)

        # Verify file was created
        if Path(output_path).exists():
            size = Path(output_path).stat().st_size
            print(f"Recording saved ({size / 1024:.1f} KB)")
            return True

        print("Error: Recording file not created")
        return False

    except AudioQueueError as e:
        print(f"Audio queue error: {e}")
        return False
    except AudioDeviceError as e:
        print(f"Audio device error: {e}")
        print("Check that an input device is available")
        return False
    except PermissionError as e:
        print(f"Permission error: {e}")
        print("Check microphone permissions in System Settings")
        return False


safe_record("safe_recording.wav", duration=0.5)
# --8<-- [end:example]
