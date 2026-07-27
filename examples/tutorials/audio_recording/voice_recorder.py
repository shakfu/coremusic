#!/usr/bin/env python3
"""A voice recorder that keeps a directory of timestamped takes."""

import sys

from coremusic.audio import AudioDeviceManager

if not AudioDeviceManager.get_input_devices():
    print("No audio input device available.")
    raise SystemExit(0)

sys.argv = [sys.argv[0], "0.5"]


# --8<-- [start:example]
import sys
from datetime import datetime
from pathlib import Path

from coremusic.capi import AudioRecorder


class VoiceRecorder:
    """Simple voice recorder with multiple recordings."""

    def __init__(self, output_dir="recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_filename(self):
        """Generate unique filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"recording_{timestamp}.wav"

    def record(self, duration):
        """Record for `duration` seconds, or until interrupted."""
        output_path = self.generate_filename()

        recorder = AudioRecorder(sample_rate=44100.0, channels=1)  # mono for voice
        recorder.setup_input(duration=duration)

        print(f"Recording: {output_path.name}")
        print(f"Duration: {duration}s (Ctrl+C to stop early)")

        recorder.start()
        try:
            while recorder.is_recording():
                recorder.run_loop(0.1)
                elapsed = recorder.get_recorded_duration()
                print(f"  Recording... {elapsed:.1f}s", end='\r')
        except KeyboardInterrupt:
            pass

        recorder.stop()
        recorder.save_to_file(str(output_path))

        print(f"\nRecorded {recorder.get_recorded_duration():.1f}s "
              f"to {output_path.name}")
        return output_path

    def list_recordings(self):
        """List all recordings."""
        recordings = sorted(
            self.output_dir.glob("*.wav"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        print(f"\nRecordings in {self.output_dir}:")
        print("-" * 50)

        for rec in recordings:
            size = rec.stat().st_size / 1024
            mtime = datetime.fromtimestamp(rec.stat().st_mtime)
            print(f"  {rec.name} ({size:.1f} KB) - {mtime:%Y-%m-%d %H:%M}")

        return recordings


def main():
    recorder = VoiceRecorder()

    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    recorder.record(duration=duration)
    recorder.list_recordings()


if __name__ == "__main__":
    main()
# --8<-- [end:example]
