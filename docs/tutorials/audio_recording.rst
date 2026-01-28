Audio Recording
===============

This tutorial covers recording audio from input devices using coremusic.

Prerequisites
-------------

- coremusic installed and built
- A working audio input device (built-in microphone, USB audio interface, etc.)
- Basic Python knowledge

Simple Recording
----------------

Using the CLI
^^^^^^^^^^^^^

The easiest way to record is via the command line:

.. code-block:: bash

   # Record for 10 seconds
   coremusic audio record -o recording.wav --duration 10

   # Record with specific settings
   coremusic audio record -o recording.wav --duration 10 --sample-rate 48000 --channels 1

   # List input devices
   coremusic device list --input

Using AudioRecorder
^^^^^^^^^^^^^^^^^^^

For programmatic recording:

.. code-block:: python

   import coremusic as cm
   import time

   def record_audio(output_path, duration_seconds):
       """Record audio to a WAV file."""
       recorder = cm.AudioRecorder()

       # Configure recording
       recorder.setup(
           sample_rate=44100.0,
           channels=2,
           output_path=output_path
       )

       print(f"Recording for {duration_seconds} seconds...")
       print("Press Ctrl+C to stop early")

       # Start recording
       recorder.start()

       try:
           # Wait for duration
           for i in range(duration_seconds):
               print(f"Recording: {i + 1}/{duration_seconds}s", end='\r')
               time.sleep(1.0)
       except KeyboardInterrupt:
           print("\nStopped early")

       # Stop and save
       recorder.stop()
       print(f"\nSaved to: {output_path}")

   record_audio("my_recording.wav", duration_seconds=10)

Recording with Progress
-----------------------

Display recording progress:

.. code-block:: python

   import coremusic as cm
   import time
   import sys

   def record_with_progress(output_path, duration):
       """Record with visual progress bar."""
       recorder = cm.AudioRecorder()
       recorder.setup(
           sample_rate=44100.0,
           channels=2,
           output_path=output_path
       )

       print(f"Recording: {output_path}")
       print(f"Duration: {duration}s")
       print()

       recorder.start()
       start_time = time.time()

       try:
           while True:
               elapsed = time.time() - start_time
               if elapsed >= duration:
                   break

               # Progress bar
               progress = elapsed / duration
               bar_width = 40
               filled = int(bar_width * progress)
               bar = '=' * filled + '-' * (bar_width - filled)

               # Level indicator (simplified)
               level = recorder.get_input_level() if hasattr(recorder, 'get_input_level') else 0
               level_bar = '|' * int(level * 20)

               sys.stdout.write(f'\r[{bar}] {elapsed:.1f}s {level_bar}')
               sys.stdout.flush()

               time.sleep(0.05)

       except KeyboardInterrupt:
           print("\nStopped by user")

       recorder.stop()
       print(f"\nRecording saved to: {output_path}")

   record_with_progress("recording.wav", duration=10)

Device Selection
----------------

List Input Devices
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def list_input_devices():
       """List all available input devices."""
       devices = cm.AudioDeviceManager.get_all_devices()

       print("Input Devices:")
       print("-" * 50)

       for device in devices:
           if device.has_input:
               print(f"Name: {device.name}")
               print(f"  ID: {device.device_id}")
               print(f"  Channels: {device.input_channels}")
               print(f"  Sample Rate: {device.sample_rate}")
               print()

       return [d for d in devices if d.has_input]

   input_devices = list_input_devices()

Record from Specific Device
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def record_from_device(device_name, output_path, duration):
       """Record from a specific audio device."""
       # Find device
       devices = cm.AudioDeviceManager.get_all_devices()
       target = None

       for device in devices:
           if device.has_input and device_name.lower() in device.name.lower():
               target = device
               break

       if not target:
           print(f"Device not found: {device_name}")
           return

       print(f"Recording from: {target.name}")

       recorder = cm.AudioRecorder()
       recorder.setup(
           sample_rate=target.sample_rate,
           channels=target.input_channels,
           output_path=output_path,
           device_id=target.device_id
       )

       recorder.start()

       import time
       time.sleep(duration)

       recorder.stop()
       print(f"Saved to: {output_path}")

   record_from_device("USB Audio", "usb_recording.wav", duration=5)

Recording Formats
-----------------

Different Sample Rates
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def record_high_quality(output_path, duration):
       """Record at professional sample rate."""
       recorder = cm.AudioRecorder()
       recorder.setup(
           sample_rate=96000.0,  # 96kHz
           channels=2,
           bits_per_sample=24,  # 24-bit
           output_path=output_path
       )

       print("Recording at 96kHz/24-bit...")
       recorder.start()

       import time
       time.sleep(duration)

       recorder.stop()
       print(f"High-quality recording saved to: {output_path}")

   record_high_quality("hq_recording.wav", duration=10)

Mono Recording
^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def record_mono(output_path, duration):
       """Record single channel (mono) audio."""
       recorder = cm.AudioRecorder()
       recorder.setup(
           sample_rate=44100.0,
           channels=1,  # Mono
           output_path=output_path
       )

       print("Recording mono...")
       recorder.start()

       import time
       time.sleep(duration)

       recorder.stop()
       print(f"Mono recording saved to: {output_path}")

   record_mono("mono_recording.wav", duration=5)

Real-Time Monitoring
--------------------

Monitor input levels while recording:

.. code-block:: python

   import coremusic as cm
   import time
   import sys

   def record_with_monitoring(output_path, duration):
       """Record with real-time level monitoring."""
       recorder = cm.AudioRecorder()
       recorder.setup(
           sample_rate=44100.0,
           channels=2,
           output_path=output_path
       )

       print("Recording with level monitoring")
       print("=" * 50)

       recorder.start()
       start_time = time.time()

       try:
           while time.time() - start_time < duration:
               elapsed = time.time() - start_time

               # Get current levels (if available)
               # This is a simplified example
               level_l = 0.5  # Would come from actual input
               level_r = 0.5

               # Display level meters
               meter_width = 20
               left_bar = '|' * int(level_l * meter_width)
               right_bar = '|' * int(level_r * meter_width)

               sys.stdout.write(
                   f'\r[{elapsed:6.1f}s] L:{left_bar:<20} R:{right_bar:<20}'
               )
               sys.stdout.flush()

               time.sleep(0.05)

       except KeyboardInterrupt:
           print("\nStopped")

       recorder.stop()
       print(f"\nSaved: {output_path}")

   record_with_monitoring("monitored.wav", duration=10)

Recording to NumPy Array
------------------------

For processing, record directly to a NumPy array:

.. code-block:: python

   import coremusic as cm

   if cm.NUMPY_AVAILABLE:
       import numpy as np

       def record_to_numpy(duration, sample_rate=44100.0):
           """Record audio directly to NumPy array."""
           # Calculate buffer size
           num_samples = int(duration * sample_rate)

           # Create buffer
           audio_buffer = np.zeros((num_samples, 2), dtype=np.float32)

           # Record using low-level API
           # (Simplified - actual implementation varies)
           recorder = cm.AudioRecorder()
           recorder.setup(
               sample_rate=sample_rate,
               channels=2
           )

           recorder.start()

           import time
           time.sleep(duration)

           recorder.stop()

           # Get recorded data as NumPy array
           audio_data = recorder.get_data_as_numpy()

           return audio_data, sample_rate

       # Record and process
       audio, sr = record_to_numpy(duration=5)
       print(f"Recorded {len(audio)} samples at {sr}Hz")
       print(f"Peak amplitude: {np.max(np.abs(audio)):.4f}")

Error Handling
--------------

Handle recording errors gracefully:

.. code-block:: python

   import coremusic as cm
   from pathlib import Path

   def safe_record(output_path, duration):
       """Record with comprehensive error handling."""
       # Check output directory exists
       output_dir = Path(output_path).parent
       if not output_dir.exists():
           output_dir.mkdir(parents=True)

       try:
           recorder = cm.AudioRecorder()
           recorder.setup(
               sample_rate=44100.0,
               channels=2,
               output_path=output_path
           )

           print(f"Recording to: {output_path}")
           recorder.start()

           import time
           time.sleep(duration)

           recorder.stop()

           # Verify file was created
           if Path(output_path).exists():
               size = Path(output_path).stat().st_size
               print(f"Recording saved ({size / 1024:.1f} KB)")
               return True
           else:
               print("Error: Recording file not created")
               return False

       except cm.AudioQueueError as e:
           print(f"Audio queue error: {e}")
           return False
       except cm.AudioDeviceError as e:
           print(f"Audio device error: {e}")
           print("Check that an input device is available")
           return False
       except PermissionError as e:
           print(f"Permission error: {e}")
           print("Check microphone permissions in System Preferences")
           return False
       except Exception as e:
           print(f"Unexpected error: {e}")
           return False

   # Safe recording
   success = safe_record("output/recording.wav", duration=5)

Complete Example: Voice Recorder
--------------------------------

A complete voice recorder application:

.. code-block:: python

   import coremusic as cm
   import sys
   import time
   from pathlib import Path
   from datetime import datetime

   class VoiceRecorder:
       """Simple voice recorder with multiple recordings."""

       def __init__(self, output_dir="recordings"):
           self.output_dir = Path(output_dir)
           self.output_dir.mkdir(exist_ok=True)
           self.recorder = None

       def generate_filename(self):
           """Generate unique filename with timestamp."""
           timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
           return self.output_dir / f"recording_{timestamp}.wav"

       def record(self, duration=None):
           """Record audio. If duration is None, record until stopped."""
           output_path = self.generate_filename()

           self.recorder = cm.AudioRecorder()
           self.recorder.setup(
               sample_rate=44100.0,
               channels=1,  # Mono for voice
               output_path=str(output_path)
           )

           print(f"Recording: {output_path.name}")
           if duration:
               print(f"Duration: {duration}s")
           else:
               print("Press Ctrl+C to stop")

           self.recorder.start()
           start_time = time.time()

           try:
               if duration:
                   # Fixed duration
                   for i in range(int(duration)):
                       elapsed = i + 1
                       remaining = duration - elapsed
                       print(f"  Recording... {elapsed}s (remaining: {remaining}s)", end='\r')
                       time.sleep(1.0)
               else:
                   # Indefinite - wait for Ctrl+C
                   while True:
                       elapsed = time.time() - start_time
                       print(f"  Recording... {elapsed:.1f}s", end='\r')
                       time.sleep(0.1)

           except KeyboardInterrupt:
               pass

           elapsed = time.time() - start_time
           self.recorder.stop()

           print(f"\nRecorded {elapsed:.1f}s to {output_path.name}")
           return output_path

       def list_recordings(self):
           """List all recordings."""
           recordings = list(self.output_dir.glob("*.wav"))
           recordings.sort(key=lambda x: x.stat().st_mtime, reverse=True)

           print(f"\nRecordings in {self.output_dir}:")
           print("-" * 50)

           for rec in recordings:
               size = rec.stat().st_size / 1024
               mtime = datetime.fromtimestamp(rec.stat().st_mtime)
               print(f"  {rec.name} ({size:.1f} KB) - {mtime:%Y-%m-%d %H:%M}")

           return recordings

   def main():
       recorder = VoiceRecorder()

       if len(sys.argv) > 1:
           # Record for specified duration
           duration = float(sys.argv[1])
           recorder.record(duration=duration)
       else:
           # Interactive mode
           print("Voice Recorder")
           print("=" * 40)
           print("Commands: r=record, l=list, q=quit")
           print()

           while True:
               cmd = input("> ").strip().lower()

               if cmd == 'r':
                   dur = input("Duration (seconds, or Enter for manual stop): ").strip()
                   if dur:
                       recorder.record(duration=float(dur))
                   else:
                       recorder.record()

               elif cmd == 'l':
                   recorder.list_recordings()

               elif cmd == 'q':
                   break

               else:
                   print("Unknown command. Use r, l, or q")

   if __name__ == "__main__":
       main()

Troubleshooting
---------------

No Input Device Found
^^^^^^^^^^^^^^^^^^^^^

1. Check System Preferences > Sound > Input
2. Ensure microphone permissions are granted
3. List devices with ``coremusic device list --input``

Recording is Silent
^^^^^^^^^^^^^^^^^^^

1. Check input device is selected correctly
2. Verify microphone is not muted
3. Test with Audio MIDI Setup app
4. Check input gain/volume

Permission Denied
^^^^^^^^^^^^^^^^^

macOS requires microphone permission:

1. Go to System Preferences > Security & Privacy > Privacy
2. Select Microphone
3. Enable permission for Terminal or your Python app

Next Steps
----------

- :doc:`audio_playback` - Play back your recordings
- :doc:`../cookbook/audio_processing` - Process recorded audio
- :doc:`../cookbook/real_time_audio` - Real-time audio monitoring

See Also
--------

- :doc:`../api/index` - Complete API reference
- :doc:`../guides/cli` - CLI recording commands
