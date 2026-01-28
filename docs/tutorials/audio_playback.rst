Audio Playback
==============

This tutorial covers audio playback using coremusic, from simple file playback to real-time streaming.

Prerequisites
-------------

- coremusic installed and built
- Basic Python knowledge
- Audio files to play (WAV, AIFF, MP3, M4A, etc.)

Simple File Playback
--------------------

Using AudioPlayer (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``AudioPlayer`` class provides the easiest way to play audio files:

.. code-block:: python

   import coremusic as cm
   import time

   # Create player and load file
   player = cm.AudioPlayer()
   player.load_file("audio.wav")
   player.setup_output()

   # Start playback
   player.start()

   # Wait for playback to complete
   while player.is_playing():
       time.sleep(0.1)

   print("Playback complete!")

Playback with Progress
^^^^^^^^^^^^^^^^^^^^^^

Monitor playback progress with a progress bar:

.. code-block:: python

   import coremusic as cm
   import time
   import sys

   def play_with_progress(filepath):
       """Play audio file with progress display."""
       player = cm.AudioPlayer()
       player.load_file(filepath)
       player.setup_output()

       # Get duration
       duration = player.duration
       print(f"Playing: {filepath}")
       print(f"Duration: {duration:.2f}s")

       player.start()

       while player.is_playing():
           progress = player.get_progress()
           current_time = progress * duration

           # Display progress bar
           bar_width = 40
           filled = int(bar_width * progress)
           bar = '=' * filled + '-' * (bar_width - filled)

           sys.stdout.write(f'\r[{bar}] {current_time:.1f}s / {duration:.1f}s')
           sys.stdout.flush()
           time.sleep(0.1)

       print('\nDone!')

   play_with_progress("audio.wav")

Looping Playback
^^^^^^^^^^^^^^^^

For continuous looping:

.. code-block:: python

   import coremusic as cm
   import time

   def play_looped(filepath, num_loops=3):
       """Play audio file multiple times."""
       player = cm.AudioPlayer()
       player.load_file(filepath)
       player.setup_output()

       for i in range(num_loops):
           print(f"Loop {i + 1}/{num_loops}")
           player.start()

           while player.is_playing():
               time.sleep(0.1)

           # Reset position for next loop
           player.seek(0)

       print("Looping complete!")

   play_looped("audio.wav", num_loops=3)

Using the CLI
-------------

The coremusic CLI provides quick playback:

.. code-block:: bash

   # Simple playback
   coremusic audio play music.wav

   # Looping playback
   coremusic audio play music.wav --loop

   # List audio devices
   coremusic device list

Low-Level Playback with AudioQueue
----------------------------------

For more control, use ``AudioQueue`` directly:

.. code-block:: python

   import coremusic as cm

   def play_with_audio_queue(filepath):
       """Low-level playback using AudioQueue."""
       # Open audio file
       with cm.AudioFile(filepath) as audio:
           fmt = audio.format

           # Create output queue with same format
           with cm.AudioQueue.new_output(fmt) as queue:
               # Allocate buffers
               buffer_size = 4096
               num_buffers = 3
               buffers = [queue.allocate_buffer(buffer_size) for _ in range(num_buffers)]

               # Prime buffers with initial data
               current_packet = 0
               for buffer in buffers:
                   data, count = audio.read_packets(current_packet, buffer_size // fmt.bytes_per_frame)
                   if count > 0:
                       # Fill buffer and enqueue
                       # buffer.fill(data)  # Implementation varies
                       current_packet += count

               # Start playback
               queue.start()

               # Main playback loop would continue filling buffers
               # This is simplified - real implementation needs callback

               import time
               time.sleep(audio.duration)

               queue.stop()

   play_with_audio_queue("audio.wav")

Streaming Playback
------------------

For large files or network streams, use chunked reading:

.. code-block:: python

   import coremusic as cm
   from coremusic.audio import AudioFileStream

   def stream_audio(filepath):
       """Stream audio file in chunks."""
       with AudioFileStream(filepath) as stream:
           # Configure chunk size
           stream.chunk_size = 4096

           print(f"Streaming: {filepath}")
           print(f"Duration: {stream.duration:.2f}s")

           # Process chunks
           for chunk in stream:
               # Each chunk is audio data
               # In a real app, you'd send this to an output queue
               pass

   stream_audio("large_audio.wav")

Async Playback
--------------

For non-blocking playback in async applications:

.. code-block:: python

   import asyncio
   import coremusic as cm

   async def async_playback(filepath):
       """Non-blocking audio playback."""
       async with cm.AsyncAudioFile(filepath) as audio:
           print(f"Duration: {audio.duration:.2f}s")

           # Stream chunks asynchronously
           async for chunk in audio.read_chunks_async(chunk_size=4096):
               # Process chunk without blocking
               await asyncio.sleep(0)  # Yield to event loop

   # Run async playback
   asyncio.run(async_playback("audio.wav"))

Playback with Effects
---------------------

Route audio through AudioUnit effects during playback:

.. code-block:: python

   import coremusic as cm

   def play_with_reverb(filepath):
       """Play audio with reverb effect."""
       # Create effects chain
       chain = cm.AudioEffectsChain()

       # Add reverb effect
       reverb_node = chain.add_effect_by_name("AUReverb2")
       output_node = chain.add_output()

       # Connect effect to output
       chain.connect(reverb_node, output_node)

       # Open and configure chain
       chain.open()
       chain.initialize()

       try:
           chain.start()

           # Load and play through chain
           with cm.AudioFile(filepath) as audio:
               # Feed audio through chain
               # (Simplified - real implementation needs buffer management)
               import time
               time.sleep(audio.duration)

       finally:
           chain.stop()
           chain.dispose()

   play_with_reverb("audio.wav")

Device Selection
----------------

Play to a specific audio device:

.. code-block:: python

   import coremusic as cm

   def list_output_devices():
       """List available output devices."""
       devices = cm.AudioDeviceManager.get_all_devices()

       print("Output Devices:")
       for device in devices:
           if device.has_output:
               print(f"  {device.name} (ID: {device.device_id})")

       return devices

   def play_to_device(filepath, device_name):
       """Play audio to specific device."""
       # Find device by name
       devices = cm.AudioDeviceManager.get_all_devices()
       target_device = None

       for device in devices:
           if device_name.lower() in device.name.lower():
               target_device = device
               break

       if not target_device:
           print(f"Device not found: {device_name}")
           return

       print(f"Playing to: {target_device.name}")

       # Create player with specific device
       player = cm.AudioPlayer()
       player.load_file(filepath)
       player.setup_output(device_id=target_device.device_id)

       player.start()
       while player.is_playing():
           import time
           time.sleep(0.1)

   # List devices
   list_output_devices()

   # Play to specific device
   play_to_device("audio.wav", "MacBook Pro Speakers")

Volume Control
--------------

Control playback volume:

.. code-block:: python

   import coremusic as cm
   import time

   def play_with_volume_fade(filepath):
       """Play with volume fade in/out."""
       player = cm.AudioPlayer()
       player.load_file(filepath)
       player.setup_output()

       # Start at zero volume
       player.volume = 0.0
       player.start()

       # Fade in over 2 seconds
       print("Fading in...")
       for i in range(20):
           player.volume = i / 20.0
           time.sleep(0.1)

       # Play at full volume
       print("Playing...")
       time.sleep(2.0)

       # Fade out over 2 seconds
       print("Fading out...")
       for i in range(20, 0, -1):
           player.volume = i / 20.0
           time.sleep(0.1)

       player.stop()
       print("Done!")

   play_with_volume_fade("audio.wav")

Error Handling
--------------

Handle playback errors gracefully:

.. code-block:: python

   import coremusic as cm
   from pathlib import Path

   def safe_play(filepath):
       """Play audio with comprehensive error handling."""
       # Check file exists
       if not Path(filepath).exists():
           print(f"Error: File not found: {filepath}")
           return False

       try:
           player = cm.AudioPlayer()
           player.load_file(filepath)
           player.setup_output()

           player.start()

           while player.is_playing():
               import time
               time.sleep(0.1)

           return True

       except cm.AudioFileError as e:
           print(f"Audio file error: {e}")
           return False
       except cm.AudioQueueError as e:
           print(f"Audio queue error: {e}")
           return False
       except cm.CoreAudioError as e:
           print(f"CoreAudio error: {e}")
           return False
       except Exception as e:
           print(f"Unexpected error: {e}")
           return False

   # Use with error handling
   success = safe_play("audio.wav")
   print(f"Playback {'succeeded' if success else 'failed'}")

Complete Example: Music Player
------------------------------

A simple command-line music player:

.. code-block:: python

   import coremusic as cm
   import sys
   import time
   from pathlib import Path

   class SimpleMusicPlayer:
       """Simple command-line music player."""

       def __init__(self):
           self.player = cm.AudioPlayer()
           self.is_paused = False

       def load(self, filepath):
           """Load audio file."""
           if not Path(filepath).exists():
               raise FileNotFoundError(f"File not found: {filepath}")

           self.player.load_file(filepath)
           self.player.setup_output()
           print(f"Loaded: {filepath}")
           print(f"Duration: {self.player.duration:.2f}s")

       def play(self):
           """Start or resume playback."""
           if self.is_paused:
               self.player.resume()
               self.is_paused = False
           else:
               self.player.start()
           print("Playing...")

       def pause(self):
           """Pause playback."""
           self.player.pause()
           self.is_paused = True
           print("Paused")

       def stop(self):
           """Stop playback."""
           self.player.stop()
           self.is_paused = False
           print("Stopped")

       def seek(self, position):
           """Seek to position (0.0 to 1.0)."""
           self.player.seek(position)
           print(f"Seeked to {position:.0%}")

       def get_status(self):
           """Get current playback status."""
           progress = self.player.get_progress()
           current = progress * self.player.duration
           return {
               'playing': self.player.is_playing(),
               'paused': self.is_paused,
               'progress': progress,
               'current_time': current,
               'duration': self.player.duration
           }

   def main():
       if len(sys.argv) < 2:
           print("Usage: python music_player.py <audio_file>")
           sys.exit(1)

       player = SimpleMusicPlayer()

       try:
           player.load(sys.argv[1])
           player.play()

           # Simple playback loop
           while player.player.is_playing():
               status = player.get_status()
               bar_width = 30
               filled = int(bar_width * status['progress'])
               bar = '=' * filled + '-' * (bar_width - filled)

               sys.stdout.write(
                   f"\r[{bar}] {status['current_time']:.1f}s / {status['duration']:.1f}s"
               )
               sys.stdout.flush()
               time.sleep(0.1)

           print("\nPlayback complete!")

       except FileNotFoundError as e:
           print(f"Error: {e}")
           sys.exit(1)
       except KeyboardInterrupt:
           print("\nInterrupted")
           player.stop()

   if __name__ == "__main__":
       main()

Next Steps
----------

- :doc:`audio_recording` - Record audio from input devices
- :doc:`../cookbook/audio_processing` - Process and manipulate audio
- :doc:`../cookbook/audiounit_hosting` - Use AudioUnit effects

See Also
--------

- :doc:`../api/index` - Complete API reference
- :doc:`../cookbook/real_time_audio` - Real-time audio techniques
