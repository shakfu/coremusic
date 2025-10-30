Real-Time Audio Recipes
=======================

Practical recipes for real-time audio processing.

Record Audio from Input Device
-------------------------------

Record audio from default input device using AudioQueue.

.. code-block:: python

   import coremusic as cm
   from pathlib import Path

   def record_audio(output_path, duration=5.0, sample_rate=44100.0):
       """Record audio from default input device"""
       format = cm.AudioFormat(
           sample_rate=sample_rate,
           format_id=cm.capi.fourchar_to_int('lpcm'),
           format_flags=cm.capi.get_linear_pcm_format_flag_is_float() |
                        cm.capi.get_linear_pcm_format_flag_is_packed(),
           bytes_per_packet=8,
           frames_per_packet=1,
           bytes_per_frame=8,
           channels_per_frame=2,
           bits_per_channel=32
       )

       queue = cm.AudioQueue.new_input(format)

       # Allocate buffers
       buffer_size = int(sample_rate * 0.5 * 8)  # 0.5 seconds
       for _ in range(3):
           queue.allocate_buffer(buffer_size)

       # Start recording
       queue.start()

       import time
       time.sleep(duration)

       # Stop and save
       queue.stop()

       # Get recorded data (simplified - actual implementation needs buffer handling)
       # For production use, implement proper buffer callback handling

   # Usage
   record_audio("recorded.wav", duration=5.0)

Play Audio with Low Latency
----------------------------

Achieve low-latency playback using AudioUnit output.

.. code-block:: python

   import coremusic as cm
   import numpy as np

   def play_low_latency(audio_data, sample_rate=44100.0):
       """Play audio with minimal latency using AudioUnit"""
       # Create default output unit
       unit = cm.AudioUnit.default_output()

       try:
           # Configure format
           format = cm.AudioFormat(
               sample_rate=sample_rate,
               format_id=cm.capi.fourchar_to_int('lpcm'),
               format_flags=cm.capi.get_linear_pcm_format_flag_is_float() |
                            cm.capi.get_linear_pcm_format_flag_is_packed(),
               channels_per_frame=2,
               bits_per_channel=32
           )

           unit.set_stream_format(format)

           # Set render callback
           def render_callback(action_flags, timestamp, bus_number,
                             num_frames, io_data):
               # Fill buffer with audio data
               # Simplified - production needs proper buffering
               return audio_data[:num_frames * 8]  # 2 channels * 4 bytes

           unit.set_render_callback(render_callback)

           # Start playback
           unit.initialize()
           unit.start()

           # Play for duration
           duration = len(audio_data) / (sample_rate * 2 * 4)
           import time
           time.sleep(duration)

           unit.stop()
       finally:
           unit.dispose()

Monitor Audio Levels
--------------------

Monitor real-time audio levels from input device.

.. code-block:: python

   import coremusic as cm
   import numpy as np

   class AudioLevelMonitor:
       """Real-time audio level monitoring"""

       def __init__(self, sample_rate=44100.0):
           self.sample_rate = sample_rate
           self.queue = None
           self.running = False
           self.peak_level = 0.0
           self.rms_level = 0.0

       def start(self):
           """Start monitoring"""
           format = cm.AudioFormat(
               sample_rate=self.sample_rate,
               format_id=cm.capi.fourchar_to_int('lpcm'),
               format_flags=cm.capi.get_linear_pcm_format_flag_is_float(),
               channels_per_frame=2,
               bits_per_channel=32
           )

           self.queue = cm.AudioQueue.new_input(format)

           # Set input callback to calculate levels
           def input_callback(data):
               samples = np.frombuffer(data, dtype=np.float32)
               self.peak_level = float(np.max(np.abs(samples)))
               self.rms_level = float(np.sqrt(np.mean(samples**2)))

           self.queue.set_callback(input_callback)
           self.queue.start()
           self.running = True

       def stop(self):
           """Stop monitoring"""
           if self.queue:
               self.queue.stop()
               self.queue.dispose()
           self.running = False

       def get_levels(self):
           """Get current levels in dB"""
           peak_db = 20 * np.log10(self.peak_level) if self.peak_level > 0 else -100
           rms_db = 20 * np.log10(self.rms_level) if self.rms_level > 0 else -100
           return {"peak_db": peak_db, "rms_db": rms_db}

   # Usage
   monitor = AudioLevelMonitor()
   monitor.start()

   import time
   for _ in range(10):
       levels = monitor.get_levels()
       print(f"Peak: {levels['peak_db']:.1f} dB, RMS: {levels['rms_db']:.1f} dB")
       time.sleep(0.1)

   monitor.stop()

Apply Real-Time Audio Effects
------------------------------

Apply effects to audio in real-time using AudioUnit.

.. code-block:: python

   import coremusic as cm

   class ReverbEffect:
       """Real-time reverb effect"""

       def __init__(self):
           self.output_unit = None
           self.reverb_unit = None

       def setup(self, sample_rate=44100.0):
           """Setup effect chain"""
           # Create output unit
           self.output_unit = cm.AudioUnit.default_output()

           # Create reverb unit
           reverb_desc = cm.AudioComponentDescription(
               type='aufx',  # kAudioUnitType_Effect
               subtype='rvb2',  # kAudioUnitSubType_MatrixReverb
               manufacturer='appl'
           )
           self.reverb_unit = cm.AudioUnit(reverb_desc)

           # Configure format
           format = cm.AudioFormat(
               sample_rate=sample_rate,
               format_id=cm.capi.fourchar_to_int('lpcm'),
               format_flags=cm.capi.get_linear_pcm_format_flag_is_float(),
               channels_per_frame=2,
               bits_per_channel=32
           )

           self.reverb_unit.set_stream_format(format, scope='input')
           self.reverb_unit.set_stream_format(format, scope='output')
           self.output_unit.set_stream_format(format)

           # Initialize
           self.reverb_unit.initialize()
           self.output_unit.initialize()

       def set_reverb_parameters(self, room_size=0.5, dry_wet_mix=0.3):
           """Configure reverb parameters"""
           # Set room size parameter (0.0 - 1.0)
           self.reverb_unit.set_parameter(
               parameter=0,  # Room size parameter
               value=room_size,
               scope='global'
           )

           # Set dry/wet mix (0.0 = dry, 1.0 = wet)
           self.reverb_unit.set_parameter(
               parameter=1,  # Mix parameter
               value=dry_wet_mix,
               scope='global'
           )

       def start(self):
           """Start processing"""
           self.output_unit.start()

       def stop(self):
           """Stop processing"""
           self.output_unit.stop()

       def cleanup(self):
           """Clean up resources"""
           if self.output_unit:
               self.output_unit.dispose()
           if self.reverb_unit:
               self.reverb_unit.dispose()

   # Usage
   reverb = ReverbEffect()
   reverb.setup()
   reverb.set_reverb_parameters(room_size=0.7, dry_wet_mix=0.4)
   reverb.start()

   import time
   time.sleep(5.0)  # Process for 5 seconds

   reverb.stop()
   reverb.cleanup()

See Also
--------

- :doc:`audio_processing` - Audio processing recipes
- :doc:`file_operations` - File I/O recipes
