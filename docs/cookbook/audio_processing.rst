Audio Processing Recipes
========================

Practical recipes for common audio processing tasks.

Normalize Audio Volume
----------------------

Normalize audio to target peak level.

.. code-block:: python

   import numpy as np
   import coremusic as cm

   def normalize_audio(input_path, output_path, target_peak=0.9):
       """Normalize audio to target peak level"""
       with cm.AudioFile(input_path) as input_file:
           # Read audio
           data_bytes, count = input_file.read(input_file.frame_count)
           samples = np.frombuffer(data_bytes, dtype=np.float32)

           # Find current peak
           current_peak = np.max(np.abs(samples))

           # Calculate and apply gain
           if current_peak > 0:
               gain = target_peak / current_peak
               samples *= gain
               print(f"Applied gain: {gain:.3f}x ({20*np.log10(gain):.2f}dB)")

           # Write output
           with cm.ExtendedAudioFile.create(
               output_path,
               cm.capi.fourchar_to_int('WAVE'),
               input_file.format
           ) as output_file:
               output_file.write(count, samples.tobytes())

   # Usage
   normalize_audio("quiet_audio.wav", "normalized.wav", target_peak=0.9)

Apply Fade In/Out
-----------------

Add smooth fade effects to audio.

.. code-block:: python

   import numpy as np
   import coremusic as cm

   def apply_fades(input_path, output_path, fade_in_duration=2.0, fade_out_duration=2.0):
       """Apply fade in and fade out to audio"""
       with cm.AudioFile(input_path) as input_file:
           data_bytes, count = input_file.read(input_file.frame_count)
           samples = np.frombuffer(data_bytes, dtype=np.float32)

           sample_rate = input_file.format.sample_rate
           channels = input_file.format.channels_per_frame

           # Calculate fade lengths
           fade_in_samples = int(fade_in_duration * sample_rate)
           fade_out_samples = int(fade_out_duration * sample_rate)

           # Reshape to (frames, channels)
           frames = len(samples) // channels
           audio_2d = samples.reshape(frames, channels)

           # Apply fades
           fade_in_curve = np.linspace(0, 1, fade_in_samples)[:, np.newaxis]
           audio_2d[:fade_in_samples] *= fade_in_curve

           fade_out_curve = np.linspace(1, 0, fade_out_samples)[:, np.newaxis]
           audio_2d[-fade_out_samples:] *= fade_out_curve

           samples = audio_2d.flatten()

           # Write output
           with cm.ExtendedAudioFile.create(
               output_path,
               cm.capi.fourchar_to_int('WAVE'),
               input_file.format
           ) as output_file:
               output_file.write(count, samples.tobytes())

Change Sample Rate
------------------

Resample audio to different sample rate using ExtendedAudioFile.

.. code-block:: python

   import coremusic as cm

   def resample_audio(input_path, output_path, target_sample_rate=48000.0):
       """Resample audio with automatic conversion"""
       with cm.ExtendedAudioFile(input_path) as input_file:
           in_format = input_file.file_format

           # Create output format with new sample rate
           out_format = cm.AudioFormat(
               sample_rate=target_sample_rate,
               format_id=in_format.format_id,
               format_flags=in_format.format_flags,
               channels_per_frame=in_format.channels_per_frame,
               bits_per_channel=in_format.bits_per_channel
           )

           # Set client format for automatic conversion
           input_file.client_format = out_format

           with cm.ExtendedAudioFile.create(
               output_path,
               cm.capi.fourchar_to_int('WAVE'),
               out_format
           ) as output_file:
               # Copy with automatic resampling
               chunk_size = 8192
               while True:
                   data, count = input_file.read(chunk_size)
                   if count == 0:
                       break
                   output_file.write(count, data)

Mix Multiple Tracks
-------------------

Mix multiple audio tracks into stereo output.

.. code-block:: python

   import numpy as np
   import coremusic as cm

   def mix_tracks(track_files, output_path, levels=None):
       """Mix multiple audio tracks with individual levels"""
       if levels is None:
           levels = [1.0] * len(track_files)

       # Load all tracks
       tracks = []
       max_frames = 0

       for file_path, level in zip(track_files, levels):
           with cm.AudioFile(file_path) as audio:
               data_bytes, count = audio.read(audio.frame_count)
               samples = np.frombuffer(data_bytes, dtype=np.float32)
               samples *= level  # Apply level
               tracks.append(samples)
               max_frames = max(max_frames, len(samples))

       # Pad tracks to same length
       for i in range(len(tracks)):
           if len(tracks[i]) < max_frames:
               tracks[i] = np.pad(tracks[i], (0, max_frames - len(tracks[i])))

       # Mix (sum all tracks)
       mixed = np.sum(tracks, axis=0)

       # Normalize to prevent clipping
       peak = np.max(np.abs(mixed))
       if peak > 1.0:
           mixed /= peak

       # Write output
       with cm.AudioFile(track_files[0]) as audio:
           format = audio.format

       with cm.ExtendedAudioFile.create(
           output_path,
           cm.capi.fourchar_to_int('WAVE'),
           format
       ) as output_file:
           num_frames = len(mixed) // format.channels_per_frame
           output_file.write(num_frames, mixed.tobytes())

   # Usage
   tracks = ["drums.wav", "bass.wav", "melody.wav"]
   levels = [1.0, 0.8, 0.9]
   mix_tracks(tracks, "mixed.wav", levels=levels)

Split Audio into Chunks
-----------------------

Split long audio file into smaller segments.

.. code-block:: python

   import coremusic as cm
   from coremusic.audio import AudioSlicer
   from pathlib import Path

   def split_audio(input_path, output_dir, chunk_duration=30.0):
       """Split audio file into fixed-duration chunks"""
       output_dir = Path(output_dir)
       output_dir.mkdir(parents=True, exist_ok=True)

       slicer = AudioSlicer(input_path)
       duration = slicer.duration
       num_chunks = int(duration / chunk_duration) + 1

       for i in range(num_chunks):
           start_time = i * chunk_duration
           end_time = min((i + 1) * chunk_duration, duration)

           # Extract slice
           slice_data = slicer.slice_time_range(start_time, end_time)

           # Save chunk
           output_path = output_dir / f"chunk_{i:03d}.wav"
           slicer.save_slice(slice_data, str(output_path))

Merge Audio Files
-----------------

Concatenate multiple audio files into one.

.. code-block:: python

   import coremusic as cm

   def merge_audio_files(input_files, output_path):
       """Merge multiple audio files sequentially"""
       with cm.AudioFile(str(input_files[0])) as first_file:
           format = first_file.format

       with cm.ExtendedAudioFile.create(
           output_path,
           cm.capi.fourchar_to_int('WAVE'),
           format
       ) as output_file:
           for input_path in input_files:
               with cm.AudioFile(str(input_path)) as input_file:
                   data, count = input_file.read(input_file.frame_count)
                   output_file.write(count, data)

   # Usage
   files = ["intro.wav", "main.wav", "outro.wav"]
   merge_audio_files(files, "complete.wav")

See Also
--------

- :doc:`file_operations` - File I/O recipes
- :doc:`/guides/performance` - Performance optimization
- :doc:`/api/audio_file` - AudioFile API reference
