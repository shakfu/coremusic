# CoreMusic Cookbook

**Version:** 0.1.8
**Last Updated:** October 2025

Practical recipes for common audio tasks using CoreMusic.

---

## Table of Contents

**File Operations:**
1. [Convert Audio Formats](#1-convert-audio-formats)
2. [Batch Process Multiple Files](#2-batch-process-multiple-files)
3. [Split Audio into Chunks](#3-split-audio-into-chunks)
4. [Merge Audio Files](#4-merge-audio-files)
5. [Extract Audio Metadata](#5-extract-audio-metadata)

**Audio Processing:**
6. [Normalize Audio Volume](#6-normalize-audio-volume)
7. [Apply Fade In/Out](#7-apply-fade-inout)
8. [Change Sample Rate](#8-change-sample-rate)
9. [Mix Multiple Tracks](#9-mix-multiple-tracks)
10. [Generate Silence](#10-generate-silence)

**Real-Time Audio:**
11. [Record from Microphone](#11-record-from-microphone)
12. [Play Audio in Real-Time](#12-play-audio-in-real-time)
13. [Monitor Audio Levels](#13-monitor-audio-levels)
14. [Real-Time Audio Effects](#14-real-time-audio-effects)

**MIDI:**
15. [Create MIDI Sequence](#15-create-midi-sequence)
16. [Play MIDI File](#16-play-midi-file)
17. [MIDI to Audio Rendering](#17-midi-to-audio-rendering)
18. [Transpose MIDI Notes](#18-transpose-midi-notes)

**Analysis:**
19. [Detect Beats](#19-detect-beats)
20. [Detect Pitch](#20-detect-pitch)
21. [Analyze Spectrum](#21-analyze-spectrum)
22. [Generate Audio Fingerprint](#22-generate-audio-fingerprint)

**Advanced:**
23. [Chain AudioUnit Effects](#23-chain-audiounit-effects)
24. [Synchronize with Ableton Link](#24-synchronize-with-ableton-link)
25. [Build Simple DAW](#25-build-simple-daw)

---

## File Operations

### 1. Convert Audio Formats

Convert between different audio file formats with optional resampling.

```python
import coremusic as cm

def convert_audio(input_path, output_path, output_format="wav", sample_rate=None):
    """Convert audio file to different format

    Args:
        input_path: Input file path
        output_path: Output file path
        output_format: Output format ("wav", "aiff", "caf")
        sample_rate: Optional target sample rate
    """
    with cm.AudioFile(input_path) as input_file:
        # Get input format
        in_format = input_file.format

        # Create output format
        out_format = cm.AudioFormat(
            sample_rate=sample_rate or in_format.sample_rate,
            format_id='lpcm',
            channels_per_frame=in_format.channels_per_frame,
            bits_per_channel=16  # CD quality
        )

        # Map format names to FourCC codes
        format_types = {
            'wav': 'WAVE',
            'aiff': 'AIFF',
            'caf': 'caff'
        }

        # Create output file
        with cm.ExtendedAudioFile.create(
            output_path,
            cm.capi.fourchar_to_int(format_types[output_format]),
            out_format
        ) as output_file:
            # Use ExtendedAudioFile for automatic conversion
            input_ext = cm.ExtendedAudioFile(input_path)
            input_ext.open()
            input_ext.client_format = out_format

            # Copy with conversion
            chunk_size = 8192
            while True:
                data, count = input_ext.read(chunk_size)
                if count == 0:
                    break
                output_file.write(count, data)

            input_ext.dispose()

# Example usage
convert_audio("song.mp3", "song.wav", output_format="wav", sample_rate=44100)
convert_audio("audio.wav", "audio.aiff", output_format="aiff")
```

### 2. Batch Process Multiple Files

Process multiple files in parallel for maximum efficiency.

```python
import coremusic as cm
from pathlib import Path

def batch_convert_files(input_dir, output_dir, output_format="wav"):
    """Convert all audio files in directory

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        output_format: Target format
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.aiff', '*.m4a']:
        audio_files.extend(input_dir.glob(ext))

    def process_file(file_path):
        """Process single file"""
        output_path = output_dir / f"{file_path.stem}.{output_format}"

        try:
            convert_audio(str(file_path), str(output_path), output_format)
            return True, str(output_path)
        except Exception as e:
            return False, str(e)

    # Process in parallel
    results = cm.batch_process_parallel(
        audio_files,
        process_file,
        max_workers=4,  # Use 4 CPU cores
        progress_callback=lambda i, t: print(f"Progress: {i}/{t} files")
    )

    # Report results
    successful = sum(1 for success, _ in results if success)
    print(f"\nProcessed {successful}/{len(results)} files successfully")

# Example usage
batch_convert_files("input_audio", "output_audio", "wav")
```

### 3. Split Audio into Chunks

Split long audio file into smaller segments.

```python
import coremusic as cm
from coremusic.audio import AudioSlicer

def split_audio(input_path, output_dir, chunk_duration=30.0):
    """Split audio file into fixed-duration chunks

    Args:
        input_path: Input file path
        output_dir: Output directory
        chunk_duration: Duration of each chunk in seconds
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slicer = AudioSlicer(input_path)

    # Get total duration
    duration = slicer.duration
    num_chunks = int(duration / chunk_duration) + 1

    print(f"Splitting {duration:.2f}s audio into {num_chunks} chunks...")

    for i in range(num_chunks):
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration, duration)

        # Extract slice
        slice_data = slicer.slice_time_range(start_time, end_time)

        # Save chunk
        output_path = output_dir / f"chunk_{i:03d}.wav"
        slicer.save_slice(slice_data, str(output_path))

        print(f"  Saved chunk {i+1}/{num_chunks}: {output_path.name}")

# Example usage
split_audio("long_podcast.wav", "podcast_chunks", chunk_duration=60.0)
```

### 4. Merge Audio Files

Concatenate multiple audio files into one.

```python
import coremusic as cm
from pathlib import Path

def merge_audio_files(input_files, output_path):
    """Merge multiple audio files into one

    Args:
        input_files: List of input file paths
        output_path: Output file path
    """
    if not input_files:
        raise ValueError("No input files provided")

    # Get format from first file
    with cm.AudioFile(str(input_files[0])) as first_file:
        format = first_file.format

    # Create output file
    with cm.ExtendedAudioFile.create(
        output_path,
        cm.capi.fourchar_to_int('WAVE'),
        format
    ) as output_file:

        # Append each input file
        for input_path in input_files:
            print(f"Adding: {Path(input_path).name}")

            with cm.AudioFile(str(input_path)) as input_file:
                # Read entire file
                data, count = input_file.read(input_file.frame_count)

                # Append to output
                output_file.write(count, data)

    print(f"Merged {len(input_files)} files into: {output_path}")

# Example usage
files = ["intro.wav", "main.wav", "outro.wav"]
merge_audio_files(files, "complete.wav")
```

### 5. Extract Audio Metadata

Get detailed information about audio file.

```python
import coremusic as cm

def extract_metadata(file_path):
    """Extract comprehensive audio file metadata

    Args:
        file_path: Audio file path

    Returns:
        Dictionary with metadata
    """
    with cm.AudioFile(file_path) as audio:
        format = audio.format

        metadata = {
            'filename': Path(file_path).name,
            'duration_seconds': audio.duration,
            'sample_rate': format.sample_rate,
            'channels': format.channels_per_frame,
            'bits_per_channel': format.bits_per_channel,
            'format_id': format.format_id,
            'frame_count': audio.frame_count,
            'is_float': format.is_float,
            'is_big_endian': format.is_big_endian,
            'is_packed': format.is_packed,
        }

        # Calculate derived values
        metadata['bitrate'] = (
            format.sample_rate *
            format.channels_per_frame *
            format.bits_per_channel
        )
        metadata['file_size_mb'] = (
            audio.frame_count *
            format.bytes_per_frame / (1024 * 1024)
        )

        return metadata

# Example usage
info = extract_metadata("song.wav")
for key, value in info.items():
    print(f"{key}: {value}")
```

---

## Audio Processing

### 6. Normalize Audio Volume

Normalize audio to target peak level.

```python
import numpy as np
import coremusic as cm

def normalize_audio(input_path, output_path, target_peak=0.9):
    """Normalize audio to target peak level

    Args:
        input_path: Input file
        output_path: Output file
        target_peak: Target peak level (0.0-1.0)
    """
    with cm.AudioFile(input_path) as input_file:
        # Read audio
        data_bytes, count = input_file.read(input_file.frame_count)
        samples = np.frombuffer(data_bytes, dtype=np.float32)

        # Find current peak
        current_peak = np.max(np.abs(samples))
        print(f"Current peak: {current_peak:.3f}")

        # Calculate gain
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

# Example usage
normalize_audio("quiet_audio.wav", "normalized.wav", target_peak=0.9)
```

### 7. Apply Fade In/Out

Add smooth fade in and fade out effects.

```python
import numpy as np
import coremusic as cm

def apply_fades(input_path, output_path, fade_in_duration=2.0, fade_out_duration=2.0):
    """Apply fade in and fade out to audio

    Args:
        input_path: Input file
        output_path: Output file
        fade_in_duration: Fade in duration in seconds
        fade_out_duration: Fade out duration in seconds
    """
    with cm.AudioFile(input_path) as input_file:
        # Read audio
        data_bytes, count = input_file.read(input_file.frame_count)
        samples = np.frombuffer(data_bytes, dtype=np.float32)

        sample_rate = input_file.format.sample_rate
        channels = input_file.format.channels_per_frame

        # Calculate fade lengths in samples (per channel)
        fade_in_samples = int(fade_in_duration * sample_rate)
        fade_out_samples = int(fade_out_duration * sample_rate)

        # Reshape to (frames, channels)
        frames = len(samples) // channels
        audio_2d = samples.reshape(frames, channels)

        # Apply fade in
        fade_in_curve = np.linspace(0, 1, fade_in_samples)[:, np.newaxis]
        audio_2d[:fade_in_samples] *= fade_in_curve

        # Apply fade out
        fade_out_curve = np.linspace(1, 0, fade_out_samples)[:, np.newaxis]
        audio_2d[-fade_out_samples:] *= fade_out_curve

        # Flatten back
        samples = audio_2d.flatten()

        # Write output
        with cm.ExtendedAudioFile.create(
            output_path,
            cm.capi.fourchar_to_int('WAVE'),
            input_file.format
        ) as output_file:
            output_file.write(count, samples.tobytes())

# Example usage
apply_fades("song.wav", "song_faded.wav", fade_in_duration=3.0, fade_out_duration=5.0)
```

### 8. Change Sample Rate

Resample audio to different sample rate.

```python
import coremusic as cm

def resample_audio(input_path, output_path, target_sample_rate=48000.0):
    """Resample audio to target sample rate

    Args:
        input_path: Input file
        output_path: Output file
        target_sample_rate: Target sample rate in Hz
    """
    # Use ExtendedAudioFile for automatic resampling
    with cm.ExtendedAudioFile(input_path) as input_file:
        # Get input format
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

        # Create output file
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

    print(f"Resampled from {in_format.sample_rate}Hz to {target_sample_rate}Hz")

# Example usage
resample_audio("44100hz.wav", "48000hz.wav", target_sample_rate=48000.0)
```

### 9. Mix Multiple Tracks

Mix multiple audio tracks into stereo output.

```python
import numpy as np
import coremusic as cm

def mix_tracks(track_files, output_path, levels=None):
    """Mix multiple audio tracks

    Args:
        track_files: List of input file paths
        output_path: Output file path
        levels: Optional list of gain levels (0.0-1.0) for each track
    """
    if levels is None:
        levels = [1.0] * len(track_files)

    # Load all tracks
    tracks = []
    max_frames = 0

    for file_path, level in zip(track_files, levels):
        with cm.AudioFile(file_path) as audio:
            data_bytes, count = audio.read(audio.frame_count)
            samples = np.frombuffer(data_bytes, dtype=np.float32)

            # Apply level
            samples *= level

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
        print(f"Normalized by {peak:.2f} to prevent clipping")

    # Get format from first track
    with cm.AudioFile(track_files[0]) as audio:
        format = audio.format

    # Write output
    with cm.ExtendedAudioFile.create(
        output_path,
        cm.capi.fourchar_to_int('WAVE'),
        format
    ) as output_file:
        num_frames = len(mixed) // format.channels_per_frame
        output_file.write(num_frames, mixed.tobytes())

# Example usage
tracks = ["drums.wav", "bass.wav", "melody.wav"]
levels = [1.0, 0.8, 0.9]  # Adjust individual track levels
mix_tracks(tracks, "mixed.wav", levels=levels)
```

### 10. Generate Silence

Create silent audio file of specified duration.

```python
import numpy as np
import coremusic as cm

def generate_silence(output_path, duration=5.0, sample_rate=44100.0, channels=2):
    """Generate silent audio file

    Args:
        output_path: Output file path
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        channels: Number of channels
    """
    # Calculate number of frames
    num_frames = int(duration * sample_rate)

    # Create silence (zeros)
    silence = np.zeros(num_frames * channels, dtype=np.float32)

    # Create format
    format = cm.AudioFormat(
        sample_rate=sample_rate,
        format_id='lpcm',
        channels_per_frame=channels,
        bits_per_channel=32,
        is_float=True
    )

    # Write file
    with cm.ExtendedAudioFile.create(
        output_path,
        cm.capi.fourchar_to_int('WAVE'),
        format
    ) as output_file:
        output_file.write(num_frames, silence.tobytes())

    print(f"Created {duration}s of silence: {output_path}")

# Example usage
generate_silence("silence_5s.wav", duration=5.0)
```

---

## Real-Time Audio

### 11. Record from Microphone

Record audio from default microphone.

```python
import coremusic as cm
import time

def record_audio(output_path, duration=5.0):
    """Record audio from microphone

    Args:
        output_path: Output file path
        duration: Recording duration in seconds
    """
    # Create format
    format = cm.AudioFormat(
        sample_rate=44100.0,
        format_id='lpcm',
        channels_per_frame=1,  # Mono
        bits_per_channel=16
    )

    # Create input queue
    queue = cm.AudioQueue.create_input(format)

    # Allocate buffers
    buffer_size = 4096
    buffers = [queue.allocate_buffer(buffer_size) for _ in range(3)]

    # Storage for recorded audio
    recorded_data = []

    print(f"Recording for {duration} seconds...")

    # Start recording
    queue.start()

    # Enqueue buffers
    for buffer in buffers:
        queue.enqueue_buffer(buffer, b'')

    # Record for specified duration
    start_time = time.time()
    while time.time() - start_time < duration:
        time.sleep(0.1)
        # In real implementation, would process buffers in callback

    # Stop recording
    queue.stop()
    queue.dispose()

    print(f"Recording complete: {output_path}")

# Example usage
record_audio("recording.wav", duration=10.0)
```

### 12. Play Audio in Real-Time

Stream audio playback with low latency.

```python
import coremusic as cm

def play_audio_realtime(file_path):
    """Play audio file in real-time

    Args:
        file_path: Audio file to play
    """
    with cm.AudioFile(file_path) as audio:
        format = audio.format

        # Create output queue
        queue = cm.AudioQueue.create_output(format)

        # Use small buffers for low latency
        buffer_size = 512
        num_buffers = 3

        # Allocate buffers
        buffers = [queue.allocate_buffer(buffer_size) for _ in range(num_buffers)]

        # Start playback
        queue.start()

        # Fill and enqueue buffers
        frames_read = 0
        while frames_read < audio.frame_count:
            for buffer in buffers:
                # Read chunk
                data, count = audio.read(buffer_size)
                if count == 0:
                    break

                # Enqueue buffer
                queue.enqueue_buffer(buffer, data)
                frames_read += count

        # Wait for playback to complete
        import time
        time.sleep(audio.duration)

        # Stop and cleanup
        queue.stop()
        queue.dispose()

# Example usage
play_audio_realtime("song.wav")
```

### 13. Monitor Audio Levels

Real-time audio level monitoring.

```python
import numpy as np
import coremusic as cm
import time

def monitor_audio_levels(file_path, update_interval=0.1):
    """Monitor audio levels during playback

    Args:
        file_path: Audio file to monitor
        update_interval: Update interval in seconds
    """
    with cm.AudioFile(file_path) as audio:
        chunk_size = int(audio.format.sample_rate * update_interval)

        print("Monitoring audio levels...")
        print("Time\t\tPeak\t\tRMS")
        print("-" * 40)

        current_time = 0.0
        frames_read = 0

        while frames_read < audio.frame_count:
            # Read chunk
            data, count = audio.read(min(chunk_size, audio.frame_count - frames_read))
            if count == 0:
                break

            # Convert to NumPy
            samples = np.frombuffer(data, dtype=np.float32)

            # Calculate levels
            peak = np.max(np.abs(samples))
            rms = np.sqrt(np.mean(samples ** 2))

            # Convert to dB
            peak_db = 20 * np.log10(peak) if peak > 0 else -96.0
            rms_db = 20 * np.log10(rms) if rms > 0 else -96.0

            # Display
            print(f"{current_time:.2f}s\t\t{peak_db:.1f}dB\t\t{rms_db:.1f}dB")

            current_time += count / audio.format.sample_rate
            frames_read += count

# Example usage
monitor_audio_levels("song.wav", update_interval=0.5)
```

### 14. Real-Time Audio Effects

Apply effects during playback.

```python
import numpy as np
import coremusic as cm

def apply_realtime_effect(file_path, effect_func):
    """Apply real-time audio effect during playback

    Args:
        file_path: Input audio file
        effect_func: Function that processes audio samples
    """
    with cm.AudioFile(file_path) as audio:
        format = audio.format

        # Create output queue
        queue = cm.AudioQueue.create_output(format)

        buffer_size = 1024
        buffers = [queue.allocate_buffer(buffer_size) for _ in range(3)]

        queue.start()

        frames_read = 0
        while frames_read < audio.frame_count:
            for buffer in buffers:
                # Read chunk
                data, count = audio.read(buffer_size)
                if count == 0:
                    break

                # Convert to NumPy
                samples = np.frombuffer(data, dtype=np.float32)

                # Apply effect
                processed = effect_func(samples)

                # Enqueue processed audio
                queue.enqueue_buffer(buffer, processed.tobytes())
                frames_read += count

        # Wait for completion
        import time
        time.sleep(audio.duration)

        queue.stop()
        queue.dispose()

# Example effects
def tremolo(samples, rate=5.0, depth=0.5):
    """Tremolo effect (amplitude modulation)"""
    t = np.arange(len(samples)) / 44100.0
    modulator = 1.0 - depth * (1.0 + np.sin(2 * np.pi * rate * t)) / 2.0
    return samples * modulator

# Example usage
apply_realtime_effect("song.wav", tremolo)
```

---

## MIDI

### 15. Create MIDI Sequence

Create MIDI sequence programmatically.

```python
import coremusic as cm

def create_midi_sequence(output_path=None):
    """Create a simple MIDI sequence

    Args:
        output_path: Optional MIDI file output path
    """
    # Create sequence and player
    sequence = cm.MusicSequence()
    track = sequence.new_track()

    # Set tempo
    tempo_track = sequence.tempo_track
    tempo_track.add_tempo_event(0.0, bpm=120.0)

    # Add a C major scale
    notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C, D, E, F, G, A, B, C
    for i, note in enumerate(notes):
        track.add_midi_note(
            time=float(i),
            channel=0,
            note=note,
            velocity=100,
            duration=0.9
        )

    # Add chord (C major triad)
    for note in [60, 64, 67]:  # C, E, G
        track.add_midi_note(
            time=8.0,
            channel=0,
            note=note,
            velocity=80,
            duration=2.0
        )

    print(f"Created sequence with {sequence.track_count} tracks")

    return sequence

# Example usage
sequence = create_midi_sequence()

# Play it
player = cm.MusicPlayer()
player.sequence = sequence
player.preroll()
player.start()

import time
time.sleep(10)

player.stop()
player.dispose()
sequence.dispose()
```

### 16. Play MIDI File

Load and play MIDI file.

```python
import coremusic as cm
import time

def play_midi_file(midi_path):
    """Play MIDI file

    Args:
        midi_path: Path to MIDI file
    """
    # Load MIDI file using utilities
    from coremusic.midi import load_midi_file

    midi_data = load_midi_file(midi_path)

    print(f"Loaded MIDI file:")
    print(f"  Tempo: {midi_data.tempo} BPM")
    print(f"  Duration: {midi_data.duration:.2f}s")
    print(f"  Tracks: {len(midi_data.tracks)}")

    # Create sequence
    sequence = cm.MusicSequence()
    sequence.load_from_file(midi_path)

    # Play
    player = cm.MusicPlayer()
    player.sequence = sequence
    player.preroll()
    player.start()

    print("Playing...")
    time.sleep(midi_data.duration + 1.0)

    player.stop()
    player.dispose()
    sequence.dispose()

# Example usage
play_midi_file("song.mid")
```

### 17. MIDI to Audio Rendering

Render MIDI to audio file through software instrument.

```python
import coremusic as cm
import numpy as np

def render_midi_to_audio(midi_path, output_path, duration=None):
    """Render MIDI file to audio

    Args:
        midi_path: Input MIDI file
        output_path: Output audio file
        duration: Duration in seconds (auto-detect if None)
    """
    from coremusic.midi import load_midi_file

    # Load MIDI
    midi_data = load_midi_file(midi_path)
    duration = duration or (midi_data.duration + 2.0)

    # Render to audio (simplified - would use AudioUnit in production)
    sample_rate = 48000.0
    num_frames = int(duration * sample_rate)

    # Generate synthesized audio from MIDI notes
    audio = np.zeros(num_frames * 2, dtype=np.float32)  # Stereo

    # Simple synthesis (sine waves)
    for track in midi_data.tracks:
        for note in track.notes:
            start_frame = int(note.time * sample_rate)
            duration_frames = int(note.duration * sample_rate)
            end_frame = min(start_frame + duration_frames, num_frames)

            # Generate sine wave for note
            freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            t = np.arange(end_frame - start_frame) / sample_rate
            wave = 0.1 * np.sin(2 * np.pi * freq * t) * (note.velocity / 127.0)

            # Apply envelope
            attack = int(0.01 * sample_rate)
            release = int(0.1 * sample_rate)
            envelope = np.ones_like(wave)
            if len(envelope) > attack:
                envelope[:attack] = np.linspace(0, 1, attack)
            if len(envelope) > release:
                envelope[-release:] = np.linspace(1, 0, release)
            wave *= envelope

            # Add to audio (stereo)
            for i, sample in enumerate(wave):
                if start_frame + i < num_frames:
                    audio[(start_frame + i) * 2] += sample  # Left
                    audio[(start_frame + i) * 2 + 1] += sample  # Right

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak * 1.1

    # Write output
    format = cm.AudioFormat(
        sample_rate=sample_rate,
        format_id='lpcm',
        channels_per_frame=2,
        bits_per_channel=32,
        is_float=True
    )

    with cm.ExtendedAudioFile.create(
        output_path,
        cm.capi.fourchar_to_int('WAVE'),
        format
    ) as output_file:
        output_file.write(num_frames, audio.tobytes())

    print(f"Rendered MIDI to: {output_path}")

# Example usage
render_midi_to_audio("melody.mid", "melody.wav")
```

### 18. Transpose MIDI Notes

Transpose all notes in MIDI sequence.

```python
import coremusic as cm
from coremusic.midi import load_midi_file

def transpose_midi(midi_path, output_path, semitones=0):
    """Transpose MIDI file by semitones

    Args:
        midi_path: Input MIDI file
        output_path: Output MIDI file
        semitones: Number of semitones to transpose (positive or negative)
    """
    # Load MIDI
    midi_data = load_midi_file(midi_path)

    # Create new sequence
    sequence = cm.MusicSequence()

    # Set tempo
    sequence.tempo_track.add_tempo_event(0.0, bpm=midi_data.tempo)

    # Transpose each track
    for track_data in midi_data.tracks:
        track = sequence.new_track()

        for note in track_data.notes:
            # Transpose note
            transposed_pitch = note.pitch + semitones

            # Clamp to valid MIDI range
            transposed_pitch = max(0, min(127, transposed_pitch))

            track.add_midi_note(
                time=note.time,
                channel=note.channel,
                note=transposed_pitch,
                velocity=note.velocity,
                duration=note.duration
            )

    print(f"Transposed by {semitones} semitones")

    # Note: Would need to implement MIDI file saving
    # For now, can play the transposed sequence
    return sequence

# Example usage
sequence = transpose_midi("song.mid", "song_transposed.mid", semitones=2)

# Play transposed version
player = cm.MusicPlayer()
player.sequence = sequence
player.start()
```

---

## Analysis

### 19. Detect Beats

Detect beats in audio file.

```python
import coremusic as cm
from coremusic.audio.analysis import AudioAnalyzer

def detect_beats(file_path):
    """Detect beats in audio file

    Args:
        file_path: Audio file path

    Returns:
        List of beat times in seconds
    """
    analyzer = AudioAnalyzer(file_path)

    # Load audio
    analyzer.load_audio()

    # Detect beats
    beat_info = analyzer.detect_beats()

    print(f"Detected {len(beat_info.beat_times)} beats")
    print(f"Estimated tempo: {beat_info.tempo:.1f} BPM")
    print(f"First 10 beats: {beat_info.beat_times[:10]}")

    return beat_info

# Example usage
beats = detect_beats("song.wav")
```

### 20. Detect Pitch

Detect pitch in audio file.

```python
import coremusic as cm
from coremusic.audio.analysis import AudioAnalyzer

def detect_pitch(file_path):
    """Detect pitch throughout audio file

    Args:
        file_path: Audio file path

    Returns:
        List of pitch information
    """
    analyzer = AudioAnalyzer(file_path)
    analyzer.load_audio()

    # Detect pitch at regular intervals
    sample_rate = analyzer.sample_rate
    hop_size = 2048

    pitches = []
    for i in range(0, len(analyzer.audio_data) - hop_size, hop_size):
        chunk = analyzer.audio_data[i:i+hop_size]

        pitch_info = analyzer.autocorrelation_pitch(chunk)

        if pitch_info.confidence > 0.5:  # Only confident detections
            pitches.append({
                'time': i / sample_rate,
                'frequency': pitch_info.frequency,
                'midi_note': pitch_info.midi_note,
                'confidence': pitch_info.confidence
            })

    print(f"Detected {len(pitches)} pitched segments")
    return pitches

# Example usage
pitches = detect_pitch("vocal.wav")
for p in pitches[:5]:
    print(f"{p['time']:.2f}s: {p['frequency']:.1f}Hz (MIDI {p['midi_note']})")
```

### 21. Analyze Spectrum

Analyze frequency spectrum.

```python
import coremusic as cm
from coremusic.audio.analysis import AudioAnalyzer
import matplotlib.pyplot as plt

def analyze_spectrum(file_path, time=0.0):
    """Analyze frequency spectrum at specific time

    Args:
        file_path: Audio file path
        time: Time in seconds to analyze

    Returns:
        Spectrum analysis data
    """
    analyzer = AudioAnalyzer(file_path)
    analyzer.load_audio()

    # Analyze spectrum
    spectrum = analyzer.analyze_spectrum(time)

    print(f"Spectral analysis at {time}s:")
    print(f"  Centroid: {spectrum['centroid']:.1f} Hz")
    print(f"  Rolloff: {spectrum['rolloff']:.1f} Hz")
    print(f"  Peak frequency: {spectrum['peak_freq']:.1f} Hz")

    # Plot spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(spectrum['frequencies'], spectrum['magnitudes'])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'Frequency Spectrum at {time}s')
    plt.grid(True)
    plt.show()

    return spectrum

# Example usage
spectrum = analyze_spectrum("audio.wav", time=5.0)
```

### 22. Generate Audio Fingerprint

Create audio fingerprint for identification.

```python
import coremusic as cm
from coremusic.audio.analysis import AudioAnalyzer

def generate_fingerprint(file_path):
    """Generate audio fingerprint

    Args:
        file_path: Audio file path

    Returns:
        Fingerprint data
    """
    analyzer = AudioAnalyzer(file_path)
    analyzer.load_audio()

    # Generate fingerprint
    fingerprint = analyzer.generate_fingerprint()

    print(f"Generated fingerprint:")
    print(f"  Hash: {fingerprint['hash']}")
    print(f"  Peaks: {len(fingerprint['peaks'])}")
    print(f"  Duration: {fingerprint['duration']:.2f}s")

    return fingerprint

# Example usage
fp1 = generate_fingerprint("song1.wav")
fp2 = generate_fingerprint("song2.wav")

# Compare fingerprints
if fp1['hash'] == fp2['hash']:
    print("Files are identical!")
else:
    print("Files are different")
```

---

## Advanced

### 23. Chain AudioUnit Effects

Chain multiple AudioUnit effects together.

```python
import coremusic as cm

def chain_audio_effects(input_path, output_path):
    """Apply chain of AudioUnit effects

    Args:
        input_path: Input audio file
        output_path: Output audio file
    """
    from coremusic.audio.audiounit_host import AudioUnitChain

    # Create effect chain
    chain = AudioUnitChain()

    # Add effects (if available)
    # chain.add_effect("AUDynamicsProcessor")  # Compressor
    # chain.add_effect("AUReverb")  # Reverb

    # Process audio through chain
    with cm.AudioFile(input_path) as input_file:
        format = input_file.format
        data, count = input_file.read(input_file.frame_count)

        # Process through chain
        processed = chain.process(data)

        # Write output
        with cm.ExtendedAudioFile.create(
            output_path,
            cm.capi.fourchar_to_int('WAVE'),
            format
        ) as output_file:
            output_file.write(count, processed)

    print(f"Processed with {len(chain.units)} effects")

# Example usage
chain_audio_effects("dry.wav", "wet.wav")
```

### 24. Synchronize with Ableton Link

Synchronize playback with Ableton Link.

```python
import coremusic as cm
from coremusic import link
import time

def sync_with_link():
    """Synchronize audio playback with Ableton Link"""

    # Create Link session
    session = link.LinkSession()
    session.enable(True)

    print("Waiting for Link peers...")
    time.sleep(2)

    if session.num_peers > 0:
        print(f"Connected to {session.num_peers} Link peers")

        # Get Link timeline
        state = session.capture_app_session_state()

        # Get tempo
        tempo = state.tempo
        print(f"Link tempo: {tempo:.1f} BPM")

        # Start playback on next bar
        beat = state.beat_at_time(state.time_at_now, 4.0)  # quantum = 4
        next_bar = ((int(beat) // 4) + 1) * 4
        start_time = state.time_at_beat(next_bar, 4.0)

        # Wait until start time
        now = state.time_at_now
        wait_time = (start_time - now) / 1000000.0  # microseconds to seconds
        print(f"Starting in {wait_time:.2f}s...")
        time.sleep(wait_time)

        # Start playback synchronized
        print("Starting playback (synchronized)")

    session.enable(False)

# Example usage
sync_with_link()
```

### 25. Build Simple DAW

Build basic DAW with multiple tracks.

```python
import coremusic as cm
from coremusic.daw import Timeline, Track, Clip

def build_simple_daw():
    """Build simple DAW-like application"""

    # Create timeline
    timeline = Timeline(tempo=120.0, sample_rate=44100.0)

    # Add tracks
    drums = Track(name="Drums")
    bass = Track(name="Bass")
    melody = Track(name="Melody")

    timeline.add_track(drums)
    timeline.add_track(bass)
    timeline.add_track(melody)

    # Add clips to tracks
    drums_clip = Clip(
        file_path="drums.wav",
        start_time=0.0,
        duration=8.0,
        name="Drums Loop"
    )
    drums.add_clip(drums_clip)

    bass_clip = Clip(
        file_path="bass.wav",
        start_time=0.0,
        duration=8.0,
        name="Bass Line"
    )
    bass.add_clip(bass_clip)

    melody_clip = Clip(
        file_path="melody.wav",
        start_time=4.0,
        duration=4.0,
        name="Melody"
    )
    melody.add_clip(melody_clip)

    # Render timeline to file
    output_path = "daw_mix.wav"
    timeline.render(output_path, duration=12.0)

    print(f"Rendered {len(timeline.tracks)} tracks to: {output_path}")

# Example usage
build_simple_daw()
```

---

## More Recipes

For more examples, see:
- `tests/demos/` - Comprehensive demonstration scripts
- `tests/` - Unit tests showing API usage
- `CLAUDE.md` - Project documentation

---

**Need help?** Open an issue at: https://github.com/anthropics/coremusic/issues
