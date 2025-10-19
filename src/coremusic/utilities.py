#!/usr/bin/env python3
"""High-level audio processing utilities for CoreMusic.

This module provides convenient, high-level utilities for common audio
processing tasks, built on top of the CoreAudio APIs.

Features:
- Audio analysis (silence detection, peak detection, RMS calculation)
- Batch file processing and conversion
- Format conversion helpers
- File metadata extraction
"""

from typing import Optional, Union, List, Tuple, Dict, Any, Callable
from pathlib import Path
import glob

from .objects import (
    AudioFile,
    AudioConverter,
    AudioFormat,
    ExtendedAudioFile,
    AudioFileError,
    AudioConverterError,
    NUMPY_AVAILABLE,
)

if NUMPY_AVAILABLE:
    import numpy as np
    from numpy.typing import NDArray


# ============================================================================
# Audio Analysis Utilities
# ============================================================================

class AudioAnalyzer:
    """High-level audio analysis utilities.

    Provides convenient methods for common audio analysis tasks such as
    silence detection, peak detection, and RMS calculation.

    Example:
        ```python
        import coremusic as cm

        # Detect silence regions
        silence_regions = cm.AudioAnalyzer.detect_silence("audio.wav", threshold_db=-40)

        # Get peak amplitude
        peak = cm.AudioAnalyzer.get_peak_amplitude("audio.wav")

        # Calculate RMS
        rms = cm.AudioAnalyzer.calculate_rms("audio.wav")
        ```
    """

    @staticmethod
    def detect_silence(
        audio_file: Union[str, Path, AudioFile],
        threshold_db: float = -40.0,
        min_duration: float = 0.5
    ) -> List[Tuple[float, float]]:
        """Detect silence regions in an audio file.

        Args:
            audio_file: Path to audio file or AudioFile instance
            threshold_db: Silence threshold in dB (default: -40)
            min_duration: Minimum silence duration in seconds (default: 0.5)

        Returns:
            List of (start_time, end_time) tuples for silence regions

        Example:
            ```python
            # Find all silence regions quieter than -40dB lasting at least 0.5s
            silence = AudioAnalyzer.detect_silence("audio.wav", threshold_db=-40, min_duration=0.5)
            for start, end in silence:
                print(f"Silence from {start:.2f}s to {end:.2f}s")
            ```
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for silence detection. Install with: pip install numpy")

        # Open file if path provided
        should_close = False
        if isinstance(audio_file, (str, Path)):
            audio_file = AudioFile(str(audio_file))
            audio_file.open()
            should_close = True

        try:
            # Read audio data as NumPy array
            audio_data = audio_file.read_as_numpy()

            # Get format info
            format = audio_file.format
            sample_rate = format.sample_rate

            # Convert to mono if stereo by taking mean across channels
            if audio_data.ndim == 2:
                audio_data = np.mean(audio_data, axis=1)

            # Convert to float and normalize
            if audio_data.dtype in [np.int16, np.int32]:
                max_val = np.iinfo(audio_data.dtype).max
                audio_data = audio_data.astype(np.float32) / max_val

            # Convert threshold from dB to linear
            threshold_linear = 10 ** (threshold_db / 20)

            # Find samples below threshold
            is_silent = np.abs(audio_data) < threshold_linear

            # Find silence regions
            silence_regions = []
            in_silence = False
            silence_start = 0

            for i, silent in enumerate(is_silent):
                if silent and not in_silence:
                    # Start of silence region
                    in_silence = True
                    silence_start = i
                elif not silent and in_silence:
                    # End of silence region
                    in_silence = False
                    duration = (i - silence_start) / sample_rate
                    if duration >= min_duration:
                        start_time = silence_start / sample_rate
                        end_time = i / sample_rate
                        silence_regions.append((start_time, end_time))

            # Handle case where file ends in silence
            if in_silence:
                duration = (len(is_silent) - silence_start) / sample_rate
                if duration >= min_duration:
                    start_time = silence_start / sample_rate
                    end_time = len(is_silent) / sample_rate
                    silence_regions.append((start_time, end_time))

            return silence_regions

        finally:
            if should_close:
                audio_file.close()

    @staticmethod
    def get_peak_amplitude(audio_file: Union[str, Path, AudioFile]) -> float:
        """Get the peak amplitude of an audio file.

        Args:
            audio_file: Path to audio file or AudioFile instance

        Returns:
            Peak amplitude as a float (0.0 to 1.0 for normalized audio)

        Example:
            ```python
            peak = AudioAnalyzer.get_peak_amplitude("audio.wav")
            print(f"Peak amplitude: {peak:.4f}")
            ```
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for peak detection. Install with: pip install numpy")

        # Open file if path provided
        should_close = False
        if isinstance(audio_file, (str, Path)):
            audio_file = AudioFile(str(audio_file))
            audio_file.open()
            should_close = True

        try:
            # Read audio data
            audio_data = audio_file.read_as_numpy()

            # Convert to float and normalize
            if audio_data.dtype in [np.int16, np.int32]:
                max_val = np.iinfo(audio_data.dtype).max
                audio_data = audio_data.astype(np.float32) / max_val

            # Get peak
            return float(np.max(np.abs(audio_data)))

        finally:
            if should_close:
                audio_file.close()

    @staticmethod
    def calculate_rms(audio_file: Union[str, Path, AudioFile]) -> float:
        """Calculate RMS (Root Mean Square) amplitude.

        Args:
            audio_file: Path to audio file or AudioFile instance

        Returns:
            RMS amplitude as a float

        Example:
            ```python
            rms = AudioAnalyzer.calculate_rms("audio.wav")
            rms_db = 20 * np.log10(rms)  # Convert to dB
            print(f"RMS: {rms:.4f} ({rms_db:.2f} dB)")
            ```
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for RMS calculation. Install with: pip install numpy")

        # Open file if path provided
        should_close = False
        if isinstance(audio_file, (str, Path)):
            audio_file = AudioFile(str(audio_file))
            audio_file.open()
            should_close = True

        try:
            # Read audio data
            audio_data = audio_file.read_as_numpy()

            # Convert to float and normalize
            if audio_data.dtype in [np.int16, np.int32]:
                max_val = np.iinfo(audio_data.dtype).max
                audio_data = audio_data.astype(np.float32) / max_val

            # Calculate RMS
            return float(np.sqrt(np.mean(audio_data ** 2)))

        finally:
            if should_close:
                audio_file.close()

    @staticmethod
    def get_file_info(audio_file: Union[str, Path]) -> Dict[str, Any]:
        """Get comprehensive information about an audio file.

        Args:
            audio_file: Path to audio file

        Returns:
            Dictionary with file information (format, duration, sample_rate, etc.)

        Example:
            ```python
            info = AudioAnalyzer.get_file_info("audio.wav")
            print(f"Duration: {info['duration']:.2f}s")
            print(f"Format: {info['format_id']}")
            print(f"Sample Rate: {info['sample_rate']} Hz")
            ```
        """
        with AudioFile(str(audio_file)) as af:
            format = af.format

            info = {
                'path': str(audio_file),
                'duration': af.duration,
                'sample_rate': format.sample_rate,
                'format_id': format.format_id,
                'channels': format.channels_per_frame,
                'bits_per_channel': format.bits_per_channel,
                'is_pcm': format.is_pcm,
                'is_stereo': format.is_stereo,
                'is_mono': format.is_mono,
            }

            # Add peak and RMS if NumPy available
            if NUMPY_AVAILABLE:
                try:
                    info['peak_amplitude'] = AudioAnalyzer.get_peak_amplitude(af)
                    info['rms'] = AudioAnalyzer.calculate_rms(af)
                except Exception:
                    pass  # Skip if reading fails

            return info


# ============================================================================
# Batch Processing Utilities
# ============================================================================

def batch_convert(
    input_pattern: str,
    output_format: AudioFormat,
    output_dir: Optional[str] = None,
    output_extension: str = "wav",
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str, int, int], None]] = None
) -> List[str]:
    """Batch convert audio files to a specified format.

    Args:
        input_pattern: Glob pattern for input files (e.g., "*.mp3", "audio/*.wav")
        output_format: Target AudioFormat for conversion
        output_dir: Output directory (default: same as input)
        output_extension: Output file extension (default: "wav")
        overwrite: Whether to overwrite existing files (default: False)
        progress_callback: Optional callback(filename, current, total)

    Returns:
        List of successfully converted output file paths

    Example:
        ```python
        import coremusic as cm

        # Convert all MP3 files to 16-bit 44.1kHz WAV
        output_format = cm.AudioFormat(
            sample_rate=44100.0,
            format_id='lpcm',
            format_flags=12,
            bytes_per_packet=4,
            frames_per_packet=1,
            bytes_per_frame=4,
            channels_per_frame=2,
            bits_per_channel=16
        )

        converted = cm.batch_convert(
            input_pattern="input/*.mp3",
            output_format=output_format,
            output_dir="output/",
            progress_callback=lambda f, c, t: print(f"Converting {f} ({c}/{t})")
        )
        print(f"Converted {len(converted)} files")
        ```
    """
    # Find all matching files
    input_files = glob.glob(input_pattern)
    if not input_files:
        return []

    total_files = len(input_files)
    converted_files = []

    for i, input_path in enumerate(input_files, 1):
        input_path = Path(input_path)

        # Determine output path
        if output_dir:
            output_path = Path(output_dir) / f"{input_path.stem}.{output_extension}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = input_path.with_suffix(f".{output_extension}")

        # Skip if file exists and not overwriting
        if output_path.exists() and not overwrite:
            continue

        # Call progress callback
        if progress_callback:
            progress_callback(str(input_path), i, total_files)

        # Convert file
        try:
            convert_audio_file(str(input_path), str(output_path), output_format)
            converted_files.append(str(output_path))
        except Exception as e:
            print(f"Warning: Failed to convert {input_path}: {e}")
            continue

    return converted_files


def convert_audio_file(
    input_path: str,
    output_path: str,
    output_format: AudioFormat
) -> None:
    """Convert a single audio file to a different format.

    Note: This is a simplified implementation that works best for WAV files.
    For complex format conversions, use AudioConverter directly.

    Args:
        input_path: Input file path
        output_path: Output file path
        output_format: Target AudioFormat

    Example:
        ```python
        import coremusic as cm

        # Convert to mono
        output_format = cm.AudioFormatPresets.wav_44100_mono()
        cm.convert_audio_file("input.wav", "output.wav", output_format)
        ```
    """
    # Read source file
    with AudioFile(input_path) as input_file:
        source_format = input_file.format

        # If formats match exactly, just copy
        if (source_format.sample_rate == output_format.sample_rate and
            source_format.channels_per_frame == output_format.channels_per_frame and
            source_format.bits_per_channel == output_format.bits_per_channel):
            import shutil
            shutil.copy(input_path, output_path)
            return

        # For simple conversions (stereo -> mono), use AudioConverter
        if (source_format.sample_rate == output_format.sample_rate and
            source_format.bits_per_channel == output_format.bits_per_channel and
            source_format.channels_per_frame != output_format.channels_per_frame):

            # Read all audio data
            audio_data, packet_count = input_file.read_packets(0, 999999999)

            # Convert using AudioConverter
            with AudioConverter(source_format, output_format) as converter:
                converted_data = converter.convert(audio_data)

            # Calculate number of frames from converted data
            num_frames = len(converted_data) // output_format.bytes_per_frame

            # Write to output file
            from . import capi
            output_ext_file = ExtendedAudioFile.create(
                output_path,
                capi.get_audio_file_wave_type(),
                output_format
            )
            try:
                output_ext_file.write(num_frames, converted_data)
            finally:
                output_ext_file.close()
        else:
            # For other conversions, use basic copy (user should use AudioConverter directly)
            raise NotImplementedError(
                f"Complex format conversion not yet supported in utilities. "
                f"Use AudioConverter directly for sample rate or bit depth changes."
            )


# ============================================================================
# Format Conversion Helpers
# ============================================================================

class AudioFormatPresets:
    """Common audio format presets for easy format conversion.

    Example:
        ```python
        import coremusic as cm

        # Use predefined formats
        cm.convert_audio_file("input.mp3", "output.wav", cm.AudioFormatPresets.wav_44100_stereo())
        cm.convert_audio_file("input.wav", "output_mono.wav", cm.AudioFormatPresets.wav_44100_mono())
        ```
    """

    @staticmethod
    def wav_44100_stereo() -> AudioFormat:
        """Standard CD quality WAV: 44.1kHz, 16-bit, stereo"""
        return AudioFormat(
            sample_rate=44100.0,
            format_id='lpcm',
            format_flags=12,  # kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked
            bytes_per_packet=4,
            frames_per_packet=1,
            bytes_per_frame=4,
            channels_per_frame=2,
            bits_per_channel=16
        )

    @staticmethod
    def wav_44100_mono() -> AudioFormat:
        """Mono WAV: 44.1kHz, 16-bit, mono"""
        return AudioFormat(
            sample_rate=44100.0,
            format_id='lpcm',
            format_flags=12,
            bytes_per_packet=2,
            frames_per_packet=1,
            bytes_per_frame=2,
            channels_per_frame=1,
            bits_per_channel=16
        )

    @staticmethod
    def wav_48000_stereo() -> AudioFormat:
        """Pro audio WAV: 48kHz, 16-bit, stereo"""
        return AudioFormat(
            sample_rate=48000.0,
            format_id='lpcm',
            format_flags=12,
            bytes_per_packet=4,
            frames_per_packet=1,
            bytes_per_frame=4,
            channels_per_frame=2,
            bits_per_channel=16
        )

    @staticmethod
    def wav_96000_stereo() -> AudioFormat:
        """High-res audio WAV: 96kHz, 24-bit, stereo"""
        return AudioFormat(
            sample_rate=96000.0,
            format_id='lpcm',
            format_flags=12,
            bytes_per_packet=6,
            frames_per_packet=1,
            bytes_per_frame=6,
            channels_per_frame=2,
            bits_per_channel=24
        )


# ============================================================================
# Audio File Operations
# ============================================================================

def trim_audio(
    input_path: str,
    output_path: str,
    start_time: float = 0.0,
    end_time: Optional[float] = None
) -> None:
    """Trim an audio file to a specific time range.

    Args:
        input_path: Input file path
        output_path: Output file path
        start_time: Start time in seconds (default: 0.0)
        end_time: End time in seconds (default: end of file)

    Example:
        ```python
        # Extract first 30 seconds of audio
        cm.trim_audio("input.wav", "output.wav", start_time=0.0, end_time=30.0)

        # Skip first 10 seconds
        cm.trim_audio("input.wav", "output.wav", start_time=10.0)
        ```
    """
    with AudioFile(input_path) as input_file:
        format = input_file.format
        sample_rate = format.sample_rate

        # Calculate packet range
        start_packet = int(start_time * sample_rate)

        if end_time is not None:
            end_packet = int(end_time * sample_rate)
            packet_count = end_packet - start_packet
        else:
            packet_count = None  # Read to end

        # Read trimmed data
        data, actual_count = input_file.read_packets(start_packet, packet_count or 999999999)

        # Write to output file using ExtendedAudioFile
        from . import capi
        output_ext_file = ExtendedAudioFile.create(
            output_path,
            capi.get_audio_file_wave_type(),  # WAV file
            format
        )
        try:
            output_ext_file.write(actual_count, data)
        finally:
            output_ext_file.close()
