#!/usr/bin/env python3
"""Demonstration of CoreMusic high-level audio utilities.

This script demonstrates the high-level convenience utilities for audio
processing, analysis, and batch conversion built on top of CoreMusic's
CoreAudio APIs.

Features demonstrated:
1. AudioAnalyzer - Silence detection, peak detection, RMS calculation
2. AudioFormatPresets - Common audio format presets
3. File conversion - Stereo to mono conversion
4. Batch processing - Converting multiple files
5. File info extraction - Comprehensive audio file metadata
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import coremusic as cm


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def example_1_file_info():
    """Example 1: Extract comprehensive file information"""
    print_section("Example 1: Extract Audio File Information")

    test_file = "tests/amen.wav"
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    # Get comprehensive file info
    info = cm.AudioAnalyzer.get_file_info(test_file)

    print(f"File: {info['path']}")
    print(f"Duration: {info['duration']:.2f} seconds")
    print(f"Sample Rate: {info['sample_rate']} Hz")
    print(f"Format: {info['format_id']}")
    print(f"Channels: {info['channels']} ({'stereo' if info['is_stereo'] else 'mono'})")
    print(f"Bits per channel: {info['bits_per_channel']}")
    print(f"Is PCM: {info['is_pcm']}")

    if cm.NUMPY_AVAILABLE:
        print(f"\nAudio Analysis (requires NumPy):")
        print(f"Peak amplitude: {info['peak_amplitude']:.4f}")
        print(f"RMS level: {info['rms']:.4f}")
        rms_db = 20 * __import__('numpy').log10(info['rms']) if info['rms'] > 0 else -float('inf')
        print(f"RMS (dB): {rms_db:.2f} dB")
    else:
        print("\nNumPy not available - install with: pip install numpy")


def example_2_audio_analysis():
    """Example 2: Audio analysis - silence detection, peak, RMS"""
    print_section("Example 2: Audio Analysis")

    test_file = "tests/amen.wav"
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    if not cm.NUMPY_AVAILABLE:
        print("NumPy is required for audio analysis")
        print("Install with: pip install numpy")
        return

    print(f"Analyzing: {test_file}\n")

    # Detect silence regions
    print("Detecting silence regions (threshold: -40 dB, min duration: 0.1s)...")
    silence_regions = cm.AudioAnalyzer.detect_silence(
        test_file,
        threshold_db=-40.0,
        min_duration=0.1
    )

    if silence_regions:
        print(f"Found {len(silence_regions)} silence region(s):")
        for i, (start, end) in enumerate(silence_regions, 1):
            print(f"  {i}. {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
    else:
        print("No silence regions found")

    # Get peak amplitude
    peak = cm.AudioAnalyzer.get_peak_amplitude(test_file)
    print(f"\nPeak amplitude: {peak:.4f}")
    print(f"Peak (dB): {20 * __import__('numpy').log10(peak):.2f} dB")

    # Calculate RMS
    rms = cm.AudioAnalyzer.calculate_rms(test_file)
    print(f"\nRMS level: {rms:.4f}")
    rms_db = 20 * __import__('numpy').log10(rms) if rms > 0 else -float('inf')
    print(f"RMS (dB): {rms_db:.2f} dB")

    # Using AudioFile object directly
    print("\nUsing AudioFile object for analysis:")
    with cm.AudioFile(test_file) as audio:
        peak_obj = cm.AudioAnalyzer.get_peak_amplitude(audio)
        rms_obj = cm.AudioAnalyzer.calculate_rms(audio)
        print(f"Peak: {peak_obj:.4f}, RMS: {rms_obj:.4f}")


def example_3_format_presets():
    """Example 3: Common audio format presets"""
    print_section("Example 3: Audio Format Presets")

    print("Available format presets:\n")

    # WAV 44.1kHz stereo (CD quality)
    format_44_stereo = cm.AudioFormatPresets.wav_44100_stereo()
    print(f"1. CD Quality WAV (44.1kHz stereo):")
    print(f"   {format_44_stereo}")

    # WAV 44.1kHz mono
    format_44_mono = cm.AudioFormatPresets.wav_44100_mono()
    print(f"\n2. WAV 44.1kHz mono:")
    print(f"   {format_44_mono}")

    # WAV 48kHz stereo (pro audio)
    format_48_stereo = cm.AudioFormatPresets.wav_48000_stereo()
    print(f"\n3. Pro Audio WAV (48kHz stereo):")
    print(f"   {format_48_stereo}")

    # WAV 96kHz stereo (high-res)
    format_96_stereo = cm.AudioFormatPresets.wav_96000_stereo()
    print(f"\n4. High-Res WAV (96kHz stereo, 24-bit):")
    print(f"   {format_96_stereo}")


def example_4_file_conversion():
    """Example 4: Convert stereo to mono"""
    print_section("Example 4: Audio File Conversion (Stereo to Mono)")

    test_file = "tests/amen.wav"
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "mono_output.wav")

        print(f"Input: {test_file}")
        print(f"Output: {output_file}")

        # Get input format
        with cm.AudioFile(test_file) as audio:
            input_format = audio.format
            print(f"\nInput format: {input_format}")

        # Convert to mono
        output_format = cm.AudioFormatPresets.wav_44100_mono()
        print(f"Target format: {output_format}")

        print("\nConverting...")
        cm.convert_audio_file(test_file, output_file, output_format)

        # Verify output
        with cm.AudioFile(output_file) as audio:
            output_actual = audio.format
            print(f"\nConversion successful!")
            print(f"Output format: {output_actual}")
            print(f"Output size: {os.path.getsize(output_file)} bytes")
            print(f"Duration: {audio.duration:.2f} seconds")


def example_5_batch_conversion():
    """Example 5: Batch convert multiple files"""
    print_section("Example 5: Batch Conversion")

    test_file = "tests/amen.wav"
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a few test files by copying
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir)

        print("Creating test files...")
        for i in range(1, 4):
            shutil.copy(test_file, os.path.join(input_dir, f"test_{i}.wav"))

        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")

        # Define conversion format
        output_format = cm.AudioFormatPresets.wav_44100_mono()
        print(f"\nTarget format: {output_format}")

        # Progress callback
        def progress_callback(filename, current, total):
            print(f"  Converting {os.path.basename(filename)} ({current}/{total})")

        # Batch convert
        print("\nBatch converting...")
        converted = cm.batch_convert(
            input_pattern=f"{input_dir}/*.wav",
            output_format=output_format,
            output_dir=output_dir,
            output_extension="wav",
            progress_callback=progress_callback
        )

        print(f"\nConverted {len(converted)} files:")
        for filepath in converted:
            size = os.path.getsize(filepath)
            print(f"  - {os.path.basename(filepath)} ({size} bytes)")


def example_6_integration_workflow():
    """Example 6: Complete workflow - analyze, convert, verify"""
    print_section("Example 6: Complete Workflow")

    test_file = "tests/amen.wav"
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    if not cm.NUMPY_AVAILABLE:
        print("NumPy is required for this example")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        print("STEP 1: Analyze original file")
        print("-" * 40)

        original_info = cm.AudioAnalyzer.get_file_info(test_file)
        print(f"File: {os.path.basename(test_file)}")
        print(f"Format: {original_info['sample_rate']} Hz, {original_info['channels']} channels")
        print(f"Duration: {original_info['duration']:.2f}s")
        print(f"Peak: {original_info['peak_amplitude']:.4f}")
        print(f"RMS: {original_info['rms']:.4f}")

        print("\nSTEP 2: Convert to mono")
        print("-" * 40)

        output_file = os.path.join(temp_dir, "converted_mono.wav")
        cm.convert_audio_file(
            test_file,
            output_file,
            cm.AudioFormatPresets.wav_44100_mono()
        )
        print(f"Converted to: {output_file}")

        print("\nSTEP 3: Analyze converted file")
        print("-" * 40)

        converted_info = cm.AudioAnalyzer.get_file_info(output_file)
        print(f"Format: {converted_info['sample_rate']} Hz, {converted_info['channels']} channels")
        print(f"Duration: {converted_info['duration']:.2f}s")
        print(f"Peak: {converted_info['peak_amplitude']:.4f}")
        print(f"RMS: {converted_info['rms']:.4f}")

        print("\nSTEP 4: Compare results")
        print("-" * 40)

        original_size = os.path.getsize(test_file)
        converted_size = os.path.getsize(output_file)
        size_reduction = (1 - converted_size / original_size) * 100

        print(f"Original size: {original_size} bytes")
        print(f"Converted size: {converted_size} bytes")
        print(f"Size reduction: {size_reduction:.1f}%")

        peak_diff = abs(original_info['peak_amplitude'] - converted_info['peak_amplitude'])
        peak_diff_pct = (peak_diff / original_info['peak_amplitude']) * 100
        print(f"Peak difference: {peak_diff_pct:.1f}%")


def main():
    """Run all examples"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║           CoreMusic High-Level Utilities Demonstration              ║
║                                                                      ║
║  This demo shows the high-level audio processing utilities built    ║
║  on top of CoreMusic's CoreAudio APIs.                              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Check if test file exists
    if not os.path.exists("tests/amen.wav"):
        print("\nWARNING: Test audio file 'tests/amen.wav' not found.")
        print("Some examples may be skipped.\n")

    # Run examples
    examples = [
        ("File Information", example_1_file_info),
        ("Audio Analysis", example_2_audio_analysis),
        ("Format Presets", example_3_format_presets),
        ("File Conversion", example_4_file_conversion),
        ("Batch Conversion", example_5_batch_conversion),
        ("Integration Workflow", example_6_integration_workflow),
    ]

    for i, (name, func) in enumerate(examples, 1):
        try:
            func()
        except Exception as e:
            print(f"\nError in example {i} ({name}): {e}")
            import traceback
            traceback.print_exc()

    print_section("Demo Complete")
    print("All examples completed successfully!\n")

    if not cm.NUMPY_AVAILABLE:
        print("NOTE: NumPy is not installed. Some analysis features are unavailable.")
        print("Install NumPy to unlock full audio analysis capabilities:")
        print("  pip install numpy\n")


if __name__ == "__main__":
    main()
