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
6. AudioEffectsChain - High-level AUGraph wrapper for effect chains
7. Simple effect chain builder - Quick linear effect chain creation
8. AudioUnit FourCC codes - Reference guide for AudioUnit identification
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


def example_7_audio_effects_chain():
    """Example 7: Creating audio effects chains with AUGraph"""
    print_section("Example 7: Audio Effects Chain")

    print("Creating an audio effects chain using AudioEffectsChain wrapper")
    print("This demonstrates the high-level API for AUGraph management.\n")

    # Create effects chain
    print("1. Creating chain and adding nodes...")
    chain = cm.AudioEffectsChain()

    # Add a 3D mixer (commonly available)
    print("   - Adding 3D Mixer effect ('aumi', '3dem', 'appl')")
    mixer_node = chain.add_effect('aumi', '3dem', 'appl')
    print(f"     Node ID: {mixer_node}")

    # Add output
    print("   - Adding default output node")
    output_node = chain.add_output()
    print(f"     Node ID: {output_node}")

    print(f"\n2. Connecting nodes in chain...")
    chain.connect(mixer_node, output_node)
    print(f"   Connected: {mixer_node} -> {output_node}")

    print(f"\n3. Chain status:")
    print(f"   - Node count: {chain.node_count}")
    print(f"   - Is open: {chain.is_open}")
    print(f"   - Is initialized: {chain.is_initialized}")

    print(f"\n4. Opening and initializing chain...")
    try:
        chain.open()
        print(f"   - Is open: {chain.is_open}")

        chain.initialize()
        print(f"   - Is initialized: {chain.is_initialized}")

        print(f"\n5. Disposing chain...")
        chain.dispose()
        print("   Chain disposed successfully")
    except Exception as e:
        print(f"   Note: {e}")
        print("   (AudioUnit availability varies by system)")
        chain.dispose()

    # Example using context manager
    print("\n6. Using context manager for automatic cleanup:")
    with cm.AudioEffectsChain() as chain2:
        mixer = chain2.add_effect('aumi', '3dem', 'appl')
        output = chain2.add_output()
        chain2.connect(mixer, output)
        print(f"   Created chain with {chain2.node_count} nodes")
    print("   Chain automatically disposed")


def example_8_simple_effect_chain_builder():
    """Example 8: Using the simple chain builder"""
    print_section("Example 8: Simple Effect Chain Builder")

    print("Using create_simple_effect_chain() for quick chain setup\n")

    # Create a linear chain of effects
    print("Creating a chain with auto-connection:")
    print("  - 3D Mixer ('aumi', '3dem', 'appl')")
    print("  - Matrix Mixer ('aumi', 'mxmx', 'appl')")
    print("  - Default Output (auto-added)")

    chain = cm.create_simple_effect_chain([
        ('aumi', '3dem', 'appl'),   # 3D Mixer
        ('aumi', 'mxmx', 'appl'),   # Matrix Mixer
    ], auto_connect=True)

    print(f"\nChain created:")
    print(f"  - Node count: {chain.node_count}")
    print(f"  - All nodes auto-connected in linear chain")

    # Open and initialize
    print(f"\nOpening and initializing (using method chaining)...")
    try:
        chain.open().initialize()
        print(f"  - Is initialized: {chain.is_initialized}")
        chain.dispose()
        print("  - Chain disposed")
    except Exception as e:
        print(f"  Note: {e}")
        print("  (AudioUnit availability varies by system)")
        chain.dispose()


def example_9_audiounit_fourcc_reference():
    """Example 9: AudioUnit FourCC codes reference"""
    print_section("Example 9: AudioUnit FourCC Codes Reference")

    print("AudioUnits are identified by FourCC (4-character codes):")
    print("This allows precise specification without name lookup.\n")

    print("Common AudioUnit Types (type parameter):")
    print("  'auou' - Output units (speakers, system audio)")
    print("  'aumu' - Music effects (reverb, delay, etc.)")
    print("  'aufx' - Audio effects (EQ, compressor, etc.)")
    print("  'aumi' - Mixer units (3D mixer, matrix mixer)")
    print("  'aumf' - Music instruments (software synths)")
    print("  'aufc' - Format converter units")

    print("\nCommon AudioUnit Subtypes (subtype parameter):")
    print("\nOutput Units ('auou'):")
    print("  'def ' - Default output (system speakers)")
    print("  'sys ' - System output")
    print("  'genr' - Generic output")

    print("\nMixer Units ('aumi'):")
    print("  '3dem' - 3D Mixer")
    print("  'mxmx' - Matrix Mixer")
    print("  'mcmx' - Multichannel Mixer")

    print("\nMusic Effects ('aumu'):")
    print("  'rvb2' - Reverb 2")
    print("  'ddly' - Delay")
    print("  'dist' - Distortion")

    print("\nAudio Effects ('aufx'):")
    print("  'eqal' - AUGraphicEQ")
    print("  'dcmp' - Dynamics Processor")
    print("  'filt' - AUFilter")

    print("\nManufacturer Codes:")
    print("  'appl' - Apple (built-in AudioUnits)")

    print("\nUsage Example:")
    print("  ```python")
    print("  # Add a reverb effect")
    print("  reverb_node = chain.add_effect('aumu', 'rvb2', 'appl')")
    print("")
    print("  # Add an EQ effect")
    print("  eq_node = chain.add_effect('aufx', 'eqal', 'appl')")
    print("")
    print("  # Add a 3D mixer")
    print("  mixer_node = chain.add_effect('aumi', '3dem', 'appl')")
    print("  ```")

    print("\nNOTE: AudioUnit availability varies by macOS version.")
    print("Use AudioComponent.find_next() to check availability programmatically.")


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
        ("Audio Effects Chain", example_7_audio_effects_chain),
        ("Simple Effect Chain Builder", example_8_simple_effect_chain_builder),
        ("AudioUnit FourCC Reference", example_9_audiounit_fourcc_reference),
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
