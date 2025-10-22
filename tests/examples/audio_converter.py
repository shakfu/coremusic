#!/usr/bin/env python3
"""
Audio Format Converter

Convert audio files between different sample rates and formats using coremusic.

Usage: python audio_converter.py <input_file> <output_file> [--rate RATE]
"""

import coremusic as cm
import sys
import argparse
from pathlib import Path


def convert_audio_file(input_path, output_path, target_sample_rate=None):
    """
    Convert audio file to different sample rate.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        target_sample_rate: Target sample rate (None = keep original)

    Returns:
        True if successful, False otherwise
    """
    print(f"Converting: {input_path}")
    print(f"Output:     {output_path}")
    print()

    try:
        # Open input file
        with cm.AudioFile(input_path) as input_audio:
            src_fmt = input_audio.format

            print("Source Format:")
            print(f"  Sample Rate:  {src_fmt.sample_rate} Hz")
            print(f"  Channels:     {src_fmt.channels_per_frame}")
            print(f"  Bit Depth:    {src_fmt.bits_per_channel}-bit")
            print(f"  Format:       {src_fmt.format_id}")
            print(f"  Duration:     {input_audio.duration:.2f}s")
            print()

            # Determine target format
            if target_sample_rate is None:
                target_sample_rate = src_fmt.sample_rate

            # Create target format (same as source but different sample rate)
            dst_fmt = cm.AudioFormat(
                sample_rate=float(target_sample_rate),
                format_id=src_fmt.format_id,
                format_flags=src_fmt.format_flags,
                bytes_per_packet=src_fmt.bytes_per_packet,
                frames_per_packet=src_fmt.frames_per_packet,
                bytes_per_frame=src_fmt.bytes_per_frame,
                channels_per_frame=src_fmt.channels_per_frame,
                bits_per_channel=src_fmt.bits_per_channel,
            )

            print("Target Format:")
            print(f"  Sample Rate:  {dst_fmt.sample_rate} Hz")
            print(f"  Channels:     {dst_fmt.channels_per_frame}")
            print(f"  Bit Depth:    {dst_fmt.bits_per_channel}-bit")
            print(f"  Format:       {dst_fmt.format_id}")
            print()

            # Check if conversion is needed
            if src_fmt.sample_rate == dst_fmt.sample_rate:
                print("Note: Source and target sample rates are the same")
                print("No conversion needed, but will still process the file")
                print()

            # Use the high-level conversion utility
            print("Converting...")
            try:
                capi.convert_audio_file(input_path, output_path, dst_fmt)
                print("Conversion complete!")
                print()

                # Verify output
                print("Verifying output...")
                with cm.AudioFile(output_path) as output_audio:
                    out_fmt = output_audio.format
                    print(f"  Output Sample Rate: {out_fmt.sample_rate} Hz")
                    print(f"  Output Duration:    {output_audio.duration:.2f}s")

                    # Check duration is preserved
                    duration_diff = abs(input_audio.duration - output_audio.duration)
                    if duration_diff < 0.01:
                        print(f"  Duration Preserved: Yes (diff: {duration_diff:.6f}s)")
                    else:
                        print(f"  WARNING: Duration changed by {duration_diff:.3f}s")

                return True

            except Exception as e:
                print(f"Error during conversion: {e}")
                return False

    except cm.AudioFileError as e:
        print(f"Audio file error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert audio files between different sample rates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to 48kHz
  python audio_converter.py input.wav output.wav --rate 48000

  # Convert to 44.1kHz
  python audio_converter.py input.wav output.wav --rate 44100

  # Copy without conversion (same sample rate)
  python audio_converter.py input.wav output.wav
        """,
    )

    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument("--rate", type=float, help="Target sample rate (Hz)")

    args = parser.parse_args()

    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Check if output file exists
    if Path(args.output).exists():
        response = input(f"Output file '{args.output}' exists. Overwrite? [y/N] ")
        if response.lower() != "y":
            print("Cancelled.")
            sys.exit(0)

    # Perform conversion
    success = convert_audio_file(args.input, args.output, args.rate)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
