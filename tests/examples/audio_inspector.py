#!/usr/bin/env python3
"""
Audio File Inspector

A comprehensive tool to inspect and display audio file information using coremusic.

Usage: python audio_inspector.py <audio_file>
"""

import coremusic as cm
import sys
from pathlib import Path


def format_bytes(num_bytes):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def format_duration(seconds):
    """Format duration as MM:SS.ms."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def get_quality_classification(fmt):
    """Classify audio quality based on format."""
    if fmt.sample_rate == 44100 and fmt.bits_per_channel == 16:
        return "CD Quality"
    elif fmt.sample_rate >= 96000:
        return "Hi-Res Audio"
    elif fmt.sample_rate >= 48000:
        return "Professional Audio"
    else:
        return "Standard Audio"


def get_channel_description(channels):
    """Get human-readable channel description."""
    channel_map = {
        1: "Mono",
        2: "Stereo",
        4: "Quadraphonic",
        6: "5.1 Surround",
        8: "7.1 Surround"
    }
    return channel_map.get(channels, f"{channels}-channel")


def inspect_audio_file(filepath):
    """Comprehensively inspect and display audio file information."""
    # Check file exists
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}")
        return False

    print("=" * 70)
    print(f"Audio File Inspector")
    print("=" * 70)
    print()

    # File information
    print("FILE INFORMATION")
    print("-" * 70)
    print(f"  Filename:     {path.name}")
    print(f"  Path:         {path.absolute()}")
    print(f"  File Size:    {format_bytes(path.stat().st_size)}")
    print()

    try:
        with cm.AudioFile(str(filepath)) as audio:
            fmt = audio.format

            # Format information
            print("FORMAT INFORMATION")
            print("-" * 70)
            print(f"  Format ID:    {fmt.format_id}")
            print(f"  Sample Rate:  {fmt.sample_rate:,.0f} Hz")
            print(f"  Channels:     {fmt.channels_per_frame} ({get_channel_description(fmt.channels_per_frame)})")
            print(f"  Bit Depth:    {fmt.bits_per_channel}-bit")
            print(f"  Bytes/Frame:  {fmt.bytes_per_frame}")
            print(f"  Bytes/Packet: {fmt.bytes_per_packet}")
            print(f"  Frames/Packet: {fmt.frames_per_packet}")
            print(f"  Format Flags: 0x{fmt.format_flags:08X}")
            print()

            # Duration information
            print("DURATION INFORMATION")
            print("-" * 70)
            # Calculate total frames from duration and sample rate
            total_frames = int(audio.duration * fmt.sample_rate)
            print(f"  Total Frames: {total_frames:,}")
            print(f"  Duration:     {format_duration(audio.duration)} ({audio.duration:.3f}s)")
            print(f"  Minutes:      {audio.duration / 60:.2f}")
            print()

            # Quality classification
            print("CLASSIFICATION")
            print("-" * 70)
            quality = get_quality_classification(fmt)
            print(f"  Quality:      {quality}")
            print(f"  Channel Type: {get_channel_description(fmt.channels_per_frame)}")

            # Calculate bitrate
            bitrate = (fmt.sample_rate * fmt.bytes_per_frame * 8) / 1000
            print(f"  Bitrate:      {bitrate:,.0f} kbps")
            print()

            # Format-specific information
            print("FORMAT DETAILS")
            print("-" * 70)
            if fmt.format_id == 'lpcm':
                print(f"  Format Type:  Linear PCM (Uncompressed)")
                is_float = fmt.format_flags & 0x01
                is_big_endian = fmt.format_flags & 0x02
                is_signed = fmt.format_flags & 0x04
                is_packed = fmt.format_flags & 0x08

                print(f"  Data Type:    {'Float' if is_float else 'Integer'}")
                print(f"  Byte Order:   {'Big Endian' if is_big_endian else 'Little Endian'}")
                print(f"  Signed:       {'Yes' if is_signed else 'No'}")
                print(f"  Packed:       {'Yes' if is_packed else 'No'}")
            elif fmt.format_id in ['aac ', '.mp3', 'alac', 'flac']:
                format_names = {
                    'aac ': 'AAC (Advanced Audio Coding) - Compressed',
                    '.mp3': 'MP3 (MPEG-1 Audio Layer 3) - Compressed',
                    'alac': 'Apple Lossless (ALAC) - Compressed Lossless',
                    'flac': 'FLAC (Free Lossless Audio Codec) - Compressed Lossless'
                }
                print(f"  Format Type:  {format_names.get(fmt.format_id, 'Unknown')}")
            else:
                print(f"  Format Type:  Unknown format '{fmt.format_id}'")
            print()

            # Sample format calculations
            print("SAMPLE INFORMATION")
            print("-" * 70)
            total_samples = total_frames * fmt.channels_per_frame
            print(f"  Total Samples:    {total_samples:,}")
            print(f"  Samples/Second:   {int(fmt.sample_rate * fmt.channels_per_frame):,}")

            # Memory requirements
            memory_size = total_frames * fmt.bytes_per_frame
            print(f"  Memory Required:  {format_bytes(memory_size)}")
            print()

            # Data rate
            print("DATA RATE")
            print("-" * 70)
            bytes_per_second = fmt.sample_rate * fmt.bytes_per_frame
            print(f"  Bytes/Second:     {format_bytes(bytes_per_second)}/s")
            print(f"  Bits/Second:      {bitrate:,.0f} kbps")
            print()

            print("=" * 70)
            print("Inspection complete!")
            print("=" * 70)

            return True

    except cm.AudioFileError as e:
        print(f"\nError opening audio file: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python audio_inspector.py <audio_file>")
        print()
        print("Examples:")
        print("  python audio_inspector.py audio.wav")
        print("  python audio_inspector.py ~/Music/song.mp3")
        sys.exit(1)

    filepath = sys.argv[1]
    success = inspect_audio_file(filepath)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
