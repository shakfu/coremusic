#!/usr/bin/env python3
"""Demo script showcasing async I/O functionality in CoreMusic.

This script demonstrates:
1. Async file reading with chunk streaming
2. Async AudioQueue operations
3. Concurrent file processing
4. Real-world async audio processing pipeline
"""

import asyncio
import sys
from pathlib import Path

import coremusic as cm


# ============================================================================
# Example 1: Basic Async File Reading
# ============================================================================

async def example_basic_file_reading():
    """Demonstrate basic async file reading."""
    print("\n" + "="*70)
    print("Example 1: Basic Async File Reading")
    print("="*70)

    # Open file with async context manager
    async with cm.AsyncAudioFile("tests/amen.wav") as audio:
        print(f"File: {audio.path}")
        print(f"Duration: {audio.duration:.2f} seconds")
        print(f"Format: {audio.format.format_id}")
        print(f"Sample Rate: {audio.format.sample_rate} Hz")
        print(f"Channels: {audio.format.channels_per_frame}")
        print(f"Bits per Channel: {audio.format.bits_per_channel}")

        # Read some packets asynchronously
        data, packet_count = await audio.read_packets_async(start_packet=0, packet_count=100)
        print(f"\nRead {packet_count} packets ({len(data)} bytes)")


# ============================================================================
# Example 2: Streaming Large Files in Chunks
# ============================================================================

async def example_streaming_chunks():
    """Demonstrate streaming audio data in chunks."""
    print("\n" + "="*70)
    print("Example 2: Streaming Large Files in Chunks")
    print("="*70)

    total_bytes = 0
    chunk_count = 0

    async with cm.AsyncAudioFile("tests/amen.wav") as audio:
        print(f"Streaming file: {audio.path}")
        print(f"Duration: {audio.duration:.2f}s\n")

        # Stream in chunks without blocking
        async for chunk in audio.read_chunks_async(chunk_size=1024):
            total_bytes += len(chunk)
            chunk_count += 1

            # Simulate async processing
            await asyncio.sleep(0.001)

            if chunk_count % 10 == 0:
                print(f"Processed {chunk_count} chunks ({total_bytes:,} bytes)...")

        print(f"\nTotal: {chunk_count} chunks, {total_bytes:,} bytes")


# ============================================================================
# Example 3: Async AudioQueue Playback
# ============================================================================

async def example_audio_queue():
    """Demonstrate async AudioQueue operations."""
    print("\n" + "="*70)
    print("Example 3: Async AudioQueue Playback")
    print("="*70)

    # Create audio format
    format = cm.AudioFormat(
        sample_rate=44100.0,
        format_id='lpcm',
        format_flags=12,  # kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked
        bytes_per_packet=4,
        frames_per_packet=1,
        bytes_per_frame=4,
        channels_per_frame=2,
        bits_per_channel=16
    )

    print(f"Creating audio queue: {format.sample_rate} Hz, {format.channels_per_frame} channels")

    async with await cm.AsyncAudioQueue.new_output_async(format) as queue:
        print("Audio queue created successfully")

        # Allocate buffers
        buffer1 = await queue.allocate_buffer_async(4096)
        buffer2 = await queue.allocate_buffer_async(4096)
        print(f"Allocated {len(queue._queue._buffers)} buffers")

        # Start the queue
        await queue.start_async()
        print("Queue started")

        # Let it run briefly
        await asyncio.sleep(0.2)

        # Stop the queue
        await queue.stop_async()
        print("Queue stopped")


# ============================================================================
# Example 4: Concurrent File Processing
# ============================================================================

async def process_audio_file(file_path: str, file_id: int):
    """Process a single audio file (simulated)."""
    async with cm.AsyncAudioFile(file_path) as audio:
        chunks_processed = 0

        async for chunk in audio.read_chunks_async(chunk_size=2048):
            # Simulate async processing
            await asyncio.sleep(0.002)
            chunks_processed += 1

        return {
            'file_id': file_id,
            'path': file_path,
            'duration': audio.duration,
            'chunks': chunks_processed
        }


async def example_concurrent_processing():
    """Demonstrate concurrent processing of multiple files."""
    print("\n" + "="*70)
    print("Example 4: Concurrent File Processing")
    print("="*70)

    # Process the same file multiple times concurrently (simulating batch processing)
    file_path = "tests/amen.wav"

    print(f"Processing file 3 times concurrently: {file_path}\n")

    # Process files concurrently
    results = await asyncio.gather(
        process_audio_file(file_path, 1),
        process_audio_file(file_path, 2),
        process_audio_file(file_path, 3)
    )

    # Display results
    for result in results:
        print(f"File {result['file_id']}: "
              f"{result['duration']:.2f}s, "
              f"{result['chunks']} chunks processed")


# ============================================================================
# Example 5: Real-World Processing Pipeline
# ============================================================================

async def example_processing_pipeline():
    """Demonstrate a real-world async audio processing pipeline."""
    print("\n" + "="*70)
    print("Example 5: Real-World Processing Pipeline")
    print("="*70)

    async def analyze_chunk(chunk: bytes, chunk_id: int):
        """Simulate async chunk analysis (e.g., feature extraction)."""
        await asyncio.sleep(0.001)  # Simulate async work
        return {
            'chunk_id': chunk_id,
            'size': len(chunk),
            'processed': True
        }

    async def save_results(results):
        """Simulate async saving of results."""
        await asyncio.sleep(0.01)
        print(f"  Saved results for {len(results)} chunks")

    print("Processing pipeline: Read -> Analyze -> Save\n")

    results = []
    chunk_id = 0

    async with cm.AsyncAudioFile("tests/amen.wav") as audio:
        print(f"File: {audio.path} ({audio.duration:.2f}s)")

        # Process chunks as they come in
        async for chunk in audio.read_chunks_async(chunk_size=2048):
            # Analyze chunk asynchronously
            result = await analyze_chunk(chunk, chunk_id)
            results.append(result)
            chunk_id += 1

            # Save results in batches
            if len(results) >= 10:
                await save_results(results)
                results = []

        # Save remaining results
        if results:
            await save_results(results)

    print(f"\nProcessed {chunk_id} chunks total")


# ============================================================================
# Example 6: NumPy Integration (if available)
# ============================================================================

async def example_numpy_integration():
    """Demonstrate async NumPy integration."""
    if not cm.NUMPY_AVAILABLE:
        print("\n" + "="*70)
        print("Example 6: NumPy Integration")
        print("="*70)
        print("NumPy not available - skipping")
        return

    import numpy as np

    print("\n" + "="*70)
    print("Example 6: NumPy Integration")
    print("="*70)

    async with cm.AsyncAudioFile("tests/amen.wav") as audio:
        print(f"Reading audio as NumPy arrays...\n")

        chunk_count = 0
        total_max = 0.0

        # Stream audio as NumPy arrays
        async for chunk in audio.read_chunks_numpy_async(chunk_size=2048):
            # Compute max amplitude
            max_amplitude = float(np.max(np.abs(chunk)))
            total_max = max(total_max, max_amplitude)

            chunk_count += 1

            if chunk_count % 5 == 0:
                print(f"Chunk {chunk_count}: shape={chunk.shape}, max_amp={max_amplitude:.4f}")

            # Limit for demo
            if chunk_count >= 10:
                break

        print(f"\nProcessed {chunk_count} chunks")
        print(f"Max amplitude across all chunks: {total_max:.4f}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("CoreMusic Async I/O Demo")
    print("="*70)

    # Check if test file exists
    if not Path("tests/amen.wav").exists():
        print("\nError: Test file 'tests/amen.wav' not found")
        print("Please run this demo from the project root directory")
        sys.exit(1)

    # Run all examples
    await example_basic_file_reading()
    await example_streaming_chunks()
    await example_audio_queue()
    await example_concurrent_processing()
    await example_processing_pipeline()
    await example_numpy_integration()

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
