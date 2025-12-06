#!/usr/bin/env python3
"""Demo: Real-Time Audio Streaming

Demonstrates the streaming capabilities of CoreMusic including:
- Audio input/output streams
- Real-time audio processing
- Stream graphs with multiple effects
- Latency management

NOTE: These examples require audio hardware to function. They demonstrate
the API but actual audio I/O requires AudioUnit implementation (currently TODO).
"""

import sys
import time
from pathlib import Path

# Add src to path for demo purposes
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import coremusic as cm
from coremusic.audio.streaming import (
    AudioInputStream,
    AudioOutputStream,
    AudioProcessor,
    StreamGraph,
    create_loopback,
)

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available - some demos will be skipped")


def demo_input_stream():
    """Demo 1: Simple audio input stream."""
    print("\n" + "=" * 70)
    print("DEMO 1: Audio Input Stream")
    print("=" * 70)

    if not NUMPY_AVAILABLE:
        print("Skipping: Requires NumPy")
        return

    print("\nCreating input stream...")
    stream = AudioInputStream(
        channels=2, sample_rate=44100.0, buffer_size=512
    )

    print(f"Configuration:")
    print(f"  Channels: {stream.channels}")
    print(f"  Sample Rate: {stream.sample_rate} Hz")
    print(f"  Buffer Size: {stream.buffer_size} frames")
    print(f"  Latency: {stream.latency * 1000:.2f} ms")

    # Add callback to capture audio
    audio_levels = []

    def measure_level(audio_data, frame_count):
        """Measure audio level."""
        if isinstance(audio_data, np.ndarray):
            rms = np.sqrt(np.mean(audio_data**2))
            audio_levels.append(rms)
            if len(audio_levels) % 10 == 0:
                print(f"  Audio level: {rms:.4f}")

    stream.add_callback(measure_level)

    print("\nStarting capture (would capture for 3 seconds with real hardware)...")
    try:
        stream.start()
        # Note: With real hardware, this would capture audio
        # For now, just demonstrate the API
        print("  Stream is active:", stream.is_active)
        stream.stop()
        print("Stopped capture")
    except RuntimeError as e:
        if "Cython-level callback" in str(e):
            print("  [Not implemented - requires Cython callback support]")
        else:
            raise


def demo_output_stream():
    """Demo 2: Audio output stream with generated audio."""
    print("\n" + "=" * 70)
    print("DEMO 2: Audio Output Stream")
    print("=" * 70)

    if not NUMPY_AVAILABLE:
        print("Skipping: Requires NumPy")
        return

    print("\nCreating output stream...")
    stream = AudioOutputStream(
        channels=2, sample_rate=44100.0, buffer_size=512
    )

    # Generate a simple sine wave
    phase = [0.0]
    frequency = 440.0  # A4 note

    def generate_sine(frame_count):
        """Generate sine wave audio."""
        samples = []
        for _ in range(frame_count):
            sample = np.sin(2 * np.pi * phase[0])
            samples.append([sample, sample])  # Stereo
            phase[0] += frequency / stream.sample_rate
            if phase[0] >= 1.0:
                phase[0] -= 1.0

        return np.array(samples, dtype=np.float32)

    stream.set_generator(generate_sine)

    print(f"Configuration:")
    print(f"  Channels: {stream.channels}")
    print(f"  Sample Rate: {stream.sample_rate} Hz")
    print(f"  Buffer Size: {stream.buffer_size} frames")
    print(f"  Frequency: {frequency} Hz")
    print(f"  Latency: {stream.latency * 1000:.2f} ms")

    print("\nStarting playback (would play tone with real hardware)...")
    try:
        stream.start()
        print("  Stream is active:", stream.is_active)
        stream.stop()
        print("Stopped playback")
    except RuntimeError as e:
        if "Cython-level callback" in str(e):
            print("  [Not implemented - requires Cython callback support]")
        else:
            raise


def demo_loopback():
    """Demo 3: Simple audio loopback."""
    print("\n" + "=" * 70)
    print("DEMO 3: Audio Loopback")
    print("=" * 70)

    print("\nCreating loopback processor (input → output)...")
    loopback = create_loopback(
        channels=2,
        sample_rate=44100.0,
        buffer_size=256,  # Lower buffer = lower latency
    )

    print(f"Configuration:")
    print(f"  Channels: {loopback.channels}")
    print(f"  Sample Rate: {loopback.sample_rate} Hz")
    print(f"  Buffer Size: {loopback.buffer_size} frames")
    print(f"  Total Latency: {loopback.latency * 1000:.2f} ms")
    print(f"    (Input + Processing + Output)")

    print("\nStarting loopback (would pass audio through with real hardware)...")
    try:
        loopback.start()
        print("  Processor is active:", loopback.is_active)
        loopback.stop()
        print("Stopped loopback")
    except RuntimeError as e:
        if "Cython-level callback" in str(e):
            print("  [Not implemented - requires Cython callback support]")
        else:
            raise


def demo_audio_processor():
    """Demo 4: Real-time audio processor with effects."""
    print("\n" + "=" * 70)
    print("DEMO 4: Real-Time Audio Processor")
    print("=" * 70)

    if not NUMPY_AVAILABLE:
        print("Skipping: Requires NumPy")
        return

    # Create a simple distortion effect
    def guitar_distortion(audio_in):
        """Apply distortion effect."""
        if not isinstance(audio_in, np.ndarray):
            return audio_in

        # Drive the signal
        gain = 10.0
        driven = np.tanh(audio_in * gain)

        # Reduce output level
        return driven * 0.5

    print("\nCreating guitar distortion processor...")
    processor = AudioProcessor(
        guitar_distortion,
        channels=2,
        sample_rate=44100.0,
        buffer_size=128,  # Very low latency for real-time feel
    )

    print(f"Configuration:")
    print(f"  Effect: Distortion (tanh)")
    print(f"  Channels: {processor.channels}")
    print(f"  Sample Rate: {processor.sample_rate} Hz")
    print(f"  Buffer Size: {processor.buffer_size} frames")
    print(f"  Latency: {processor.latency * 1000:.2f} ms")

    print("\nStarting processor...")
    try:
        processor.start()
        print("  Processor is active:", processor.is_active)
        processor.stop()
        print("Stopped processor")
    except RuntimeError as e:
        if "Cython-level callback" in str(e):
            print("  [Not implemented - requires Cython callback support]")

            # Still demonstrate the processing logic
            print("\nDemonstrating processing logic without AudioUnit...")
            test_audio = np.random.randn(128, 2).astype(np.float32) * 0.1
            processor._input_buffer = test_audio

            # Process through effect
            output = processor._generate_output(128)
            print(f"  Input RMS: {np.sqrt(np.mean(test_audio**2)):.4f}")
            print(f"  Output RMS: {np.sqrt(np.mean(output**2)):.4f}")
        else:
            raise


def demo_stream_graph():
    """Demo 5: Stream graph with multiple effects."""
    print("\n" + "=" * 70)
    print("DEMO 5: Stream Graph with Multiple Effects")
    print("=" * 70)

    if not NUMPY_AVAILABLE:
        print("Skipping: Requires NumPy")
        return

    print("\nBuilding effects chain...")
    graph = StreamGraph(sample_rate=44100.0, buffer_size=256)

    # Define effects
    def high_pass_filter(audio):
        """Simple high-pass filter (demonstration)."""
        if not isinstance(audio, np.ndarray):
            return audio
        # Simplified HPF: subtract low frequencies
        return audio * 0.9

    def compressor(audio):
        """Simple compressor (demonstration)."""
        if not isinstance(audio, np.ndarray):
            return audio
        # Soft clipping compression
        return np.tanh(audio * 1.5) / 1.5

    def reverb(audio):
        """Simple reverb effect (demonstration)."""
        if not isinstance(audio, np.ndarray):
            return audio
        # Very simple reverb: mix with delayed signal
        return audio * 0.8

    # Add nodes to graph
    print("  Adding nodes:")
    print("    - Input")
    print("    - High-pass filter")
    print("    - Compressor")
    print("    - Reverb")
    print("    - Output")

    graph.add_node("input", lambda x: x)
    graph.add_node("hpf", high_pass_filter)
    graph.add_node("compressor", compressor)
    graph.add_node("reverb", reverb)
    graph.add_node("output", lambda x: x)

    # Connect nodes
    print("\n  Connecting: input → HPF → compressor → reverb → output")
    graph.connect("input", "hpf")
    graph.connect("hpf", "compressor")
    graph.connect("compressor", "reverb")
    graph.connect("reverb", "output")

    print(f"\nGraph configuration:")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Connections: {len(graph.connections)}")
    print(f"  Sample Rate: {graph.sample_rate} Hz")
    print(f"  Latency: {graph.latency * 1000:.2f} ms")

    # Test the graph processing
    print("\nTesting graph processing...")
    combined = graph._create_combined_processor()

    test_audio = np.random.randn(256, 2).astype(np.float32) * 0.3
    output = combined(test_audio)

    print(f"  Input shape: {test_audio.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Input RMS: {np.sqrt(np.mean(test_audio**2)):.4f}")
    print(f"  Output RMS: {np.sqrt(np.mean(output**2)):.4f}")

    print("\nStarting graph...")
    try:
        graph.start()
        print("  Graph is active:", graph.is_active)
        graph.stop()
        print("Stopped graph")
    except RuntimeError as e:
        if "Cython-level callback" in str(e):
            print("  [Not implemented - requires Cython callback support]")
        else:
            raise


def demo_branching_graph():
    """Demo 6: Stream graph with branching/merging."""
    print("\n" + "=" * 70)
    print("DEMO 6: Branching Stream Graph")
    print("=" * 70)

    print("\nBuilding branching graph...")
    graph = StreamGraph()

    # Create parallel processing paths
    graph.add_node("input", lambda x: x)
    graph.add_node("wet", lambda x: x * 0.5)  # Effect path
    graph.add_node("dry", lambda x: x * 1.0)  # Dry path
    graph.add_node("mix", lambda x: x)  # Mix both paths

    # Connect: input → wet → mix
    #          input → dry → mix
    print("  Structure:")
    print("           ┌─→ wet ─→┐")
    print("    input ─┤          ├─→ mix")
    print("           └─→ dry ─→┘")

    graph.connect("input", "wet")
    graph.connect("input", "dry")
    graph.connect("wet", "mix")
    graph.connect("dry", "mix")

    # Get processing order
    processing_order = graph._topological_sort()
    print(f"\n  Processing order: {' → '.join(processing_order)}")

    # Test the graph
    test_value = 10
    combined = graph._create_combined_processor()
    result = combined(test_value)

    print(f"\n  Input: {test_value}")
    print(f"  Wet path: {test_value * 0.5}")
    print(f"  Dry path: {test_value * 1.0}")
    print(f"  Mixed result: {result}")


def demo_context_managers():
    """Demo 7: Using context managers."""
    print("\n" + "=" * 70)
    print("DEMO 7: Context Managers")
    print("=" * 70)

    print("\nDemonstrating context manager usage...")

    # Input stream with context manager
    print("\n1. AudioInputStream:")
    try:
        with AudioInputStream(buffer_size=512) as stream:
            print(f"   Stream is active inside context: {stream.is_active}")
        print(f"   Stream is active after context: {stream.is_active}")
    except RuntimeError as e:
        if "Cython-level callback" in str(e):
            print("   [Not implemented - requires Cython callback support]")
        else:
            raise

    # Processor with context manager
    print("\n2. AudioProcessor:")
    try:
        with create_loopback(buffer_size=256) as processor:
            print(f"   Processor is active inside context: {processor.is_active}")
            print(f"   Latency: {processor.latency * 1000:.2f} ms")
        print(f"   Processor is active after context: {processor.is_active}")
    except RuntimeError as e:
        if "Cython-level callback" in str(e):
            print("   [Not implemented - requires Cython callback support]")
        else:
            raise

    # Stream graph with context manager
    print("\n3. StreamGraph:")
    graph = StreamGraph()
    graph.add_node("passthrough", lambda x: x)

    try:
        with graph as g:
            print(f"   Graph is active inside context: {g.is_active}")
        print(f"   Graph is active after context: {graph.is_active}")
    except RuntimeError as e:
        if "Cython-level callback" in str(e):
            print("   [Not implemented - requires Cython callback support]")
        else:
            raise


def demo_latency_comparison():
    """Demo 8: Latency comparison with different buffer sizes."""
    print("\n" + "=" * 70)
    print("DEMO 8: Latency vs Buffer Size")
    print("=" * 70)

    print("\nComparing latency for different buffer sizes...")
    print(f"\n{'Buffer Size':<15} {'Latency (ms)':<15} {'Use Case':<30}")
    print("-" * 60)

    configs = [
        (64, "Ultra-low latency (guitar/vocals)"),
        (128, "Very low latency (live monitoring)"),
        (256, "Low latency (real-time effects)"),
        (512, "Balanced (general use)"),
        (1024, "Higher latency (less CPU usage)"),
        (2048, "High latency (background processing)"),
    ]

    for buffer_size, use_case in configs:
        processor = create_loopback(buffer_size=buffer_size)
        latency_ms = processor.latency * 1000
        print(f"{buffer_size:<15} {latency_ms:<15.2f} {use_case:<30}")

    print("\nNote: Actual latency includes hardware buffers and driver latency.")
    print("      These are theoretical values for 44.1 kHz sampling.")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("COREMUSIC REAL-TIME STREAMING DEMO")
    print("=" * 70)
    print("\nThis demo showcases real-time audio streaming capabilities:")
    print("- Audio input/output streams")
    print("- Real-time audio processing")
    print("- Stream graphs with effects chains")
    print("- Latency management")
    print("\nNOTE: Actual audio I/O requires hardware and AudioUnit implementation.")
    print("      These demos demonstrate the API and processing logic.")

    try:
        demo_input_stream()
        demo_output_stream()
        demo_loopback()
        demo_audio_processor()
        demo_stream_graph()
        demo_branching_graph()
        demo_context_managers()
        demo_latency_comparison()

        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("- AudioInputStream captures audio with callbacks")
        print("- AudioOutputStream generates audio with a generator function")
        print("- AudioProcessor combines input/output for effects")
        print("- StreamGraph allows complex routing and effects chains")
        print("- Lower buffer sizes = lower latency but more CPU usage")
        print("- All classes support context managers for automatic cleanup")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
