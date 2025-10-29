#!/usr/bin/env python3
"""Tests for real-time audio streaming module."""

import time
import pytest
import coremusic as cm
from coremusic.audio.streaming import (
    AudioInputStream,
    AudioOutputStream,
    AudioProcessor,
    StreamGraph,
    StreamNode,
    create_loopback,
)


# ============================================================================
# Test AudioInputStream
# ============================================================================


class TestAudioInputStream:
    """Test AudioInputStream class."""

    def test_create_input_stream(self):
        """Test creating an input stream."""
        stream = AudioInputStream(channels=2, sample_rate=44100.0, buffer_size=512)

        assert stream.channels == 2
        assert stream.sample_rate == 44100.0
        assert stream.buffer_size == 512
        assert not stream.is_active
        assert stream.latency == pytest.approx(512 / 44100.0)

    def test_create_with_custom_parameters(self):
        """Test creating stream with custom parameters."""
        stream = AudioInputStream(channels=1, sample_rate=48000.0, buffer_size=256)

        assert stream.channels == 1
        assert stream.sample_rate == 48000.0
        assert stream.buffer_size == 256
        assert stream.latency == pytest.approx(256 / 48000.0)

    def test_add_callback(self):
        """Test adding callbacks."""
        stream = AudioInputStream()
        callback_count = 0

        def my_callback(data, frame_count):
            nonlocal callback_count
            callback_count += 1

        stream.add_callback(my_callback)
        assert len(stream._callbacks) == 1

        # Add another callback
        def another_callback(data, frame_count):
            pass

        stream.add_callback(another_callback)
        assert len(stream._callbacks) == 2

    def test_remove_callback(self):
        """Test removing callbacks."""
        stream = AudioInputStream()

        def my_callback(data, frame_count):
            pass

        stream.add_callback(my_callback)
        assert len(stream._callbacks) == 1

        stream.remove_callback(my_callback)
        assert len(stream._callbacks) == 0

    def test_start_not_implemented(self):
        """Test that start raises error (requires Cython callbacks)."""
        stream = AudioInputStream()

        assert not stream.is_active

        # Start requires Cython-level callback implementation
        with pytest.raises(RuntimeError, match="Cython-level callback"):
            stream.start()

    def test_start_already_active_raises_error(self):
        """Test that starting already active stream raises error."""
        stream = AudioInputStream()

        # Manually set active to test the check
        stream._is_active = True

        with pytest.raises(RuntimeError, match="already active"):
            stream.start()

        stream._is_active = False

    def test_stop_safe_when_not_active(self):
        """Test that stop is safe when not active."""
        stream = AudioInputStream()

        assert not stream.is_active

        # Should not raise
        stream.stop()

        assert not stream.is_active


# ============================================================================
# Test AudioOutputStream
# ============================================================================


class TestAudioOutputStream:
    """Test AudioOutputStream class."""

    def test_create_output_stream(self):
        """Test creating an output stream."""
        stream = AudioOutputStream(channels=2, sample_rate=44100.0, buffer_size=512)

        assert stream.channels == 2
        assert stream.sample_rate == 44100.0
        assert stream.buffer_size == 512
        assert not stream.is_active
        assert stream.latency == pytest.approx(512 / 44100.0)

    def test_set_generator(self):
        """Test setting generator function."""
        stream = AudioOutputStream()

        def my_generator(frame_count):
            try:
                import numpy as np

                return np.zeros((frame_count, 2), dtype=np.float32)
            except ImportError:
                return b"\x00" * (frame_count * 2 * 4)

        stream.set_generator(my_generator)
        assert stream._generator is not None

    def test_start_without_generator_raises_error(self):
        """Test that starting without generator raises error."""
        stream = AudioOutputStream()

        with pytest.raises(RuntimeError, match="No generator"):
            stream.start()

    def test_start_not_implemented_with_generator(self):
        """Test that start raises RuntimeError even with generator."""
        stream = AudioOutputStream()

        def generator(frame_count):
            try:
                import numpy as np

                return np.zeros((frame_count, 2), dtype=np.float32)
            except ImportError:
                return b"\x00" * (frame_count * 2 * 4)

        stream.set_generator(generator)

        # Start requires Cython-level callback implementation
        with pytest.raises(RuntimeError, match="Cython-level callback"):
            stream.start()

    def test_stop_safe_when_not_active(self):
        """Test that stop is safe when not active."""
        stream = AudioOutputStream()

        # Should not raise
        stream.stop()

        assert not stream.is_active


# ============================================================================
# Test AudioProcessor
# ============================================================================


class TestAudioProcessor:
    """Test AudioProcessor class."""

    def test_create_processor(self):
        """Test creating an audio processor."""

        def passthrough(audio_in):
            return audio_in

        processor = AudioProcessor(passthrough, channels=2, sample_rate=44100.0, buffer_size=512)

        assert processor.channels == 2
        assert processor.sample_rate == 44100.0
        assert processor.buffer_size == 512
        assert not processor.is_active

    def test_processor_latency(self):
        """Test latency calculation."""

        def passthrough(audio_in):
            return audio_in

        processor = AudioProcessor(passthrough, buffer_size=512, sample_rate=44100.0)

        # Latency should be 3 buffers (input + processing + output)
        expected_latency = (512 * 3) / 44100.0
        assert processor.latency == pytest.approx(expected_latency)

    def test_processor_callbacks_connected(self):
        """Test that processor connects callbacks correctly."""

        def process_func(audio_in):
            return audio_in

        processor = AudioProcessor(process_func)

        # Input stream should have callback
        assert len(processor.input_stream._callbacks) == 1

        # Output stream should have generator
        assert processor.output_stream._generator is not None

    def test_processor_on_input(self):
        """Test input callback stores data."""

        def process_func(audio_in):
            return audio_in

        processor = AudioProcessor(process_func)

        # Simulate input data
        test_data = b"test audio data"
        processor._on_input(test_data, 512)

        assert processor._input_buffer == test_data

    def test_processor_generate_output_with_input(self):
        """Test output generation with available input."""

        def process_func(audio_in):
            # Simple processing - just return input
            return audio_in

        processor = AudioProcessor(process_func)

        # Provide input data
        test_data = b"test audio data"
        processor._input_buffer = test_data

        # Generate output
        output = processor._generate_output(512)

        assert output == test_data

    def test_processor_generate_output_without_input(self):
        """Test output generation without input (should return silence)."""

        def process_func(audio_in):
            return audio_in

        processor = AudioProcessor(process_func)

        # No input data yet
        output = processor._generate_output(512)

        # Should return silence (zeros or zero bytes)
        assert output is not None

    def test_processor_start_not_implemented(self):
        """Test processor start raises RuntimeError."""

        def process_func(audio_in):
            return audio_in

        processor = AudioProcessor(process_func)

        # Start requires Cython-level callback implementation
        with pytest.raises(RuntimeError, match="Cython-level callback"):
            processor.start()

    def test_processor_stop_safe(self):
        """Test processor stop is safe."""

        def process_func(audio_in):
            return audio_in

        processor = AudioProcessor(process_func)

        # Should not raise
        processor.stop()

        assert not processor.is_active

    def test_processor_error_handling(self):
        """Test that processor handles errors gracefully."""

        def failing_process(audio_in):
            raise ValueError("Processing error")

        processor = AudioProcessor(failing_process)
        processor._input_buffer = b"test"

        # Should return silence on error, not raise
        output = processor._generate_output(512)
        assert output is not None


# ============================================================================
# Test StreamGraph
# ============================================================================


class TestStreamGraph:
    """Test StreamGraph class."""

    def test_create_graph(self):
        """Test creating a stream graph."""
        graph = StreamGraph(sample_rate=44100.0, buffer_size=512)

        assert graph.sample_rate == 44100.0
        assert graph.buffer_size == 512
        assert len(graph.nodes) == 0
        assert len(graph.connections) == 0
        assert not graph.is_active

    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = StreamGraph()

        def process(x):
            return x

        node = graph.add_node("test_node", process)

        assert isinstance(node, StreamNode)
        assert node.name == "test_node"
        assert "test_node" in graph.nodes
        assert graph.nodes["test_node"] is node

    def test_add_duplicate_node_raises_error(self):
        """Test that adding duplicate node raises error."""
        graph = StreamGraph()

        graph.add_node("test", lambda x: x)

        with pytest.raises(ValueError, match="already exists"):
            graph.add_node("test", lambda x: x)

    def test_connect_nodes(self):
        """Test connecting nodes."""
        graph = StreamGraph()

        graph.add_node("node1", lambda x: x)
        graph.add_node("node2", lambda x: x)

        graph.connect("node1", "node2")

        assert ("node1", "node2") in graph.connections
        assert "node2" in graph.nodes["node1"].outputs
        assert "node1" in graph.nodes["node2"].inputs

    def test_connect_nonexistent_source_raises_error(self):
        """Test that connecting nonexistent source raises error."""
        graph = StreamGraph()
        graph.add_node("node1", lambda x: x)

        with pytest.raises(ValueError, match="not found"):
            graph.connect("nonexistent", "node1")

    def test_connect_nonexistent_destination_raises_error(self):
        """Test that connecting to nonexistent destination raises error."""
        graph = StreamGraph()
        graph.add_node("node1", lambda x: x)

        with pytest.raises(ValueError, match="not found"):
            graph.connect("node1", "nonexistent")

    def test_topological_sort_linear(self):
        """Test topological sort with linear graph."""
        graph = StreamGraph()

        graph.add_node("node1", lambda x: x)
        graph.add_node("node2", lambda x: x)
        graph.add_node("node3", lambda x: x)

        graph.connect("node1", "node2")
        graph.connect("node2", "node3")

        sorted_nodes = graph._topological_sort()

        assert sorted_nodes == ["node1", "node2", "node3"]

    def test_topological_sort_branching(self):
        """Test topological sort with branching graph."""
        graph = StreamGraph()

        graph.add_node("input", lambda x: x)
        graph.add_node("branch1", lambda x: x)
        graph.add_node("branch2", lambda x: x)
        graph.add_node("output", lambda x: x)

        graph.connect("input", "branch1")
        graph.connect("input", "branch2")
        graph.connect("branch1", "output")
        graph.connect("branch2", "output")

        sorted_nodes = graph._topological_sort()

        # Input should be first
        assert sorted_nodes[0] == "input"
        # Output should be last
        assert sorted_nodes[-1] == "output"
        # Branches should be in middle (order doesn't matter)
        assert set(sorted_nodes[1:3]) == {"branch1", "branch2"}

    def test_topological_sort_cycle_raises_error(self):
        """Test that cyclic graph raises error."""
        graph = StreamGraph()

        graph.add_node("node1", lambda x: x)
        graph.add_node("node2", lambda x: x)

        # Create cycle
        graph.connect("node1", "node2")
        graph.connect("node2", "node1")

        with pytest.raises(ValueError, match="cycle"):
            graph._topological_sort()

    def test_combined_processor_linear_chain(self):
        """Test combined processor with linear chain."""
        graph = StreamGraph()

        # Add nodes that multiply by 2, 3, 4
        graph.add_node("x2", lambda x: x * 2)
        graph.add_node("x3", lambda x: x * 3)
        graph.add_node("x4", lambda x: x * 4)

        # Connect: x2 → x3 → x4
        graph.connect("x2", "x3")
        graph.connect("x3", "x4")

        # Create combined processor
        combined = graph._create_combined_processor()

        # Test: 1 * 2 * 3 * 4 = 24
        result = combined(1)
        assert result == 24

    def test_start_empty_graph_raises_error(self):
        """Test that starting empty graph raises error."""
        graph = StreamGraph()

        with pytest.raises(ValueError, match="empty graph"):
            graph.start()

    def test_start_not_implemented(self):
        """Test that starting graph raises RuntimeError."""
        graph = StreamGraph()
        graph.add_node("test", lambda x: x)

        # Start requires Cython-level callback implementation
        with pytest.raises(RuntimeError, match="Cython-level callback"):
            graph.start()

    def test_start_already_active_raises_error(self):
        """Test that starting already active graph raises error."""
        graph = StreamGraph()
        graph.add_node("test", lambda x: x)

        # Manually set active to test the check
        graph._is_active = True

        with pytest.raises(RuntimeError, match="already active"):
            graph.start()

        graph._is_active = False


# ============================================================================
# Test StreamNode
# ============================================================================


class TestStreamNode:
    """Test StreamNode class."""

    def test_create_node(self):
        """Test creating a stream node."""

        def process(x):
            return x * 2

        node = StreamNode("test", process)

        assert node.name == "test"
        assert node.processor is process
        assert len(node.inputs) == 0
        assert len(node.outputs) == 0

    def test_node_process(self):
        """Test node processing."""

        def multiply_by_two(x):
            return x * 2

        node = StreamNode("test", multiply_by_two)

        result = node.process(5)
        assert result == 10

    def test_node_error_handling(self):
        """Test node handles errors gracefully."""

        def failing_process(x):
            raise ValueError("Error")

        node = StreamNode("test", failing_process)

        # Should return input unchanged on error
        result = node.process(42)
        assert result == 42


# ============================================================================
# Test Convenience Functions
# ============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_loopback(self):
        """Test creating loopback processor."""
        loopback = create_loopback(channels=2, sample_rate=48000.0, buffer_size=256)

        assert isinstance(loopback, AudioProcessor)
        assert loopback.channels == 2
        assert loopback.sample_rate == 48000.0
        assert loopback.buffer_size == 256

    def test_loopback_passes_through_data(self):
        """Test that loopback passes data through unchanged."""
        loopback = create_loopback()

        # Set input data
        test_data = b"test audio"
        loopback._input_buffer = test_data

        # Generate output
        output = loopback._generate_output(512)

        assert output == test_data


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for streaming."""

    def test_simple_processing_pipeline(self):
        """Test simple processing pipeline."""

        def gain_effect(audio_in):
            """Apply 0.5 gain."""
            try:
                import numpy as np

                return audio_in * 0.5
            except ImportError:
                return audio_in

        processor = AudioProcessor(gain_effect, buffer_size=256)

        # Processor should be properly configured
        assert processor.input_stream is not None
        assert processor.output_stream is not None
        assert processor.process_func is gain_effect

    def test_multi_node_graph(self):
        """Test multi-node processing graph."""
        graph = StreamGraph()

        # Build graph: input → gain → filter → output
        graph.add_node("gain", lambda x: x * 0.5)
        graph.add_node("filter", lambda x: x * 0.8)

        graph.connect("gain", "filter")

        # Get combined processor
        combined = graph._create_combined_processor()

        # Test: 10 * 0.5 * 0.8 = 4.0
        result = combined(10)
        assert result == pytest.approx(4.0)

    @pytest.mark.skipif(not cm.NUMPY_AVAILABLE, reason="NumPy not available")
    def test_numpy_processing(self):
        """Test processing with NumPy arrays."""
        import numpy as np

        def process_numpy(audio_in):
            """Apply simple gain."""
            return audio_in * 0.5

        processor = AudioProcessor(process_numpy)

        # Create test audio data
        test_audio = np.random.randn(512, 2).astype(np.float32)
        processor._input_buffer = test_audio

        # Process
        output = processor._generate_output(512)

        # Verify output
        assert isinstance(output, np.ndarray)
        assert np.allclose(output, test_audio * 0.5)
