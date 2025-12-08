#!/usr/bin/env python3
"""Real-time audio streaming module.

This module provides high-level abstractions for real-time audio I/O with minimal latency:
- Audio input/output streams
- Real-time processing callbacks
- Stream graph connections
- Latency management

Classes:
    AudioInputStream: Real-time audio input stream from device
    AudioOutputStream: Real-time audio output stream to device
    AudioProcessor: Combined input → process → output pipeline
    StreamGraph: Audio processing graph with node connections
    StreamNode: Individual processing node in the graph

Example:
    # Simple audio loopback
    def process(audio_in):
        return audio_in  # Pass through

    processor = AudioProcessor(process, buffer_size=256)
    processor.start()
    time.sleep(5)
    processor.stop()
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from .. import capi
from ..objects import AudioDevice, AudioDeviceManager

if TYPE_CHECKING:

    try:
        from numpy.typing import NDArray as NDArray_
    except ImportError:
        NDArray_ = Any  # type: ignore[misc,assignment]

    NDArray = NDArray_


logger = logging.getLogger(__name__)


# ============================================================================
# Core Stream Classes
# ============================================================================


class AudioInputStream:
    """Real-time audio input stream from device.

    Captures audio from an input device with minimal latency and calls
    registered callbacks with each buffer of audio data.

    Attributes:
        device: Input device (None = default)
        channels: Number of input channels
        sample_rate: Sample rate in Hz
        buffer_size: Buffer size in frames (smaller = lower latency)
        is_active: Whether the stream is currently active
    """

    def __init__(
        self,
        device: Optional[AudioDevice] = None,
        channels: int = 2,
        sample_rate: float = 44100.0,
        buffer_size: int = 512,
    ):
        """Initialize input stream.

        Args:
            device: Input device (None = default)
            channels: Number of input channels
            sample_rate: Sample rate in Hz
            buffer_size: Buffer size in frames (smaller = lower latency)
        """
        self.device = device or AudioDeviceManager.get_default_input_device()
        self.channels = channels
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self._callbacks: List[Callable[[Any, int], None]] = []
        self._is_active = False
        self._audio_unit_id: Optional[int] = None
        self._lock = threading.Lock()

        # Validate NumPy availability if needed
        self._check_numpy()

    def _check_numpy(self) -> None:
        """Check if NumPy is available for array operations."""
        try:
            import numpy as np

            self._np = np
            self._has_numpy = True
        except ImportError:
            self._np = None  # type: ignore[assignment]
            self._has_numpy = False
            logger.warning(
                "NumPy not available - audio data will be provided as bytes"
            )

    def add_callback(self, callback: Callable[[Any, int], None]) -> None:
        """Add callback for audio data.

        Args:
            callback: Function(audio_data, frame_count) -> None
                - audio_data: NumPy array (frames, channels) if NumPy available, else bytes
                - frame_count: Number of frames in buffer
                Called in real-time thread with each buffer
        """
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[Any, int], None]) -> None:
        """Remove a previously registered callback.

        Args:
            callback: Callback function to remove
        """
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def start(self) -> None:
        """Start capturing audio.

        Raises:
            RuntimeError: If stream is already active or setup fails

        Note:
            Full implementation requires Cython-level render callback support.
            This method provides the structure and will work once the render
            callback infrastructure is added to capi.pyx.
        """
        if self._is_active:
            raise RuntimeError("Stream is already active")

        try:
            self._setup_audio_unit()
            self._is_active = True
            logger.info(
                f"Started audio input stream: {self.channels} channels, "
                f"{self.sample_rate} Hz, {self.buffer_size} frames"
            )
        except Exception as e:
            logger.error(f"Failed to start input stream: {e}")
            raise RuntimeError(f"Failed to start input stream: {e}")

    def _setup_audio_unit(self) -> None:
        """Set up AudioUnit for input capture.

        This method demonstrates the required setup structure. Full functionality
        requires adding input capture callback support to capi.pyx:

        1. Create callback function in Cython:
           cdef OSStatus input_capture_callback(
               void* user_data,
               AudioUnitRenderActionFlags* flags,
               const AudioTimeStamp* timestamp,
               UInt32 bus_number,
               UInt32 num_frames,
               AudioBufferList* io_data
           ):
               # Pull audio from input
               # Call Python callbacks with audio data
               return noErr

        2. Set up AudioUnit for input (HAL I/O unit):
           - component_desc.type = 'auou' (kAudioUnitType_Output)
           - component_desc.subtype = 'ahal' (kAudioUnitSubType_HALOutput)
           - Enable input on element 1
           - Set format on input scope, element 1
           - Install input callback via kAudioOutputUnitProperty_SetInputCallback

        For now, this raises NotImplementedError to indicate Cython work is needed.
        """
        raise NotImplementedError(
            "Audio input capture requires Cython-level callback implementation. "
            "To implement:\n"
            "1. Add input_capture_callback() to capi.pyx\n"
            "2. Add audio_input_stream_create() wrapper function\n"
            "3. Expose via Python API\n"
            "See audio_player_render_callback() in capi.pyx as reference."
        )

    def stop(self) -> None:
        """Stop capturing audio."""
        if not self._is_active:
            return

        try:
            self._teardown_audio_unit()
            self._is_active = False
            logger.info("Stopped audio input stream")
        except Exception as e:
            logger.warning(f"Error stopping input stream: {e}")
            self._is_active = False

    def _teardown_audio_unit(self) -> None:
        """Tear down AudioUnit.

        Stops and disposes the AudioUnit if it was created.
        """
        if self._audio_unit_id is not None:
            try:
                # Stop the audio unit
                capi.audio_output_unit_stop(self._audio_unit_id)
                # Uninitialize
                capi.audio_unit_uninitialize(self._audio_unit_id)
                # Dispose
                capi.audio_component_instance_dispose(self._audio_unit_id)
            except Exception as e:
                logger.warning(f"Error during AudioUnit teardown: {e}")
            finally:
                self._audio_unit_id = None

    @property
    def is_active(self) -> bool:
        """Check if stream is currently active."""
        return self._is_active

    @property
    def latency(self) -> float:
        """Get input latency in seconds."""
        return self.buffer_size / self.sample_rate

    def __enter__(self) -> "AudioInputStream":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()


class AudioOutputStream:
    """Real-time audio output stream to device.

    Plays audio to an output device with minimal latency by calling
    a generator function to produce each buffer of audio data.

    Attributes:
        device: Output device (None = default)
        channels: Number of output channels
        sample_rate: Sample rate in Hz
        buffer_size: Buffer size in frames (smaller = lower latency)
        is_active: Whether the stream is currently active
    """

    def __init__(
        self,
        device: Optional[AudioDevice] = None,
        channels: int = 2,
        sample_rate: float = 44100.0,
        buffer_size: int = 512,
    ):
        """Initialize output stream.

        Args:
            device: Output device (None = default)
            channels: Number of output channels
            sample_rate: Sample rate in Hz
            buffer_size: Buffer size in frames (smaller = lower latency)
        """
        self.device = device or AudioDeviceManager.get_default_output_device()
        self.channels = channels
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self._generator: Optional[Callable[[int], Any]] = None
        self._is_active = False
        self._audio_unit_id: Optional[int] = None
        self._lock = threading.Lock()

        # Validate NumPy availability if needed
        self._check_numpy()

    def _check_numpy(self) -> None:
        """Check if NumPy is available for array operations."""
        try:
            import numpy as np

            self._np = np
            self._has_numpy = True
        except ImportError:
            self._np = None  # type: ignore[assignment]
            self._has_numpy = False
            logger.warning(
                "NumPy not available - generator must return bytes"
            )

    def set_generator(self, generator: Callable[[int], Any]) -> None:
        """Set audio generator function.

        Args:
            generator: Function(frame_count: int) -> audio_data
                - frame_count: Number of frames requested
                - Returns: NumPy array (frames, channels) if NumPy available, else bytes
                Called in real-time thread to generate each buffer
        """
        with self._lock:
            self._generator = generator

    def start(self) -> None:
        """Start audio playback.

        Raises:
            RuntimeError: If stream is already active, no generator set, or setup fails

        Note:
            Full implementation requires Cython-level render callback support.
            This method provides the structure and will work once the render
            callback infrastructure is extended in capi.pyx.
        """
        if self._is_active:
            raise RuntimeError("Stream is already active")

        if self._generator is None:
            raise RuntimeError("No generator function set")

        try:
            self._setup_audio_unit()
            self._is_active = True
            logger.info(
                f"Started audio output stream: {self.channels} channels, "
                f"{self.sample_rate} Hz, {self.buffer_size} frames"
            )
        except Exception as e:
            logger.error(f"Failed to start output stream: {e}")
            raise RuntimeError(f"Failed to start output stream: {e}")

    def _setup_audio_unit(self) -> None:
        """Set up AudioUnit for output playback.

        This method demonstrates the required setup structure. Full functionality
        requires extending the render callback support in capi.pyx:

        1. Extend existing audio_player_render_callback() or create new callback:
           cdef OSStatus output_stream_callback(
               void* user_data,
               AudioUnitRenderActionFlags* flags,
               const AudioTimeStamp* timestamp,
               UInt32 bus_number,
               UInt32 num_frames,
               AudioBufferList* io_data
           ):
               # Call Python generator function
               # Fill io_data buffers with generated audio
               return noErr

        2. Set up AudioUnit for output (Default Output unit):
           - component_desc.type = 'auou' (kAudioUnitType_Output)
           - component_desc.subtype = 'def ' (kAudioUnitSubType_DefaultOutput)
           - Set format on output scope, element 0
           - Install render callback via kAudioUnitProperty_SetRenderCallback

        For now, this raises NotImplementedError to indicate Cython work is needed.
        """
        raise NotImplementedError(
            "Audio output playback requires Cython-level callback implementation. "
            "To implement:\n"
            "1. Extend or create output_stream_callback() in capi.pyx\n"
            "2. Add audio_output_stream_create() wrapper function\n"
            "3. Expose via Python API\n"
            "See audio_player_render_callback() in capi.pyx as reference.\n"
            "The existing audio player infrastructure provides a working example."
        )

    def stop(self) -> None:
        """Stop audio playback."""
        if not self._is_active:
            return

        try:
            self._teardown_audio_unit()
            self._is_active = False
            logger.info("Stopped audio output stream")
        except Exception as e:
            logger.warning(f"Error stopping output stream: {e}")
            self._is_active = False

    def _teardown_audio_unit(self) -> None:
        """Tear down AudioUnit.

        Stops and disposes the AudioUnit if it was created.
        """
        if self._audio_unit_id is not None:
            try:
                # Stop the audio unit
                capi.audio_output_unit_stop(self._audio_unit_id)
                # Uninitialize
                capi.audio_unit_uninitialize(self._audio_unit_id)
                # Dispose
                capi.audio_component_instance_dispose(self._audio_unit_id)
            except Exception as e:
                logger.warning(f"Error during AudioUnit teardown: {e}")
            finally:
                self._audio_unit_id = None

    @property
    def is_active(self) -> bool:
        """Check if stream is currently active."""
        return self._is_active

    @property
    def latency(self) -> float:
        """Get output latency in seconds."""
        return self.buffer_size / self.sample_rate

    def __enter__(self) -> "AudioOutputStream":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()


class AudioProcessor:
    """Real-time audio processor (input → process → output).

    Combines an input stream and output stream with a processing function
    to create a complete real-time audio processing pipeline.

    Attributes:
        process_func: Audio processing function
        input_stream: Audio input stream
        output_stream: Audio output stream
        channels: Number of channels
        sample_rate: Sample rate in Hz
        buffer_size: Buffer size in frames
    """

    def __init__(
        self,
        process_func: Callable[[Any], Any],
        channels: int = 2,
        sample_rate: float = 44100.0,
        buffer_size: int = 512,
        input_device: Optional[AudioDevice] = None,
        output_device: Optional[AudioDevice] = None,
    ):
        """Initialize real-time processor.

        Args:
            process_func: Function(input_audio) -> output_audio
                - input_audio: NumPy array (frames, channels) or bytes
                - output_audio: NumPy array (frames, channels) or bytes
                Audio processing function (must be real-time safe!)
            channels: Number of channels
            sample_rate: Sample rate
            buffer_size: Buffer size (smaller = lower latency, more CPU)
            input_device: Input device (None = default)
            output_device: Output device (None = default)
        """
        self.process_func = process_func
        self.channels = channels
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        # Create input and output streams
        self.input_stream = AudioInputStream(
            device=input_device,
            channels=channels,
            sample_rate=sample_rate,
            buffer_size=buffer_size,
        )
        self.output_stream = AudioOutputStream(
            device=output_device,
            channels=channels,
            sample_rate=sample_rate,
            buffer_size=buffer_size,
        )

        # Connect input → process → output
        self._input_buffer: Optional[Any] = None
        self._buffer_lock = threading.Lock()
        self.input_stream.add_callback(self._on_input)
        self.output_stream.set_generator(self._generate_output)

    def _on_input(self, data: Any, frame_count: int) -> None:
        """Store input data for processing (called in real-time thread)."""
        with self._buffer_lock:
            self._input_buffer = data

    def _generate_output(self, frame_count: int) -> Any:
        """Generate output by processing input (called in real-time thread)."""
        with self._buffer_lock:
            if self._input_buffer is None:
                # No input yet - return silence
                try:
                    import numpy as np

                    return np.zeros((frame_count, self.channels), dtype=np.float32)
                except ImportError:
                    # Return silence as bytes
                    bytes_per_sample = 4  # float32
                    return b"\x00" * (frame_count * self.channels * bytes_per_sample)

            # Process the input buffer
            try:
                return self.process_func(self._input_buffer)
            except Exception as e:
                logger.error(f"Error in process function: {e}")
                # Return silence on error
                try:
                    import numpy as np

                    return np.zeros((frame_count, self.channels), dtype=np.float32)
                except ImportError:
                    bytes_per_sample = 4
                    return b"\x00" * (frame_count * self.channels * bytes_per_sample)

    def start(self) -> None:
        """Start real-time processing.

        Starts both input and output streams.
        """
        self.input_stream.start()
        self.output_stream.start()

    def stop(self) -> None:
        """Stop real-time processing.

        Stops both input and output streams.
        """
        self.output_stream.stop()
        self.input_stream.stop()

    @property
    def is_active(self) -> bool:
        """Check if processor is currently active."""
        return self.input_stream.is_active and self.output_stream.is_active

    @property
    def latency(self) -> float:
        """Get total system latency in seconds.

        This includes input latency + processing time + output latency.
        Processing time is approximated as one buffer period.
        """
        # Input buffer + processing buffer + output buffer
        return (self.buffer_size * 3) / self.sample_rate

    def __enter__(self) -> "AudioProcessor":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()


# ============================================================================
# Stream Graph Classes
# ============================================================================


@dataclass
class StreamNode:
    """Individual processing node in a stream graph.

    Attributes:
        name: Unique name for this node
        processor: Processing function for this node
        inputs: List of input node names
        outputs: List of output node names
    """

    name: str
    processor: Callable[[Any], Any]
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    def process(self, input_data: Any) -> Any:
        """Process input data through this node.

        Args:
            input_data: Input audio data

        Returns:
            Processed audio data
        """
        try:
            return self.processor(input_data)
        except Exception as e:
            logger.error(f"Error processing node '{self.name}': {e}")
            return input_data  # Pass through on error


class StreamGraph:
    """Audio processing graph with node connections.

    Allows building complex audio processing pipelines by connecting
    multiple processing nodes in a directed graph.

    Attributes:
        sample_rate: Sample rate for all nodes
        nodes: Dictionary of nodes by name
        connections: List of (source, destination) connections
    """

    def __init__(
        self,
        sample_rate: float = 44100.0,
        buffer_size: int = 512,
    ):
        """Initialize stream graph.

        Args:
            sample_rate: Sample rate for graph processing
            buffer_size: Buffer size for real-time processing
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.nodes: Dict[str, StreamNode] = {}
        self.connections: List[Tuple[str, str]] = []
        self._is_active = False
        self._processor: Optional[AudioProcessor] = None

    def add_node(self, name: str, processor: Callable[[Any], Any]) -> StreamNode:
        """Add processing node to graph.

        Args:
            name: Unique name for the node
            processor: Processing function (input_audio) -> output_audio

        Returns:
            Created StreamNode

        Raises:
            ValueError: If node name already exists
        """
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")

        node = StreamNode(name, processor)
        self.nodes[name] = node
        return node

    def connect(self, source: str, destination: str) -> None:
        """Connect two nodes (source → destination).

        Args:
            source: Source node name
            destination: Destination node name

        Raises:
            ValueError: If either node doesn't exist
        """
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' not found")
        if destination not in self.nodes:
            raise ValueError(f"Destination node '{destination}' not found")

        self.connections.append((source, destination))
        self.nodes[source].outputs.append(destination)
        self.nodes[destination].inputs.append(source)

    def _topological_sort(self) -> List[str]:
        """Topologically sort nodes for processing order.

        Returns:
            List of node names in processing order

        Raises:
            ValueError: If graph contains cycles
        """
        # Count incoming edges for each node
        in_degree = {name: len(node.inputs) for name, node in self.nodes.items()}

        # Find all nodes with no incoming edges
        queue_nodes = [name for name, degree in in_degree.items() if degree == 0]
        sorted_nodes = []

        while queue_nodes:
            # Remove node from queue
            current = queue_nodes.pop(0)
            sorted_nodes.append(current)

            # Reduce in-degree for all neighbors
            for output_name in self.nodes[current].outputs:
                in_degree[output_name] -= 1
                if in_degree[output_name] == 0:
                    queue_nodes.append(output_name)

        # Check for cycles
        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("Graph contains cycles - cannot process")

        return sorted_nodes

    def _create_combined_processor(self) -> Callable[[Any], Any]:
        """Create a combined processor function from the graph.

        Returns:
            Combined processing function
        """
        processing_order = self._topological_sort()

        def combined_process(input_data: Any) -> Any:
            """Process data through all nodes in order."""
            # Initialize node outputs
            node_outputs: Dict[str, Any] = {}

            # Process each node in topological order
            for node_name in processing_order:
                node = self.nodes[node_name]

                # Get input data for this node
                if not node.inputs:
                    # No inputs - use graph input
                    node_input = input_data
                elif len(node.inputs) == 1:
                    # Single input - use output from that node
                    node_input = node_outputs[node.inputs[0]]
                else:
                    # Multiple inputs - sum them (simple mixing)
                    try:
                        import numpy as np

                        inputs = [node_outputs[inp] for inp in node.inputs]
                        node_input = np.sum(inputs, axis=0)
                    except ImportError:
                        # Without NumPy, just use first input
                        node_input = node_outputs[node.inputs[0]]

                # Process through this node
                node_outputs[node_name] = node.process(node_input)

            # Return output from the last node
            if processing_order:
                return node_outputs[processing_order[-1]]
            return input_data

        return combined_process

    def start(self) -> None:
        """Start processing graph.

        Creates a combined processor from all nodes and starts real-time processing.

        Raises:
            ValueError: If graph is invalid (cycles, no nodes, etc.)
            RuntimeError: If graph is already active
        """
        if self._is_active:
            raise RuntimeError("Graph is already active")

        if not self.nodes:
            raise ValueError("Cannot start empty graph")

        # Create combined processor
        combined_processor = self._create_combined_processor()

        # Create AudioProcessor with combined function
        self._processor = AudioProcessor(
            combined_processor,
            sample_rate=self.sample_rate,
            buffer_size=self.buffer_size,
        )

        self._processor.start()
        self._is_active = True
        logger.info(f"Started stream graph with {len(self.nodes)} nodes")

    def stop(self) -> None:
        """Stop processing graph."""
        if not self._is_active:
            return

        if self._processor:
            self._processor.stop()

        self._is_active = False
        logger.info("Stopped stream graph")

    @property
    def is_active(self) -> bool:
        """Check if graph is currently active."""
        return self._is_active

    @property
    def latency(self) -> float:
        """Get total graph latency in seconds."""
        if self._processor:
            return self._processor.latency
        return (self.buffer_size * 3) / self.sample_rate

    def __enter__(self) -> "StreamGraph":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()


# ============================================================================
# Convenience Functions
# ============================================================================


def create_loopback(
    channels: int = 2,
    sample_rate: float = 44100.0,
    buffer_size: int = 512,
) -> AudioProcessor:
    """Create a simple audio loopback (input → output).

    Args:
        channels: Number of channels
        sample_rate: Sample rate in Hz
        buffer_size: Buffer size in frames

    Returns:
        AudioProcessor configured for loopback

    Example:
        >>> loopback = create_loopback(buffer_size=256)
        >>> loopback.start()
        >>> time.sleep(5)
        >>> loopback.stop()
    """

    def passthrough(audio_in: Any) -> Any:
        """Pass audio through unchanged."""
        return audio_in

    return AudioProcessor(
        passthrough,
        channels=channels,
        sample_rate=sample_rate,
        buffer_size=buffer_size,
    )


__all__ = [
    "AudioInputStream",
    "AudioOutputStream",
    "AudioProcessor",
    "StreamGraph",
    "StreamNode",
    "create_loopback",
]
