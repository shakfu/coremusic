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

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .. import capi
from .devices import AudioDevice, AudioDeviceManager

if TYPE_CHECKING:
    try:
        from numpy.typing import NDArray as NDArray_
    except ImportError:
        NDArray_ = Any  # type: ignore[misc,assignment]

    NDArray = NDArray_[Any]


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
        device: AudioDevice | None = None,
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
        self._callbacks: list[Callable[[Any, int], None]] = []
        self._is_active = False
        self._audio_unit_id: int | None = None
        self._impl: Any = None
        self._ring: Any = None
        self._drain_thread: threading.Thread | None = None
        self._running = False
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
            logger.warning("NumPy not available - audio data will be provided as bytes")

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

        Registered callbacks are invoked on the CoreAudio input thread with each
        captured buffer. This is best-effort real time.

        Raises:
            RuntimeError: If the stream is already active, no input device is
                available, setup fails, or microphone access is denied.
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

    # kAudioUnitErr_CannotDoInCurrentContext: returned by AudioUnitRender when
    # microphone (TCC) permission has not been granted to the process.
    _PERMISSION_DENIED_STATUS = -10863

    def _setup_audio_unit(self) -> None:
        """Set up the HAL input unit and start ring-buffered capture.

        The capture thread pushes samples into a lock-free ring (no GIL); a
        drain thread pops from the ring and delivers each buffer to registered
        callbacks as a NumPy float32 `(frames, channels)` array (or raw float32
        interleaved bytes without NumPy).

        Raises:
            RuntimeError: If no input device is available or microphone access is
                denied.
        """
        device = self.device
        if device is None:
            raise RuntimeError("No input device available")

        # Ring sized to several device buffers so the drain thread has slack.
        ring = capi.AudioRingBuffer(8 * self.buffer_size, self.channels)
        impl = capi.AudioInputStreamImpl()
        impl.setup_direct(
            ring,
            int(device.object_id),
            float(self.sample_rate),
            int(self.channels),
            max(8192, 4 * self.buffer_size),
        )
        impl.start()

        # Capture failures (notably denied microphone permission) surface
        # asynchronously once the render callback first runs, so poll briefly.
        import time as _time

        for _ in range(5):
            _time.sleep(0.02)
            if impl.last_status == self._PERMISSION_DENIED_STATUS:
                impl.stop()
                impl.close()
                raise RuntimeError(
                    "Microphone access denied. Grant Microphone permission to "
                    "your terminal or app in System Settings > Privacy & "
                    "Security > Microphone, then retry."
                )

        self._impl = impl
        self._ring = ring
        self._running = True
        self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._drain_thread.start()

    def _drain_loop(self) -> None:
        """Pop captured samples off the ring and dispatch to callbacks."""
        import time as _time

        channels = self.channels
        elems = self.buffer_size * channels
        np = self._np
        has_numpy = self._has_numpy
        if has_numpy and np is not None:
            scratch = np.empty(elems, dtype=np.float32)
        else:
            import array

            scratch = array.array("f", bytes(elems * 4))

        while self._running:
            got = self._ring.pop_into(scratch)
            if got == 0:
                _time.sleep(0.001)
                continue
            frame_count = got // channels
            if has_numpy and np is not None:
                flat = np.array(scratch[:got], copy=True)  # own the data past this call
                payload: Any = flat.reshape(-1, channels) if channels > 1 else flat
            else:
                payload = scratch[:got].tobytes()
            with self._lock:
                callbacks = list(self._callbacks)
            for callback in callbacks:
                try:
                    callback(payload, frame_count)
                except Exception as e:
                    logger.error(f"Error in input callback: {e}")

    def _teardown_audio_unit(self) -> None:
        """Stop the capture unit and drain thread, then release resources.

        Order matters: stop the capture unit first so the ring stops filling,
        then join the drain thread, then drop the ring.
        """
        self._running = False
        if self._impl is not None:
            try:
                self._impl.stop()
                self._impl.close()
            except Exception as e:
                logger.warning(f"Error during input stream teardown: {e}")
            finally:
                self._impl = None
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=1.0)
            self._drain_thread = None
        self._ring = None

    @property
    def overruns(self) -> int:
        """Ring-buffer overruns (producer outran the drain thread)."""
        return int(self._ring.overruns) if self._ring is not None else 0

    @property
    def underruns(self) -> int:
        """Ring-buffer underruns (drain thread found the ring empty)."""
        return int(self._ring.underruns) if self._ring is not None else 0

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
        device: AudioDevice | None = None,
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
        self._generator: Callable[[int], Any] | None = None
        self._is_active = False
        self._audio_unit_id: int | None = None
        self._impl: Any = None
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
            logger.warning("NumPy not available - generator must return bytes")

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

        The generator is invoked on the CoreAudio render thread to produce each
        buffer. This is best-effort real time (suitable for tone/synthesis
        generation), not hard-real-time low-latency work.

        Raises:
            RuntimeError: If stream is already active, no generator set, or setup fails
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
        """Set up the generator-driven output unit via the Cython backend.

        Wraps the user generator so its output (bytes or a NumPy float array of
        shape (frames,) or (frames, channels)) is delivered to the render
        callback as float32 interleaved samples.
        """
        generator = self._generator
        if generator is None:
            raise RuntimeError("No generator function set")

        np = self._np
        channels = self.channels
        has_numpy = self._has_numpy

        def adapter(num_frames: int) -> Any:
            data = generator(num_frames)
            if data is None:
                return None
            if has_numpy and np is not None and isinstance(data, np.ndarray):
                arr = np.asarray(data, dtype=np.float32)
                if arr.ndim == 1 and channels > 1:
                    arr = np.repeat(arr[:, None], channels, axis=1)
                return np.ascontiguousarray(arr, dtype=np.float32).tobytes()
            return data

        self._impl = capi.AudioOutputStreamImpl()
        self._impl.setup(adapter, float(self.sample_rate), int(channels))
        self._impl.start()

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
        """Stop and dispose the output unit."""
        if self._impl is not None:
            try:
                self._impl.stop()
                self._impl.close()
            except Exception as e:
                logger.warning(f"Error during output stream teardown: {e}")
            finally:
                self._impl = None

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
        input_device: AudioDevice | None = None,
        output_device: AudioDevice | None = None,
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
        self._input_buffer: Any | None = None
        self._buffer_lock = threading.Lock()
        self.input_stream.add_callback(self._on_input)
        self.output_stream.set_generator(self._generate_output)

        # Two-ring worker state (set up on start()).
        self._out_ring: Any = None
        self._output_impl: Any = None
        self._np: Any = None
        self._has_numpy = False
        try:
            import numpy as np

            self._np = np
            self._has_numpy = True
        except ImportError:
            pass

    def _on_input(self, data: Any, frame_count: int) -> None:
        """Process a captured buffer (runs on the input drain/worker thread).

        Stores the latest input and, when running, applies ``process_func`` and
        enqueues the result into the output ring. Because this runs on the drain
        thread rather than the audio render thread, ``process_func`` never holds
        up the real-time output callback.
        """
        with self._buffer_lock:
            self._input_buffer = data
        if self._out_ring is not None:
            self._enqueue_output(data)

    def _enqueue_output(self, data: Any) -> None:
        """Apply process_func and push the result into the output ring."""
        try:
            result = self.process_func(data)
        except Exception as e:
            logger.error(f"Error in process function: {e}")
            return  # skip this buffer -> the output underruns to silence
        if result is None:
            return
        ring = self._out_ring
        if ring is None:
            return
        if self._has_numpy and self._np is not None and isinstance(
            result, self._np.ndarray
        ):
            arr = self._np.ascontiguousarray(result, dtype=self._np.float32).ravel()
            ring.push_floats(arr)
        elif isinstance(result, (bytes, bytearray)):
            ring.push_bytes(result)
        else:
            logger.error(
                "process_func must return a NumPy float array or bytes, "
                f"got {type(result).__name__}"
            )

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
        """Start real-time processing (two-ring worker model).

        The capture callback pushes into the input stream's ring, whose drain
        thread applies ``process_func`` and pushes into an output ring; the
        render callback pops that output ring. Both audio threads run without
        the GIL -- ``process_func`` executes on the drain/worker thread.

        Raises:
            RuntimeError: If setup fails or microphone access is denied.
        """
        out_ring = capi.AudioRingBuffer(8 * self.buffer_size, self.channels)
        output_impl = capi.AudioOutputStreamImpl()
        output_impl.setup_direct(
            out_ring, float(self.sample_rate), int(self.channels)
        )

        # Publish the ring before the input drain thread starts so _on_input can
        # enqueue immediately once capture begins.
        self._out_ring = out_ring
        try:
            self.input_stream.start()  # may raise (e.g. microphone permission)
        except Exception:
            output_impl.close()
            self._out_ring = None
            raise

        output_impl.start()
        self._output_impl = output_impl

    def stop(self) -> None:
        """Stop real-time processing.

        Stops the consumer (output) first, then the producer (input capture and
        its worker thread), then drops the shared output ring.
        """
        if self._output_impl is not None:
            try:
                self._output_impl.stop()
                self._output_impl.close()
            except Exception as e:
                logger.warning(f"Error stopping processor output: {e}")
            finally:
                self._output_impl = None
        self.input_stream.stop()
        self._out_ring = None

    @property
    def is_active(self) -> bool:
        """Check if processor is currently active."""
        return (
            self.input_stream.is_active
            and self._output_impl is not None
            and self._output_impl.is_active
        )

    @property
    def overruns(self) -> int:
        """Output-ring overruns (worker outran playback)."""
        return int(self._out_ring.overruns) if self._out_ring is not None else 0

    @property
    def underruns(self) -> int:
        """Output-ring underruns (playback outran the worker)."""
        return int(self._out_ring.underruns) if self._out_ring is not None else 0

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
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

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
        self.nodes: dict[str, StreamNode] = {}
        self.connections: list[tuple[str, str]] = []
        self._is_active = False
        self._processor: AudioProcessor | None = None

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

    def _topological_sort(self) -> list[str]:
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
            node_outputs: dict[str, Any] = {}

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


class DirectLoopback:
    """Zero-copy input-to-output loopback with no Python on the audio path.

    A single lock-free ring is shared by the capture and render threads: the
    HAL input callback pushes captured frames and the output render callback
    pops them, both in C without acquiring the GIL. This is lower-latency and
    glitch-resistant compared to routing audio through a Python callback.

    Attributes:
        channels: Number of channels (input and output must match)
        sample_rate: Sample rate in Hz
        buffer_size: Device buffer size in frames
    """

    _PERMISSION_DENIED_STATUS = -10863

    def __init__(
        self,
        channels: int = 2,
        sample_rate: float = 44100.0,
        buffer_size: int = 512,
    ):
        self.channels = channels
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.device = AudioDeviceManager.get_default_input_device()
        self._ring: Any = None
        self._input: Any = None
        self._output: Any = None
        self._is_active = False

    def start(self) -> None:
        """Start the loopback.

        Raises:
            RuntimeError: If already active, no input device is available, or
                microphone access is denied.
        """
        if self._is_active:
            raise RuntimeError("Loopback is already active")
        if self.device is None:
            raise RuntimeError("No input device available")

        ring = capi.AudioRingBuffer(8 * self.buffer_size, self.channels)
        input_impl = capi.AudioInputStreamImpl()
        output_impl = capi.AudioOutputStreamImpl()

        input_impl.setup_direct(
            ring,
            int(self.device.object_id),
            float(self.sample_rate),
            int(self.channels),
            max(8192, 4 * self.buffer_size),
        )
        input_impl.start()

        # Surface a denied-microphone-permission failure clearly.
        import time as _time

        for _ in range(5):
            _time.sleep(0.02)
            if input_impl.last_status == self._PERMISSION_DENIED_STATUS:
                input_impl.stop()
                input_impl.close()
                raise RuntimeError(
                    "Microphone access denied. Grant Microphone permission to "
                    "your terminal or app in System Settings > Privacy & "
                    "Security > Microphone, then retry."
                )

        output_impl.setup_direct(ring, float(self.sample_rate), int(self.channels))
        output_impl.start()

        self._ring = ring
        self._input = input_impl
        self._output = output_impl
        self._is_active = True
        logger.info(
            f"Started loopback: {self.channels} channels, {self.sample_rate} Hz, "
            f"{self.buffer_size} frames"
        )

    def stop(self) -> None:
        """Stop the loopback and release resources.

        Order matters: stop the consumer (output) first, then the producer
        (input), then drop the shared ring.
        """
        if not self._is_active:
            return
        if self._output is not None:
            try:
                self._output.stop()
                self._output.close()
            except Exception as e:
                logger.warning(f"Error stopping loopback output: {e}")
            finally:
                self._output = None
        if self._input is not None:
            try:
                self._input.stop()
                self._input.close()
            except Exception as e:
                logger.warning(f"Error stopping loopback input: {e}")
            finally:
                self._input = None
        self._ring = None
        self._is_active = False
        logger.info("Stopped loopback")

    @property
    def is_active(self) -> bool:
        """Check whether the loopback is running."""
        return self._is_active

    @property
    def latency(self) -> float:
        """Approximate one-way latency in seconds (ring fill target)."""
        return self.buffer_size / self.sample_rate

    @property
    def overruns(self) -> int:
        """Ring overruns (capture outran playback)."""
        return int(self._ring.overruns) if self._ring is not None else 0

    @property
    def underruns(self) -> int:
        """Ring underruns (playback outran capture)."""
        return int(self._ring.underruns) if self._ring is not None else 0

    def __enter__(self) -> "DirectLoopback":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


def create_loopback(
    channels: int = 2,
    sample_rate: float = 44100.0,
    buffer_size: int = 512,
) -> DirectLoopback:
    """Create a zero-GIL audio loopback (input → output).

    The capture and render threads exchange audio through a single lock-free
    ring with no Python on the audio path (see :class:`DirectLoopback`).

    Args:
        channels: Number of channels
        sample_rate: Sample rate in Hz
        buffer_size: Buffer size in frames

    Returns:
        A :class:`DirectLoopback` (start/stop, context manager, and
        overrun/underrun counters).

    Example:
        >>> with create_loopback(buffer_size=256) as loopback:
        ...     time.sleep(5)
    """
    return DirectLoopback(
        channels=channels,
        sample_rate=sample_rate,
        buffer_size=buffer_size,
    )


__all__ = [
    "AudioInputStream",
    "AudioOutputStream",
    "AudioProcessor",
    "DirectLoopback",
    "StreamGraph",
    "StreamNode",
    "create_loopback",
]
