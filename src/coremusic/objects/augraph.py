"""AUGraph classes for coremusic.

This module provides classes for working with Audio Unit Graphs:
- AUGraph: Audio Unit Graph for managing and connecting multiple AudioUnits
"""

from __future__ import annotations

from typing import Any, Tuple

from .. import capi
from .audiounit import AudioComponentDescription
from .exceptions import AUGraphError

__all__ = [
    "AUGraph",
]


class AUGraph(capi.CoreAudioObject):
    """Audio Unit Graph for managing and connecting multiple AudioUnits

    AUGraph provides a high-level API for creating and managing graphs of
    AudioUnits, including connections between nodes and overall graph lifecycle.

    Note: AUGraph is deprecated by Apple in favor of AVAudioEngine, but remains
    fully functional and useful for advanced audio processing scenarios.
    """

    def __init__(self) -> None:
        """Create a new AUGraph

        Raises:
            AUGraphError: If graph creation fails
        """
        super().__init__()
        try:
            graph_id = capi.au_graph_new()
            self._set_object_id(graph_id)
        except Exception as e:
            raise AUGraphError(f"Failed to create AUGraph: {e}")

    def open(self) -> "AUGraph":
        """Open the graph (opens AudioUnits but doesn't initialize them)

        Returns:
            Self for method chaining

        Raises:
            AUGraphError: If open fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_open(self.object_id)
            return self
        except Exception as e:
            raise AUGraphError(f"Failed to open graph: {e}")

    def close(self) -> None:
        """Close the graph (closes all AudioUnits)

        Raises:
            AUGraphError: If close fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_close(self.object_id)
        except Exception as e:
            raise AUGraphError(f"Failed to close graph: {e}")

    def initialize(self) -> "AUGraph":
        """Initialize the graph (prepares all AudioUnits for rendering)

        Returns:
            Self for method chaining

        Raises:
            AUGraphError: If initialization fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_initialize(self.object_id)
            return self
        except Exception as e:
            raise AUGraphError(f"Failed to initialize graph: {e}")

    def uninitialize(self) -> None:
        """Uninitialize the graph

        Raises:
            AUGraphError: If uninitialization fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_uninitialize(self.object_id)
        except Exception as e:
            raise AUGraphError(f"Failed to uninitialize graph: {e}")

    def start(self) -> None:
        """Start the graph (begins audio rendering)

        Raises:
            AUGraphError: If start fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_start(self.object_id)
        except Exception as e:
            raise AUGraphError(f"Failed to start graph: {e}")

    def stop(self) -> None:
        """Stop the graph (stops audio rendering)

        Raises:
            AUGraphError: If stop fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_stop(self.object_id)
        except Exception as e:
            raise AUGraphError(f"Failed to stop graph: {e}")

    @property
    def is_open(self) -> bool:
        """Check if the graph is open"""
        self._ensure_not_disposed()
        return capi.au_graph_is_open(self.object_id)

    @property
    def is_initialized(self) -> bool:
        """Check if the graph is initialized"""
        self._ensure_not_disposed()
        return capi.au_graph_is_initialized(self.object_id)

    @property
    def is_running(self) -> bool:
        """Check if the graph is running"""
        self._ensure_not_disposed()
        return capi.au_graph_is_running(self.object_id)

    def add_node(self, description: AudioComponentDescription) -> int:
        """Add a node to the graph

        Args:
            description: AudioComponentDescription for the node

        Returns:
            Node ID

        Raises:
            AUGraphError: If adding node fails

        Example::

            import coremusic as cm

            # Create a graph with an effect and output node
            with cm.AUGraph() as graph:
                # Add a reverb effect
                reverb_desc = cm.AudioComponentDescription(
                    type='aufx',
                    subtype='rvb2',
                    manufacturer='appl'
                )
                reverb_node = graph.add_node(reverb_desc)

                # Add default output
                output_desc = cm.AudioComponentDescription(
                    type='auou',
                    subtype='def ',
                    manufacturer='appl'
                )
                output_node = graph.add_node(output_desc)

                # Connect reverb -> output
                graph.connect_nodes(reverb_node, 0, output_node, 0)
        """
        self._ensure_not_disposed()
        try:
            desc_dict = {
                "type": capi.fourchar_to_int(description.type)
                if isinstance(description.type, str)
                else description.type,
                "subtype": capi.fourchar_to_int(description.subtype)
                if isinstance(description.subtype, str)
                else description.subtype,
                "manufacturer": capi.fourchar_to_int(description.manufacturer)
                if isinstance(description.manufacturer, str)
                else description.manufacturer,
                "flags": description.flags,
                "flags_mask": description.flags_mask,
            }
            return capi.au_graph_add_node(self.object_id, desc_dict)
        except Exception as e:
            raise AUGraphError(f"Failed to add node: {e}")

    def remove_node(self, node_id: int) -> None:
        """Remove a node from the graph

        Args:
            node_id: Node ID to remove

        Raises:
            AUGraphError: If removing node fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_remove_node(self.object_id, node_id)
        except Exception as e:
            raise AUGraphError(f"Failed to remove node: {e}")

    @property
    def node_count(self) -> int:
        """Get the number of nodes in the graph"""
        self._ensure_not_disposed()
        return capi.au_graph_get_node_count(self.object_id)

    def get_node_at_index(self, index: int) -> int:
        """Get node ID at the specified index

        Args:
            index: Node index (must be non-negative and < node_count)

        Returns:
            Node ID

        Raises:
            ValueError: If index is negative
            IndexError: If index >= node_count
            AUGraphError: If getting node fails
        """
        if index < 0:
            raise ValueError(f"index must be non-negative, got {index}")

        self._ensure_not_disposed()

        count = self.node_count
        if index >= count:
            if count == 0:
                raise IndexError(f"node index {index} out of range (graph has no nodes)")
            raise IndexError(f"node index {index} out of range (0-{count-1})")

        try:
            return capi.au_graph_get_ind_node(self.object_id, index)
        except Exception as e:
            raise AUGraphError(f"Failed to get node at index {index}: {e}")

    def get_node_info(self, node_id: int) -> Tuple[AudioComponentDescription, int]:
        """Get information about a node

        Args:
            node_id: Node ID

        Returns:
            Tuple of (AudioComponentDescription, AudioUnit ID)

        Raises:
            AUGraphError: If getting node info fails
        """
        self._ensure_not_disposed()
        try:
            result: Any = capi.au_graph_node_info(self.object_id, node_id)
            desc_dict: dict[str, int] = result[0]
            audio_unit_id: int = result[1]

            # Convert back to AudioComponentDescription
            desc = AudioComponentDescription(
                type=capi.int_to_fourchar(desc_dict["type"]),
                subtype=capi.int_to_fourchar(desc_dict["subtype"]),
                manufacturer=capi.int_to_fourchar(desc_dict["manufacturer"]),
                flags=desc_dict["flags"],
                flags_mask=desc_dict["flags_mask"],
            )

            return (desc, audio_unit_id)
        except Exception as e:
            raise AUGraphError(f"Failed to get node info: {e}")

    def connect(
        self, source_node: int, source_output: int, dest_node: int, dest_input: int
    ) -> None:
        """Connect two nodes in the graph

        Args:
            source_node: Source node ID
            source_output: Source output bus number
            dest_node: Destination node ID
            dest_input: Destination input bus number

        Raises:
            AUGraphError: If connection fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_connect_node_input(
                self.object_id, source_node, source_output, dest_node, dest_input
            )
        except Exception as e:
            raise AUGraphError(f"Failed to connect nodes: {e}")

    def disconnect(self, dest_node: int, dest_input: int) -> None:
        """Disconnect a node's input

        Args:
            dest_node: Destination node ID
            dest_input: Destination input bus number

        Raises:
            AUGraphError: If disconnection fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_disconnect_node_input(self.object_id, dest_node, dest_input)
        except Exception as e:
            raise AUGraphError(f"Failed to disconnect node: {e}")

    def clear_connections(self) -> None:
        """Clear all connections in the graph

        Raises:
            AUGraphError: If clearing connections fails
        """
        self._ensure_not_disposed()
        try:
            capi.au_graph_clear_connections(self.object_id)
        except Exception as e:
            raise AUGraphError(f"Failed to clear connections: {e}")

    def update(self) -> bool:
        """Update the graph after making changes

        Returns:
            True if update completed immediately, False if pending

        Raises:
            AUGraphError: If update fails
        """
        self._ensure_not_disposed()
        try:
            return capi.au_graph_update(self.object_id)
        except Exception as e:
            raise AUGraphError(f"Failed to update graph: {e}")

    @property
    def cpu_load(self) -> float:
        """Get current CPU load (0.0-1.0)"""
        self._ensure_not_disposed()
        return capi.au_graph_get_cpu_load(self.object_id)

    @property
    def max_cpu_load(self) -> float:
        """Get maximum CPU load since last query (0.0-1.0)"""
        self._ensure_not_disposed()
        return capi.au_graph_get_max_cpu_load(self.object_id)

    def dispose(self) -> None:
        """Dispose of the graph"""
        if not self.is_disposed:
            try:
                capi.au_graph_dispose(self.object_id)
            except Exception:
                pass  # Best effort cleanup
            finally:
                super().dispose()

    def __enter__(self) -> "AUGraph":
        """Enter context manager"""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and dispose"""
        self.dispose()

    def __repr__(self) -> str:
        status = []
        if not self.is_disposed:
            try:
                if self.is_open:
                    status.append("open")
                if self.is_initialized:
                    status.append("initialized")
                if self.is_running:
                    status.append("running")
            except Exception:
                pass
        else:
            status.append("disposed")

        status_str = ", ".join(status) if status else "closed"
        return f"AUGraph({status_str}, nodes={self.node_count if not self.is_disposed else 0})"
