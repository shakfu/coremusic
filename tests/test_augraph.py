#!/usr/bin/env python3
"""Tests for AUGraph API (both functional and object-oriented)."""

import pytest
import time

import coremusic as cm


class TestAUGraphFunctionalAPI:
    """Test AUGraph functional API"""

    def test_au_graph_creation_disposal(self):
        """Test creating and disposing an AUGraph"""
        graph_id = cm.au_graph_new()
        assert graph_id is not None
        assert graph_id > 0

        cm.au_graph_dispose(graph_id)

    def test_au_graph_lifecycle(self):
        """Test AUGraph lifecycle: open -> initialize -> start -> stop"""
        graph_id = cm.au_graph_new()

        try:
            # Initially not open
            assert not cm.au_graph_is_open(graph_id)
            assert not cm.au_graph_is_initialized(graph_id)
            assert not cm.au_graph_is_running(graph_id)

            # Open the graph
            cm.au_graph_open(graph_id)
            assert cm.au_graph_is_open(graph_id)
            assert not cm.au_graph_is_initialized(graph_id)

            # Initialize the graph
            cm.au_graph_initialize(graph_id)
            assert cm.au_graph_is_open(graph_id)
            assert cm.au_graph_is_initialized(graph_id)
            assert not cm.au_graph_is_running(graph_id)

            # Close the graph
            cm.au_graph_close(graph_id)
            assert not cm.au_graph_is_open(graph_id)

        finally:
            cm.au_graph_dispose(graph_id)

    def test_au_graph_add_remove_nodes(self):
        """Test adding and removing nodes"""
        graph_id = cm.au_graph_new()

        try:
            # Initially no nodes
            assert cm.au_graph_get_node_count(graph_id) == 0

            # Add a default output node
            desc = {
                'type': cm.fourchar_to_int('auou'),  # kAudioUnitType_Output
                'subtype': cm.fourchar_to_int('def '),  # kAudioUnitSubType_DefaultOutput
                'manufacturer': cm.fourchar_to_int('appl'),  # kAudioUnitManufacturer_Apple
                'flags': 0,
                'flags_mask': 0
            }

            node_id = cm.au_graph_add_node(graph_id, desc)
            assert node_id is not None

            # Should now have 1 node
            assert cm.au_graph_get_node_count(graph_id) == 1

            # Get the node at index 0
            retrieved_node = cm.au_graph_get_ind_node(graph_id, 0)
            assert retrieved_node == node_id

            # Get node info
            node_desc, audio_unit_id = cm.au_graph_node_info(graph_id, node_id)
            assert node_desc['type'] == desc['type']
            assert node_desc['subtype'] == desc['subtype']
            # audio_unit_id is 0 until graph is opened
            assert audio_unit_id >= 0

            # Remove the node
            cm.au_graph_remove_node(graph_id, node_id)
            assert cm.au_graph_get_node_count(graph_id) == 0

        finally:
            cm.au_graph_dispose(graph_id)

    def test_au_graph_cpu_load(self):
        """Test getting CPU load"""
        graph_id = cm.au_graph_new()

        try:
            cpu_load = cm.au_graph_get_cpu_load(graph_id)
            assert isinstance(cpu_load, float)
            assert 0.0 <= cpu_load <= 1.0

            max_load = cm.au_graph_get_max_cpu_load(graph_id)
            assert isinstance(max_load, float)
            assert 0.0 <= max_load <= 1.0

        finally:
            cm.au_graph_dispose(graph_id)


class TestAUGraphOO:
    """Test AUGraph object-oriented wrapper"""

    def test_au_graph_creation(self):
        """Test AUGraph object creation"""
        graph = cm.AUGraph()
        assert isinstance(graph, cm.AUGraph)
        assert isinstance(graph, cm.capi.CoreAudioObject)
        assert not graph.is_disposed
        assert graph.object_id != 0

    def test_au_graph_lifecycle(self):
        """Test AUGraph lifecycle with OO API"""
        graph = cm.AUGraph()

        # Initially closed
        assert not graph.is_open
        assert not graph.is_initialized
        assert not graph.is_running

        # Open
        graph.open()
        assert graph.is_open
        assert not graph.is_initialized

        # Initialize
        graph.initialize()
        assert graph.is_open
        assert graph.is_initialized
        assert not graph.is_running

        # Close
        graph.close()
        assert not graph.is_open

        # Dispose
        graph.dispose()
        assert graph.is_disposed

    def test_au_graph_method_chaining(self):
        """Test method chaining for open() and initialize()"""
        graph = cm.AUGraph()

        # Method chaining
        result = graph.open().initialize()
        assert result is graph
        assert graph.is_open
        assert graph.is_initialized

        graph.dispose()

    def test_au_graph_context_manager(self):
        """Test AUGraph as context manager"""
        with cm.AUGraph() as graph:
            assert isinstance(graph, cm.AUGraph)
            assert not graph.is_disposed
            graph.open()
            assert graph.is_open

        # Should be disposed after exiting context
        assert graph.is_disposed

    def test_au_graph_add_remove_nodes(self):
        """Test adding and removing nodes with OO API"""
        graph = cm.AUGraph()

        try:
            # Initially no nodes
            assert graph.node_count == 0

            # Create output unit description
            desc = cm.AudioComponentDescription(
                type='auou',  # kAudioUnitType_Output
                subtype='def ',  # kAudioUnitSubType_DefaultOutput
                manufacturer='appl'  # kAudioUnitManufacturer_Apple
            )

            # Add node
            node_id = graph.add_node(desc)
            assert node_id is not None
            assert graph.node_count == 1

            # Get node at index
            retrieved_node = graph.get_node_at_index(0)
            assert retrieved_node == node_id

            # Get node info
            node_desc, audio_unit_id = graph.get_node_info(node_id)
            assert isinstance(node_desc, cm.AudioComponentDescription)
            assert node_desc.type == desc.type
            assert node_desc.subtype == desc.subtype
            # audio_unit_id is 0 until graph is opened
            assert audio_unit_id >= 0

            # Remove node
            graph.remove_node(node_id)
            assert graph.node_count == 0

        finally:
            graph.dispose()

    def test_au_graph_connections(self):
        """Test connecting nodes"""
        graph = cm.AUGraph()

        try:
            graph.open()

            # Add a music device (generator)
            generator_desc = cm.AudioComponentDescription(
                type='aumu',  # kAudioUnitType_MusicDevice
                subtype='dls ',  # kAudioUnitSubType_DLSSynth
                manufacturer='appl'
            )
            generator_node = graph.add_node(generator_desc)

            # Add output
            output_desc = cm.AudioComponentDescription(
                type='auou',
                subtype='def ',
                manufacturer='appl'
            )
            output_node = graph.add_node(output_desc)

            assert graph.node_count == 2

            # Connect generator to output
            graph.connect(
                source_node=generator_node,
                source_output=0,
                dest_node=output_node,
                dest_input=0
            )

            # Update graph to apply changes
            is_updated = graph.update()
            assert isinstance(is_updated, bool)

            # Disconnect
            graph.disconnect(dest_node=output_node, dest_input=0)

            # Clear all connections
            graph.clear_connections()

        finally:
            graph.dispose()

    def test_au_graph_cpu_load_properties(self):
        """Test CPU load properties"""
        graph = cm.AUGraph()

        try:
            cpu_load = graph.cpu_load
            assert isinstance(cpu_load, float)
            assert 0.0 <= cpu_load <= 1.0

            max_load = graph.max_cpu_load
            assert isinstance(max_load, float)
            assert 0.0 <= max_load <= 1.0

        finally:
            graph.dispose()

    def test_au_graph_repr(self):
        """Test AUGraph string representation"""
        graph = cm.AUGraph()
        repr_str = repr(graph)
        assert 'AUGraph' in repr_str
        assert 'nodes=0' in repr_str

        graph.open()
        repr_str = repr(graph)
        assert 'open' in repr_str

        graph.initialize()
        repr_str = repr(graph)
        assert 'initialized' in repr_str

        graph.dispose()
        repr_str = repr(graph)
        assert 'disposed' in repr_str

    def test_au_graph_operations_after_disposal(self):
        """Test that operations on disposed graph raise errors"""
        graph = cm.AUGraph()
        graph.dispose()

        with pytest.raises(RuntimeError, match="has been disposed"):
            graph.open()

        with pytest.raises(RuntimeError, match="has been disposed"):
            _ = graph.is_open

        with pytest.raises(RuntimeError, match="has been disposed"):
            _ = graph.node_count

    def test_au_graph_error_handling(self):
        """Test AUGraph error handling"""
        graph = cm.AUGraph()

        try:
            # Try to get invalid node
            with pytest.raises(cm.AUGraphError):
                graph.get_node_at_index(999)

        finally:
            graph.dispose()

    def test_au_graph_double_dispose(self):
        """Test that double dispose is safe"""
        graph = cm.AUGraph()
        graph.dispose()
        graph.dispose()  # Should not raise
        assert graph.is_disposed


class TestAUGraphIntegration:
    """Integration tests for AUGraph with real audio setup"""

    def test_au_graph_simple_playback_setup(self):
        """Test setting up a simple playback graph"""
        graph = cm.AUGraph()

        try:
            # Open and initialize
            graph.open().initialize()

            # Add output node
            output_desc = cm.AudioComponentDescription(
                type='auou',
                subtype='def ',
                manufacturer='appl'
            )
            output_node = graph.add_node(output_desc)

            # Update and verify
            assert graph.update()
            assert graph.node_count == 1

            # We could start the graph here, but let's not actually play audio in tests
            # graph.start()
            # time.sleep(0.1)
            # graph.stop()

        finally:
            graph.uninitialize()
            graph.close()
            graph.dispose()
