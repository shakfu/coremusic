"""Tests for AUGraph API (both functional and object-oriented)."""

import pytest
import time
import coremusic as cm
import coremusic.capi as capi


class TestAUGraphFunctionalAPI:
    """Test AUGraph functional API"""

    def test_au_graph_creation_disposal(self):
        """Test creating and disposing an AUGraph"""
        graph_id = capi.au_graph_new()
        assert graph_id is not None
        assert graph_id > 0
        capi.au_graph_dispose(graph_id)

    def test_au_graph_lifecycle(self):
        """Test AUGraph lifecycle: open -> initialize -> start -> stop"""
        graph_id = capi.au_graph_new()
        try:
            assert not capi.au_graph_is_open(graph_id)
            assert not capi.au_graph_is_initialized(graph_id)
            assert not capi.au_graph_is_running(graph_id)
            capi.au_graph_open(graph_id)
            assert capi.au_graph_is_open(graph_id)
            assert not capi.au_graph_is_initialized(graph_id)
            capi.au_graph_initialize(graph_id)
            assert capi.au_graph_is_open(graph_id)
            assert capi.au_graph_is_initialized(graph_id)
            assert not capi.au_graph_is_running(graph_id)
            capi.au_graph_close(graph_id)
            assert not capi.au_graph_is_open(graph_id)
        finally:
            capi.au_graph_dispose(graph_id)

    def test_au_graph_add_remove_nodes(self):
        """Test adding and removing nodes"""
        graph_id = capi.au_graph_new()
        try:
            assert capi.au_graph_get_node_count(graph_id) == 0
            desc = {
                "type": capi.fourchar_to_int("auou"),
                "subtype": capi.fourchar_to_int("def "),
                "manufacturer": capi.fourchar_to_int("appl"),
                "flags": 0,
                "flags_mask": 0,
            }
            node_id = capi.au_graph_add_node(graph_id, desc)
            assert node_id is not None
            assert capi.au_graph_get_node_count(graph_id) == 1
            retrieved_node = capi.au_graph_get_ind_node(graph_id, 0)
            assert retrieved_node == node_id
            node_desc, audio_unit_id = capi.au_graph_node_info(graph_id, node_id)
            assert node_desc["type"] == desc["type"]
            assert node_desc["subtype"] == desc["subtype"]
            assert audio_unit_id >= 0
            capi.au_graph_remove_node(graph_id, node_id)
            assert capi.au_graph_get_node_count(graph_id) == 0
        finally:
            capi.au_graph_dispose(graph_id)

    def test_au_graph_cpu_load(self):
        """Test getting CPU load"""
        graph_id = capi.au_graph_new()
        try:
            cpu_load = capi.au_graph_get_cpu_load(graph_id)
            assert isinstance(cpu_load, float)
            assert 0.0 <= cpu_load <= 1.0
            max_load = capi.au_graph_get_max_cpu_load(graph_id)
            assert isinstance(max_load, float)
            assert 0.0 <= max_load <= 1.0
        finally:
            capi.au_graph_dispose(graph_id)


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
        assert not graph.is_open
        assert not graph.is_initialized
        assert not graph.is_running
        graph.open()
        assert graph.is_open
        assert not graph.is_initialized
        graph.initialize()
        assert graph.is_open
        assert graph.is_initialized
        assert not graph.is_running
        graph.close()
        assert not graph.is_open
        graph.dispose()
        assert graph.is_disposed

    def test_au_graph_method_chaining(self):
        """Test method chaining for open() and initialize()"""
        graph = cm.AUGraph()
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
        assert graph.is_disposed

    def test_au_graph_add_remove_nodes(self):
        """Test adding and removing nodes with OO API"""
        graph = cm.AUGraph()
        try:
            assert graph.node_count == 0
            desc = cm.AudioComponentDescription(
                type="auou", subtype="def ", manufacturer="appl"
            )
            node_id = graph.add_node(desc)
            assert node_id is not None
            assert graph.node_count == 1
            retrieved_node = graph.get_node_at_index(0)
            assert retrieved_node == node_id
            node_desc, audio_unit_id = graph.get_node_info(node_id)
            assert isinstance(node_desc, cm.AudioComponentDescription)
            assert node_desc.type == desc.type
            assert node_desc.subtype == desc.subtype
            assert audio_unit_id >= 0
            graph.remove_node(node_id)
            assert graph.node_count == 0
        finally:
            graph.dispose()

    def test_au_graph_connections(self):
        """Test connecting nodes"""
        graph = cm.AUGraph()
        try:
            graph.open()
            generator_desc = cm.AudioComponentDescription(
                type="aumu", subtype="dls ", manufacturer="appl"
            )
            generator_node = graph.add_node(generator_desc)
            output_desc = cm.AudioComponentDescription(
                type="auou", subtype="def ", manufacturer="appl"
            )
            output_node = graph.add_node(output_desc)
            assert graph.node_count == 2
            graph.connect(
                source_node=generator_node,
                source_output=0,
                dest_node=output_node,
                dest_input=0,
            )
            is_updated = graph.update()
            assert isinstance(is_updated, bool)
            graph.disconnect(dest_node=output_node, dest_input=0)
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
        assert "AUGraph" in repr_str
        assert "nodes=0" in repr_str
        graph.open()
        repr_str = repr(graph)
        assert "open" in repr_str
        graph.initialize()
        repr_str = repr(graph)
        assert "initialized" in repr_str
        graph.dispose()
        repr_str = repr(graph)
        assert "disposed" in repr_str

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
            # Out-of-range index raises IndexError
            with pytest.raises(IndexError):
                graph.get_node_at_index(999)

            # Negative index raises ValueError
            with pytest.raises(ValueError):
                graph.get_node_at_index(-1)
        finally:
            graph.dispose()

    def test_au_graph_double_dispose(self):
        """Test that double dispose is safe"""
        graph = cm.AUGraph()
        graph.dispose()
        graph.dispose()
        assert graph.is_disposed


class TestAUGraphIntegration:
    """Integration tests for AUGraph with real audio setup"""

    def test_au_graph_simple_playback_setup(self):
        """Test setting up a simple playback graph"""
        graph = cm.AUGraph()
        try:
            graph.open().initialize()
            output_desc = cm.AudioComponentDescription(
                type="auou", subtype="def ", manufacturer="appl"
            )
            output_node = graph.add_node(output_desc)
            assert graph.update()
            assert graph.node_count == 1
        finally:
            graph.uninitialize()
            graph.close()
            graph.dispose()
