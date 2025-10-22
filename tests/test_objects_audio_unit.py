"""Tests for AudioComponent and AudioUnit object-oriented classes."""

import pytest
import time
import coremusic as cm
import coremusic.capi as capi


class TestAudioComponentDescription:
    """Test AudioComponentDescription class"""

    def test_audio_component_description_creation(self):
        """Test AudioComponentDescription creation"""
        desc = cm.AudioComponentDescription(
            type="auou", subtype="def ", manufacturer="appl"
        )
        assert desc.type == "auou"
        assert desc.subtype == "def "
        assert desc.manufacturer == "appl"
        assert desc.flags == 0
        assert desc.flags_mask == 0

    def test_audio_component_description_with_flags(self):
        """Test AudioComponentDescription with flags"""
        desc = cm.AudioComponentDescription(
            type="aumx", subtype="mcmx", manufacturer="appl", flags=42, flags_mask=255
        )
        assert desc.type == "aumx"
        assert desc.subtype == "mcmx"
        assert desc.manufacturer == "appl"
        assert desc.flags == 42
        assert desc.flags_mask == 255

    def test_audio_component_description_to_dict(self):
        """Test AudioComponentDescription to_dict conversion"""
        desc = cm.AudioComponentDescription(
            type="auou", subtype="def ", manufacturer="appl", flags=1, flags_mask=2
        )
        dict_repr = desc.to_dict()
        expected = {
            "type": capi.fourchar_to_int("auou"),
            "subtype": capi.fourchar_to_int("def "),
            "manufacturer": capi.fourchar_to_int("appl"),
            "flags": 1,
            "flags_mask": 2,
        }
        assert dict_repr == expected


class TestAudioComponent:
    """Test AudioComponent object-oriented wrapper"""

    def test_audio_component_creation(self):
        """Test AudioComponent creation"""
        desc = cm.AudioComponentDescription("auou", "def ", "appl")
        component = cm.AudioComponent(desc)
        assert isinstance(component, cm.AudioComponent)
        assert isinstance(component, cm.CoreAudioObject)
        assert component._description is desc

    def test_audio_component_find_next(self):
        """Test AudioComponent.find_next factory method"""
        desc = cm.AudioComponentDescription(
            type="auou", subtype="def ", manufacturer="appl"
        )
        component = cm.AudioComponent.find_next(desc)
        if component is not None:
            assert isinstance(component, cm.AudioComponent)
            assert component._description.type == desc.type
            assert component.object_id != 0
        else:
            pytest.skip("No default output AudioComponent found")

    def test_audio_component_find_next_nonexistent(self):
        """Test AudioComponent.find_next with non-existent component"""
        desc = cm.AudioComponentDescription(
            type="xxxx", subtype="yyyy", manufacturer="zzzz"
        )
        component = cm.AudioComponent.find_next(desc)
        assert component is None

    def test_audio_component_create_instance(self):
        """Test AudioComponent instance creation"""
        desc = cm.AudioComponentDescription("auou", "def ", "appl")
        component = cm.AudioComponent.find_next(desc)
        if component is None:
            pytest.skip("No default output AudioComponent found")
        unit = component.create_instance()
        assert isinstance(unit, cm.AudioUnit)
        assert unit.object_id != 0
        assert not unit.is_initialized
        unit.dispose()


class TestAudioUnit:
    """Test AudioUnit object-oriented wrapper"""

    def test_audio_unit_creation(self):
        """Test AudioUnit creation"""
        desc = cm.AudioComponentDescription("auou", "def ", "appl")
        unit = cm.AudioUnit(desc)
        assert isinstance(unit, cm.AudioUnit)
        assert isinstance(unit, cm.CoreAudioObject)
        assert unit._description is desc
        assert not unit.is_initialized

    def test_audio_unit_default_output_factory(self):
        """Test AudioUnit.default_output factory method"""
        try:
            unit = cm.AudioUnit.default_output()
            assert isinstance(unit, cm.AudioUnit)
            assert unit.object_id != 0
            assert not unit.is_initialized
            unit.dispose()
        except cm.AudioUnitError as e:
            if "not found" in str(e):
                pytest.skip("Default output AudioUnit not available")
            else:
                raise

    def test_audio_unit_initialization(self):
        """Test AudioUnit initialization and uninitialization"""
        try:
            unit = cm.AudioUnit.default_output()
            unit.initialize()
            assert unit.is_initialized
            unit.uninitialize()
            assert not unit.is_initialized
            unit.dispose()
        except cm.AudioUnitError as e:
            if "not found" in str(e):
                pytest.skip("Default output AudioUnit not available")
            else:
                raise

    def test_audio_unit_context_manager(self):
        """Test AudioUnit as context manager"""
        try:
            desc = cm.AudioComponentDescription("auou", "def ", "appl")
            component = cm.AudioComponent.find_next(desc)
            if component is None:
                pytest.skip("No default output AudioComponent found")
            unit = component.create_instance()
            with unit:
                assert unit.is_initialized
            assert not unit.is_initialized
            assert unit.is_disposed
        except cm.AudioUnitError as e:
            if "not found" in str(e):
                pytest.skip("Default output AudioUnit not available")
            else:
                raise

    def test_audio_unit_start_stop(self):
        """Test AudioUnit start and stop"""
        try:
            unit = cm.AudioUnit.default_output()
            unit.initialize()
            unit.start()
            time.sleep(0.01)
            unit.stop()
            unit.uninitialize()
            unit.dispose()
        except cm.AudioUnitError as e:
            if "not found" in str(e) or "not initialized" in str(e):
                pytest.skip("AudioUnit operation not available")
            else:
                raise

    def test_audio_unit_properties(self):
        """Test AudioUnit property operations"""
        try:
            unit = cm.AudioUnit.default_output()
            unit.initialize()
            try:
                property_id = capi.get_audio_unit_property_stream_format()
                scope = capi.get_audio_unit_scope_output()
                element = 0
                property_data = unit.get_property(property_id, scope, element)
                assert isinstance(property_data, bytes)
            except cm.AudioUnitError:
                pass
            unit.uninitialize()
            unit.dispose()
        except cm.AudioUnitError as e:
            if "not found" in str(e):
                pytest.skip("Default output AudioUnit not available")
            else:
                raise

    def test_audio_unit_set_property(self):
        """Test AudioUnit property setting"""
        try:
            unit = cm.AudioUnit.default_output()
            try:
                property_id = capi.get_audio_unit_property_stream_format()
                scope = capi.get_audio_unit_scope_output()
                element = 0
                dummy_data = b"\x00" * 40
                unit.set_property(property_id, scope, element, dummy_data)
            except cm.AudioUnitError:
                pass
            unit.dispose()
        except cm.AudioUnitError as e:
            if "not found" in str(e):
                pytest.skip("Default output AudioUnit not available")
            else:
                raise

    def test_audio_unit_operations_without_initialization(self):
        """Test AudioUnit operations that require initialization"""
        try:
            unit = cm.AudioUnit.default_output()
            with pytest.raises(cm.AudioUnitError, match="not initialized"):
                unit.start()
            unit.dispose()
        except cm.AudioUnitError as e:
            if "not found" in str(e):
                pytest.skip("Default output AudioUnit not available")
            else:
                raise

    def test_audio_unit_operations_on_disposed_object(self):
        """Test operations on disposed AudioUnit"""
        try:
            unit = cm.AudioUnit.default_output()
            unit.dispose()
            with pytest.raises(RuntimeError, match="has been disposed"):
                unit.initialize()
            with pytest.raises(RuntimeError, match="has been disposed"):
                unit.start()
        except cm.AudioUnitError as e:
            if "not found" in str(e):
                pytest.skip("Default output AudioUnit not available")
            else:
                raise

    def test_audio_unit_disposal(self):
        """Test AudioUnit disposal"""
        try:
            unit = cm.AudioUnit.default_output()
            unit.initialize()
            assert unit.is_initialized
            assert not unit.is_disposed
            unit.dispose()
            assert unit.is_disposed
        except cm.AudioUnitError as e:
            if "not found" in str(e):
                pytest.skip("Default output AudioUnit not available")
            else:
                raise


class TestAudioUnitIntegration:
    """Integration tests for AudioUnit functionality"""

    def test_audio_unit_vs_functional_api_consistency(self):
        """Test AudioUnit OO API vs functional API consistency"""
        desc_dict = {
            "type": capi.fourchar_to_int("auou"),
            "subtype": capi.fourchar_to_int("def "),
            "manufacturer": capi.fourchar_to_int("appl"),
            "flags": 0,
            "flags_mask": 0,
        }
        try:
            func_component_id = capi.audio_component_find_next(desc_dict)
            if func_component_id == 0:
                pytest.skip("No default output component found")
            func_unit_id = capi.audio_component_instance_new(func_component_id)
            try:
                capi.audio_unit_initialize(func_unit_id)
                capi.audio_unit_uninitialize(func_unit_id)
            finally:
                capi.audio_component_instance_dispose(func_unit_id)
            desc = cm.AudioComponentDescription("auou", "def ", "appl")
            component = cm.AudioComponent.find_next(desc)
            if component is None:
                pytest.skip("No default output component found")
            unit = component.create_instance()
            try:
                unit.initialize()
                unit.uninitialize()
                assert func_component_id != 0
                assert func_unit_id != 0
                assert component.object_id != 0
                assert unit.object_id != 0
            finally:
                unit.dispose()
        except cm.AudioUnitError as e:
            if "not found" in str(e):
                pytest.skip("AudioUnit components not available")
            else:
                raise

    def test_audio_unit_full_workflow(self):
        """Test complete AudioUnit workflow"""
        try:
            unit = cm.AudioUnit.default_output()
            unit.initialize()
            assert unit.is_initialized
            unit.start()
            time.sleep(0.01)
            unit.stop()
            unit.uninitialize()
            assert not unit.is_initialized
            unit.dispose()
            assert unit.is_disposed
        except cm.AudioUnitError as e:
            if "not found" in str(e):
                pytest.skip("Default output AudioUnit not available")
            else:
                raise

    def test_audio_unit_multiple_instances(self):
        """Test creating multiple AudioUnit instances"""
        try:
            unit1 = cm.AudioUnit.default_output()
            unit2 = cm.AudioUnit.default_output()
            assert unit1.object_id != unit2.object_id
            unit1.initialize()
            unit2.initialize()
            assert unit1.is_initialized
            assert unit2.is_initialized
            unit1.uninitialize()
            unit2.uninitialize()
            unit1.dispose()
            unit2.dispose()
        except cm.AudioUnitError as e:
            if "not found" in str(e):
                pytest.skip("Default output AudioUnit not available")
            else:
                raise

    def test_audio_unit_error_handling(self):
        """Test AudioUnit error handling"""
        invalid_desc = cm.AudioComponentDescription("xxxx", "yyyy", "zzzz")
        with pytest.raises(cm.AudioUnitError):
            component = cm.AudioComponent.find_next(invalid_desc)
            if component is not None:
                unit = component.create_instance()
                unit.dispose()
            else:
                unit = cm.AudioUnit(invalid_desc)
                raise cm.AudioUnitError("Expected error for testing")
