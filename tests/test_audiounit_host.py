"""Tests for AudioUnit Host functionality

Tests plugin discovery, instantiation, parameter control, and preset management.
"""

import pytest
import coremusic.capi as capi


class TestAudioUnitPluginDiscovery:
    """Test AudioUnit plugin discovery"""

    def test_find_all_effect_plugins(self):
        """Test finding all effect AudioUnits"""
        # Find all effect plugins
        components = capi.audio_unit_find_all_components(component_type='aufx')

        print(f"\nFound {len(components)} effect AudioUnits")
        assert isinstance(components, list)
        # Should have at least some Apple effects on macOS
        assert len(components) > 0

    def test_find_all_instrument_plugins(self):
        """Test finding all instrument AudioUnits"""
        # Find all instrument plugins
        components = capi.audio_unit_find_all_components(component_type='aumu')

        print(f"\nFound {len(components)} instrument AudioUnits")
        assert isinstance(components, list)
        # Should have at least DLSMusicDevice on macOS
        assert len(components) > 0

    def test_find_apple_plugins(self):
        """Test finding Apple AudioUnits"""
        # Find Apple plugins only
        components = capi.audio_unit_find_all_components(manufacturer='appl')

        print(f"\nFound {len(components)} Apple AudioUnits")
        assert isinstance(components, list)
        assert len(components) > 0

    def test_get_component_info(self):
        """Test getting component information"""
        # Find any effect
        components = capi.audio_unit_find_all_components(component_type='aufx')
        assert len(components) > 0

        # Get info for first component
        info = capi.audio_unit_get_component_info(components[0])

        print(f"\nFirst effect plugin info:")
        print(f"  Name: {info['name']}")
        print(f"  Type: {info['type']}")
        print(f"  Subtype: {info['subtype']}")
        print(f"  Manufacturer: {info['manufacturer']}")
        print(f"  Version: {info['version']}")

        assert 'name' in info
        assert 'type' in info
        assert 'subtype' in info
        assert 'manufacturer' in info
        assert info['type'] == 'aufx'

    def test_list_all_plugins_by_category(self):
        """Test listing plugins by category"""
        categories = {
            'Output': 'auou',
            'Effect': 'aufx',
            'Instrument': 'aumu',
            'Generator': 'augn',
            'Mixer': 'aumx',
        }

        print("\nAll AudioUnit Plugins by Category:")
        print("=" * 60)

        for category_name, category_type in categories.items():
            components = capi.audio_unit_find_all_components(component_type=category_type)
            print(f"\n{category_name} ({len(components)}):")

            for i, comp_id in enumerate(components[:5]):  # Show first 5
                try:
                    info = capi.audio_unit_get_component_info(comp_id)
                    print(f"  {i+1}. {info['name']} ({info['manufacturer']})")
                except Exception as e:
                    print(f"  {i+1}. Error getting info: {e}")

            if len(components) > 5:
                print(f"  ... and {len(components) - 5} more")

        # Should have at least some plugins
        total = sum(len(capi.audio_unit_find_all_components(component_type=t))
                   for t in categories.values())
        assert total > 0


class TestAudioUnitParameterDiscovery:
    """Test AudioUnit parameter discovery and control"""

    @pytest.fixture
    def audio_unit(self):
        """Create an AudioUnit instance for testing"""
        # Find Apple's AUDelay effect (should exist on all macOS systems)
        components = capi.audio_unit_find_all_components(
            component_type='aufx',
            manufacturer='appl'
        )

        assert len(components) > 0, "No Apple effects found"

        # Create instance
        component_id = components[0]
        unit_id = capi.audio_component_instance_new(component_id)

        # Initialize
        capi.audio_unit_initialize(unit_id)

        yield unit_id

        # Cleanup
        capi.audio_unit_uninitialize(unit_id)
        capi.audio_component_instance_dispose(unit_id)

    def test_get_parameter_list(self, audio_unit):
        """Test getting list of parameters"""
        params = capi.audio_unit_get_parameter_list(audio_unit)

        print(f"\nFound {len(params)} parameters")
        assert isinstance(params, list)
        # Most AudioUnits have at least some parameters
        # (Some might have 0 if they're simple pass-through)

    def test_get_parameter_info(self, audio_unit):
        """Test getting parameter information"""
        params = capi.audio_unit_get_parameter_list(audio_unit)

        if len(params) == 0:
            pytest.skip("AudioUnit has no parameters")

        # Get info for first parameter
        param_id = params[0]
        info = capi.audio_unit_get_parameter_info(audio_unit, param_id)

        print(f"\nParameter info:")
        print(f"  Name: {info['name']}")
        print(f"  Unit: {info['unit']} ({info['unit_name']})")
        print(f"  Range: {info['min_value']} to {info['max_value']}")
        print(f"  Default: {info['default_value']}")

        assert 'param_id' in info
        assert 'name' in info
        assert 'min_value' in info
        assert 'max_value' in info
        assert info['param_id'] == param_id

    def test_get_set_parameter(self, audio_unit):
        """Test getting and setting parameter values"""
        params = capi.audio_unit_get_parameter_list(audio_unit)

        if len(params) == 0:
            pytest.skip("AudioUnit has no parameters")

        param_id = params[0]
        info = capi.audio_unit_get_parameter_info(audio_unit, param_id)

        # Get current value
        original_value = capi.audio_unit_get_parameter(audio_unit, param_id)
        print(f"\nOriginal value: {original_value}")

        # Set to midpoint
        midpoint = (info['min_value'] + info['max_value']) / 2.0
        capi.audio_unit_set_parameter(audio_unit, param_id, midpoint)

        # Verify it changed
        new_value = capi.audio_unit_get_parameter(audio_unit, param_id)
        print(f"New value: {new_value}")
        assert abs(new_value - midpoint) < 0.01

        # Restore original
        capi.audio_unit_set_parameter(audio_unit, param_id, original_value)

    def test_list_all_parameters(self, audio_unit):
        """Test listing all parameters with details"""
        params = capi.audio_unit_get_parameter_list(audio_unit)

        print(f"\nAll Parameters ({len(params)}):")
        print("=" * 60)

        for param_id in params:
            try:
                info = capi.audio_unit_get_parameter_info(audio_unit, param_id)
                value = capi.audio_unit_get_parameter(audio_unit, param_id)

                print(f"\n{info['name']}:")
                print(f"  ID: {param_id}")
                print(f"  Value: {value:.3f}")
                print(f"  Range: {info['min_value']:.3f} - {info['max_value']:.3f}")
                print(f"  Default: {info['default_value']:.3f}")
                if info['unit_name']:
                    print(f"  Unit: {info['unit_name']}")
            except Exception as e:
                print(f"\nParameter {param_id}: Error - {e}")


class TestAudioUnitPresets:
    """Test AudioUnit preset functionality"""

    @pytest.fixture
    def audio_unit(self):
        """Create an AudioUnit instance for testing"""
        # Find an Apple effect
        components = capi.audio_unit_find_all_components(
            component_type='aufx',
            manufacturer='appl'
        )

        assert len(components) > 0

        component_id = components[0]
        unit_id = capi.audio_component_instance_new(component_id)
        capi.audio_unit_initialize(unit_id)

        yield unit_id

        capi.audio_unit_uninitialize(unit_id)
        capi.audio_component_instance_dispose(unit_id)

    def test_get_factory_presets(self, audio_unit):
        """Test getting factory presets"""
        presets = capi.audio_unit_get_factory_presets(audio_unit)

        print(f"\nFound {len(presets)} factory presets")

        if len(presets) > 0:
            print("\nFactory Presets:")
            for preset in presets[:10]:  # Show first 10
                print(f"  [{preset['number']}] {preset['name']}")

            assert isinstance(presets, list)
            assert 'number' in presets[0]
            assert 'name' in presets[0]

    def test_set_current_preset(self, audio_unit):
        """Test setting current preset"""
        presets = capi.audio_unit_get_factory_presets(audio_unit)

        if len(presets) == 0:
            pytest.skip("AudioUnit has no factory presets")

        # Set first preset
        preset_number = presets[0]['number']
        capi.audio_unit_set_current_preset(audio_unit, preset_number)

        print(f"\nSet preset to: {presets[0]['name']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
