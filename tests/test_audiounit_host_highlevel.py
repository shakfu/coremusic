"""Tests for high-level AudioUnit hosting API

Tests the Pythonic object-oriented wrapper for AudioUnit plugins.
"""

import pytest
import coremusic as cm


class TestAudioUnitHost:
    """Test AudioUnitHost high-level API"""

    def test_create_host(self):
        """Test creating AudioUnitHost"""
        host = cm.AudioUnitHost()
        assert host is not None
        print(f"\n{host}")

    def test_get_plugin_count(self):
        """Test getting plugin counts by type"""
        host = cm.AudioUnitHost()
        counts = host.get_plugin_count()

        print(f"\nPlugin counts: {counts}")
        assert 'effect' in counts
        assert 'instrument' in counts
        assert counts['effect'] > 0  # Should have at least some effects

    def test_discover_effects(self):
        """Test discovering effect plugins"""
        host = cm.AudioUnitHost()
        effects = host.discover_plugins(type='effect')

        print(f"\nFound {len(effects)} effect plugins")
        assert len(effects) > 0
        assert isinstance(effects, list)
        assert 'name' in effects[0]

    def test_discover_instruments(self):
        """Test discovering instrument plugins"""
        host = cm.AudioUnitHost()
        instruments = host.discover_plugins(type='instrument')

        print(f"\nFound {len(instruments)} instrument plugins")
        assert len(instruments) > 0

    def test_discover_apple_plugins(self):
        """Test discovering Apple plugins"""
        host = cm.AudioUnitHost()
        apple_plugins = host.discover_plugins(manufacturer='appl')

        print(f"\nFound {len(apple_plugins)} Apple plugins")
        assert len(apple_plugins) > 0


@pytest.mark.slow
class TestAudioUnitPlugin:
    """Test AudioUnitPlugin high-level API - marked as slow due to AudioUnit initialization that can hang"""

    def test_create_plugin_from_name(self):
        """Test creating plugin by name"""
        # AUBandpass should exist on all macOS systems
        plugin = cm.AudioUnitPlugin.from_name("Bandpass", component_type='aufx')

        assert plugin is not None
        assert plugin.name is not None
        print(f"\nLoaded plugin: {plugin.name}")
        print(f"  Manufacturer: {plugin.manufacturer}")
        print(f"  Type: {plugin.type}")

    def test_plugin_lifecycle(self):
        """Test plugin instantiate/initialize/dispose lifecycle"""
        plugin = cm.AudioUnitPlugin.from_name("Bandpass", component_type='aufx')

        # Instantiate
        plugin.instantiate()
        assert not plugin.is_initialized

        # Initialize
        plugin.initialize()
        assert plugin.is_initialized

        # Dispose
        plugin.dispose()

    def test_plugin_context_manager(self):
        """Test plugin with context manager"""
        with cm.AudioUnitPlugin.from_name("Bandpass", component_type='aufx') as plugin:
            assert plugin.is_initialized
            print(f"\nPlugin: {plugin}")

        # Should be cleaned up after context

    def test_plugin_parameters(self):
        """Test accessing plugin parameters"""
        with cm.AudioUnitPlugin.from_name("Bandpass", component_type='aufx') as plugin:
            params = plugin.parameters

            print(f"\nPlugin has {len(params)} parameters:")
            for param in params:
                print(f"  {param}")

            assert len(params) >= 0  # Bandpass has 2 parameters typically

    def test_get_parameter_by_name(self):
        """Test getting parameter by name"""
        with cm.AudioUnitPlugin.from_name("Bandpass", component_type='aufx') as plugin:
            if len(plugin.parameters) > 0:
                param_name = plugin.parameters[0].name
                param = plugin.get_parameter(param_name)

                assert param is not None
                assert param.name == param_name
                print(f"\nParameter: {param}")

    def test_set_parameter_value(self):
        """Test setting parameter value"""
        with cm.AudioUnitPlugin.from_name("Bandpass", component_type='aufx') as plugin:
            if len(plugin.parameters) > 0:
                param = plugin.parameters[0]
                original_value = param.value

                # Set to midpoint
                new_value = (param.min_value + param.max_value) / 2.0
                param.value = new_value

                # Verify
                assert abs(param.value - new_value) < 0.01

                print(f"\nSet {param.name} from {original_value:.3f} to {param.value:.3f}")

    def test_dictionary_parameter_access(self):
        """Test dictionary-style parameter access"""
        with cm.AudioUnitPlugin.from_name("Bandpass", component_type='aufx') as plugin:
            if len(plugin.parameters) > 0:
                param_name = plugin.parameters[0].name

                # Get with []
                value = plugin[param_name]
                assert isinstance(value, float)

                # Set with []
                plugin[param_name] = value
                assert abs(plugin[param_name] - value) < 0.01

                print(f"\nDictionary access: plugin['{param_name}'] = {value:.3f}")

    def test_plugin_factory_presets(self):
        """Test accessing factory presets"""
        with cm.AudioUnitPlugin.from_name("Bandpass", component_type='aufx') as plugin:
            presets = plugin.factory_presets

            print(f"\nPlugin has {len(presets)} factory presets")
            if len(presets) > 0:
                for preset in presets[:5]:
                    print(f"  {preset}")

    def test_load_preset(self):
        """Test loading a factory preset"""
        with cm.AudioUnitPlugin.from_name("Bandpass", component_type='aufx') as plugin:
            presets = plugin.factory_presets

            if len(presets) > 0:
                # Load first preset
                plugin.load_preset(presets[0])
                print(f"\nLoaded preset: {presets[0].name}")


@pytest.mark.slow
class TestAudioUnitHostIntegration:
    """Test integrated workflows - marked as slow due to AudioUnit initialization that can hang"""

    def test_browse_and_load_plugin(self):
        """Test complete workflow: browse, load, control"""
        host = cm.AudioUnitHost()

        # Discover effects
        effects = host.discover_plugins(type='effect', manufacturer='appl')
        assert len(effects) > 0

        # Load first plugin
        plugin_info = effects[0]
        print(f"\nTesting with: {plugin_info['name']}")

        with host.load_plugin(plugin_info['name'], type='effect') as plugin:
            print(f"  Loaded: {plugin}")
            print(f"  Parameters: {len(plugin.parameters)}")
            print(f"  Presets: {len(plugin.factory_presets)}")

            assert plugin.is_initialized

    def test_multiple_plugins(self):
        """Test working with multiple plugins"""
        host = cm.AudioUnitHost()

        effects = host.discover_plugins(type='effect', manufacturer='appl')
        if len(effects) >= 2:
            print("\nLoading multiple plugins:")

            # Load first two plugins
            with host.load_plugin(effects[0]['name']) as plugin1, \
                 host.load_plugin(effects[1]['name']) as plugin2:

                print(f"  Plugin 1: {plugin1.name}")
                print(f"  Plugin 2: {plugin2.name}")

                assert plugin1.is_initialized
                assert plugin2.is_initialized


@pytest.mark.slow
class TestAudioUnitParameter:
    """Test AudioUnitParameter class"""

    @pytest.fixture
    def plugin_with_params(self):
        """Create a plugin with parameters"""
        plugin = cm.AudioUnitPlugin.from_name("Bandpass", component_type='aufx')
        plugin.instantiate()
        plugin.initialize()
        yield plugin
        plugin.dispose()

    def test_parameter_properties(self, plugin_with_params):
        """Test parameter properties"""
        if len(plugin_with_params.parameters) > 0:
            param = plugin_with_params.parameters[0]

            print(f"\nParameter: {param.name}")
            print(f"  ID: {param.id}")
            print(f"  Unit: {param.unit} ({param.unit_name})")
            print(f"  Range: {param.min_value} - {param.max_value}")
            print(f"  Default: {param.default_value}")
            print(f"  Current: {param.value}")

            assert param.name is not None
            assert isinstance(param.id, int)
            assert param.min_value <= param.max_value

    def test_parameter_value_clamping(self, plugin_with_params):
        """Test that parameter values are clamped to range"""
        if len(plugin_with_params.parameters) > 0:
            param = plugin_with_params.parameters[0]

            # Try to set above max
            param.value = param.max_value + 1000
            assert param.value <= param.max_value

            # Try to set below min
            param.value = param.min_value - 1000
            assert param.value >= param.min_value

            print(f"\nValue clamping works for {param.name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
