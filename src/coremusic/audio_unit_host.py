"""High-level AudioUnit plugin hosting API

Provides Pythonic object-oriented wrapper for AudioUnit plugin hosting.
"""

from typing import List, Dict, Optional, Any
import struct
from . import capi


class AudioUnitParameter:
    """Represents a single AudioUnit parameter with metadata and control"""

    def __init__(self, plugin: 'AudioUnitPlugin', param_id: int, info: Dict[str, Any]):
        self._plugin = plugin
        self._param_id = param_id
        self._info = info

    @property
    def id(self) -> int:
        """Parameter ID"""
        return self._param_id

    @property
    def name(self) -> str:
        """Parameter name"""
        return self._info['name']

    @property
    def unit(self) -> int:
        """Parameter unit type"""
        return self._info['unit']

    @property
    def unit_name(self) -> str:
        """Parameter unit name (e.g., 'Hz', 'dB')"""
        return self._info.get('unit_name', '')

    @property
    def min_value(self) -> float:
        """Minimum parameter value"""
        return self._info['min_value']

    @property
    def max_value(self) -> float:
        """Maximum parameter value"""
        return self._info['max_value']

    @property
    def default_value(self) -> float:
        """Default parameter value"""
        return self._info['default_value']

    @property
    def value(self) -> float:
        """Current parameter value"""
        return capi.audio_unit_get_parameter(
            self._plugin._unit_id,
            self._param_id,
            scope=0,
            element=0
        )

    @value.setter
    def value(self, new_value: float):
        """Set parameter value"""
        # Clamp to valid range
        clamped = max(self.min_value, min(self.max_value, new_value))
        capi.audio_unit_set_parameter(
            self._plugin._unit_id,
            self._param_id,
            clamped,
            scope=0,
            element=0
        )

    def __repr__(self) -> str:
        unit_str = f" {self.unit_name}" if self.unit_name else ""
        return (f"AudioUnitParameter('{self.name}', value={self.value:.3f}{unit_str}, "
                f"range={self.min_value:.3f}-{self.max_value:.3f})")


class AudioUnitPreset:
    """Represents an AudioUnit factory preset"""

    def __init__(self, number: int, name: str):
        self.number = number
        self.name = name

    def __repr__(self) -> str:
        return f"AudioUnitPreset({self.number}, '{self.name}')"


class AudioUnitPlugin:
    """High-level AudioUnit plugin wrapper with automatic resource management

    Example:
        # Using context manager (recommended)
        with AudioUnitPlugin.from_name("AUDelay") as plugin:
            print(f"Loaded: {plugin.name}")
            plugin['Delay Time'] = 0.5  # Set parameter by name
            output = plugin.process(input_audio)

        # Manual lifecycle
        plugin = AudioUnitPlugin.from_name("AUReverb")
        plugin.instantiate()
        plugin.initialize()
        # ... use plugin ...
        plugin.cleanup()
    """

    def __init__(self, component_id: int, component_info: Optional[Dict[str, Any]] = None):
        """Initialize plugin from component ID

        Args:
            component_id: AudioComponent ID from plugin discovery
            component_info: Optional pre-fetched component info
        """
        self._component_id = component_id
        self._unit_id: Optional[int] = None
        self._initialized = False
        self._parameters: Optional[List[AudioUnitParameter]] = None
        self._parameter_map: Optional[Dict[str, AudioUnitParameter]] = None
        self._presets: Optional[List[AudioUnitPreset]] = None

        # Get component info
        if component_info is None:
            self._info = capi.audio_unit_get_component_info(component_id)
        else:
            self._info = component_info

    @classmethod
    def from_name(cls, name: str, component_type: Optional[str] = None) -> 'AudioUnitPlugin':
        """Create plugin by name

        Args:
            name: Plugin name (case-insensitive, partial match)
            component_type: Optional component type filter ('aufx', 'aumu', etc.)

        Returns:
            AudioUnitPlugin instance

        Raises:
            ValueError: If plugin not found
        """
        # Find all components
        components = capi.audio_unit_find_all_components(component_type=component_type)

        # Search for matching name
        name_lower = name.lower()
        for comp_id in components:
            info = capi.audio_unit_get_component_info(comp_id)
            if name_lower in info['name'].lower():
                return cls(comp_id, info)

        raise ValueError(f"Plugin '{name}' not found")

    @classmethod
    def from_component_id(cls, component_id: int) -> 'AudioUnitPlugin':
        """Create plugin from component ID"""
        return cls(component_id)

    @property
    def name(self) -> str:
        """Plugin name"""
        return self._info['name']

    @property
    def manufacturer(self) -> str:
        """Plugin manufacturer"""
        return self._info['manufacturer']

    @property
    def version(self) -> int:
        """Plugin version"""
        return self._info['version']

    @property
    def type(self) -> str:
        """Plugin type (e.g., 'aufx', 'aumu')"""
        return self._info['type']

    @property
    def subtype(self) -> str:
        """Plugin subtype"""
        return self._info['subtype']

    @property
    def is_initialized(self) -> bool:
        """Check if plugin is initialized"""
        return self._initialized

    def instantiate(self) -> 'AudioUnitPlugin':
        """Instantiate the AudioUnit

        Returns:
            self for chaining
        """
        if self._unit_id is not None:
            raise RuntimeError("Plugin already instantiated")

        self._unit_id = capi.audio_component_instance_new(self._component_id)
        return self

    def initialize(self) -> 'AudioUnitPlugin':
        """Initialize the AudioUnit

        Returns:
            self for chaining
        """
        if self._unit_id is None:
            raise RuntimeError("Plugin not instantiated. Call instantiate() first.")

        if self._initialized:
            return self

        capi.audio_unit_initialize(self._unit_id)
        self._initialized = True

        # Discover parameters
        self._discover_parameters()

        # Discover presets
        self._discover_presets()

        return self

    def uninitialize(self) -> 'AudioUnitPlugin':
        """Uninitialize the AudioUnit

        Returns:
            self for chaining
        """
        if self._unit_id is None or not self._initialized:
            return self

        capi.audio_unit_uninitialize(self._unit_id)
        self._initialized = False
        self._parameters = None
        self._parameter_map = None
        return self

    def dispose(self):
        """Dispose the AudioUnit instance"""
        if self._unit_id is not None:
            if self._initialized:
                self.uninitialize()
            capi.audio_component_instance_dispose(self._unit_id)
            self._unit_id = None

    def cleanup(self):
        """Complete cleanup (alias for dispose)"""
        self.dispose()

    def _discover_parameters(self):
        """Discover all parameters"""
        param_ids = capi.audio_unit_get_parameter_list(self._unit_id)

        self._parameters = []
        self._parameter_map = {}

        for param_id in param_ids:
            try:
                info = capi.audio_unit_get_parameter_info(self._unit_id, param_id)
                param = AudioUnitParameter(self, param_id, info)
                self._parameters.append(param)
                self._parameter_map[param.name] = param
            except Exception:
                pass  # Skip parameters we can't access

    def _discover_presets(self):
        """Discover factory presets"""
        preset_list = capi.audio_unit_get_factory_presets(self._unit_id)
        self._presets = [
            AudioUnitPreset(p['number'], p['name'])
            for p in preset_list
        ]

    @property
    def parameters(self) -> List[AudioUnitParameter]:
        """List of all parameters"""
        if not self._initialized:
            raise RuntimeError("Plugin not initialized")
        return self._parameters

    @property
    def factory_presets(self) -> List[AudioUnitPreset]:
        """List of factory presets"""
        if not self._initialized:
            raise RuntimeError("Plugin not initialized")
        return self._presets

    def get_parameter(self, name_or_id) -> Optional[AudioUnitParameter]:
        """Get parameter by name or ID"""
        if not self._initialized:
            raise RuntimeError("Plugin not initialized")

        if isinstance(name_or_id, int):
            for param in self._parameters:
                if param.id == name_or_id:
                    return param
            return None
        else:
            return self._parameter_map.get(name_or_id)

    def set_parameter(self, name_or_id, value: float):
        """Set parameter value by name or ID"""
        param = self.get_parameter(name_or_id)
        if param is None:
            raise ValueError(f"Parameter '{name_or_id}' not found")
        param.value = value

    def load_preset(self, preset: AudioUnitPreset):
        """Load a factory preset"""
        if not self._initialized:
            raise RuntimeError("Plugin not initialized")
        capi.audio_unit_set_current_preset(self._unit_id, preset.number)

    def process(self, input_data: bytes, num_frames: Optional[int] = None,
                sample_rate: float = 44100.0, num_channels: int = 2) -> bytes:
        """Process audio through the plugin

        Args:
            input_data: Input audio as bytes (float32, interleaved)
            num_frames: Number of frames (auto-detected if None)
            sample_rate: Sample rate (default 44100)
            num_channels: Number of channels (default 2)

        Returns:
            Processed audio as bytes (float32, interleaved)
        """
        if not self._initialized:
            raise RuntimeError("Plugin not initialized")

        if num_frames is None:
            bytes_per_frame = num_channels * 4  # float32
            num_frames = len(input_data) // bytes_per_frame

        return capi.audio_unit_render(
            self._unit_id,
            input_data,
            num_frames,
            sample_rate,
            num_channels
        )

    def __getitem__(self, key: str) -> float:
        """Get parameter value by name"""
        param = self.get_parameter(key)
        if param is None:
            raise KeyError(f"Parameter '{key}' not found")
        return param.value

    def __setitem__(self, key: str, value: float):
        """Set parameter value by name"""
        self.set_parameter(key, value)

    def __enter__(self) -> 'AudioUnitPlugin':
        """Context manager entry"""
        self.instantiate()
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
        return False

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"AudioUnitPlugin('{self.name}', {self.manufacturer}, {status})"


class AudioUnitHost:
    """High-level AudioUnit host for plugin discovery and management

    Example:
        host = AudioUnitHost()

        # Discover plugins
        effects = host.discover_plugins(type='effect')
        print(f"Found {len(effects)} effects")

        # Load a plugin
        with host.load_plugin("AUDelay") as delay:
            delay['Delay Time'] = 0.5
            output = delay.process(input_audio)
    """

    TYPE_MAP = {
        'output': 'auou',
        'effect': 'aufx',
        'instrument': 'aumu',
        'generator': 'augn',
        'mixer': 'aumx',
    }

    def __init__(self):
        """Initialize the AudioUnit host"""
        self._plugin_cache: Optional[List[Dict[str, Any]]] = None

    def discover_plugins(self, type: Optional[str] = None,
                        manufacturer: Optional[str] = None,
                        refresh: bool = False) -> List[Dict[str, Any]]:
        """Discover available AudioUnit plugins

        Args:
            type: Plugin type ('effect', 'instrument', 'generator', 'mixer', 'output')
            manufacturer: Manufacturer code filter
            refresh: Force refresh of plugin cache

        Returns:
            List of plugin info dictionaries
        """
        # Map friendly names to fourcc codes
        component_type = None
        if type is not None:
            component_type = self.TYPE_MAP.get(type.lower(), type)

        # Find components
        components = capi.audio_unit_find_all_components(
            component_type=component_type,
            manufacturer=manufacturer
        )

        # Get info for each
        plugins = []
        for comp_id in components:
            try:
                info = capi.audio_unit_get_component_info(comp_id)
                plugins.append(info)
            except Exception:
                pass

        return plugins

    def load_plugin(self, name_or_id, type: Optional[str] = None) -> AudioUnitPlugin:
        """Load a plugin by name or component ID

        Args:
            name_or_id: Plugin name (string) or component ID (int)
            type: Optional type filter for name search

        Returns:
            AudioUnitPlugin instance (not yet initialized)
        """
        if isinstance(name_or_id, int):
            return AudioUnitPlugin.from_component_id(name_or_id)
        else:
            component_type = None
            if type is not None:
                component_type = self.TYPE_MAP.get(type.lower(), type)
            return AudioUnitPlugin.from_name(name_or_id, component_type)

    def get_plugin_count(self) -> Dict[str, int]:
        """Get count of plugins by type

        Returns:
            Dictionary mapping type names to counts
        """
        counts = {}
        for type_name, type_code in self.TYPE_MAP.items():
            components = capi.audio_unit_find_all_components(component_type=type_code)
            counts[type_name] = len(components)
        return counts

    def __repr__(self) -> str:
        counts = self.get_plugin_count()
        total = sum(counts.values())
        return f"AudioUnitHost({total} plugins: {counts})"
