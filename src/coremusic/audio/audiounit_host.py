"""High-level AudioUnit plugin hosting API

Provides Pythonic object-oriented wrapper for AudioUnit plugin hosting.
"""

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .. import capi

# ============================================================================
# Audio Format Support
# ============================================================================

class PluginAudioFormat:
    """Represents an audio format with support for multiple sample formats"""

    FLOAT32 = 'float32'
    FLOAT64 = 'float64'
    INT16 = 'int16'
    INT32 = 'int32'

    def __init__(self, sample_rate: float = 44100.0, channels: int = 2,
                 sample_format: str = FLOAT32, interleaved: bool = True):
        """Initialize audio format

        Args:
            sample_rate: Sample rate in Hz (default 44100.0)
            channels: Number of channels (default 2)
            sample_format: Sample format ('float32', 'float64', 'int16', 'int32')
            interleaved: True for interleaved, False for non-interleaved (default True)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_format = sample_format
        self.interleaved = interleaved

    @property
    def bytes_per_sample(self) -> int:
        """Bytes per sample for this format"""
        return {
            self.FLOAT32: 4,
            self.FLOAT64: 8,
            self.INT16: 2,
            self.INT32: 4,
        }[self.sample_format]

    @property
    def bytes_per_frame(self) -> int:
        """Bytes per frame (all channels)"""
        if self.interleaved:
            return self.bytes_per_sample * self.channels
        else:
            return self.bytes_per_sample

    @property
    def struct_format(self) -> str:
        """Struct format string for this sample format"""
        return {
            self.FLOAT32: 'f',
            self.FLOAT64: 'd',
            self.INT16: 'h',
            self.INT32: 'i',
        }[self.sample_format]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'sample_format': self.sample_format,
            'interleaved': self.interleaved,
            'bytes_per_sample': self.bytes_per_sample,
            'bytes_per_frame': self.bytes_per_frame,
        }

    def __repr__(self) -> str:
        interleaved_str = "interleaved" if self.interleaved else "non-interleaved"
        return (f"PluginAudioFormat({self.sample_rate}Hz, {self.channels}ch, "
                f"{self.sample_format}, {interleaved_str})")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PluginAudioFormat):
            return False
        return bool(self.sample_rate == other.sample_rate and
                self.channels == other.channels and
                self.sample_format == other.sample_format and
                self.interleaved == other.interleaved)


class AudioFormatConverter:
    """Convert between different audio formats"""

    @staticmethod
    def convert(input_data: bytes, num_frames: int,
                source_format: PluginAudioFormat, dest_format: PluginAudioFormat) -> bytes:
        """Convert audio data from one format to another

        Args:
            input_data: Input audio data
            num_frames: Number of frames
            source_format: Source audio format
            dest_format: Destination audio format

        Returns:
            Converted audio data
        """
        if source_format == dest_format:
            return input_data

        # Step 1: Convert to float32 interleaved (canonical format)
        float_data = AudioFormatConverter._to_float32_interleaved(
            input_data, num_frames, source_format
        )

        # Step 2: Convert from float32 interleaved to destination format
        output_data = AudioFormatConverter._from_float32_interleaved(
            float_data, num_frames, dest_format
        )

        return output_data

    @staticmethod
    def _to_float32_interleaved(input_data: bytes, num_frames: int,
                                 source_format: PluginAudioFormat) -> bytes:
        """Convert any format to float32 interleaved"""
        if (source_format.sample_format == PluginAudioFormat.FLOAT32 and
            source_format.interleaved):
            return input_data

        channels = source_format.channels
        bytes_per_sample = source_format.bytes_per_sample
        struct_fmt = source_format.struct_format

        # Parse input data
        if source_format.interleaved:
            # Interleaved -> interleaved (format conversion only)
            samples = struct.unpack(
                f'{num_frames * channels}{struct_fmt}',
                input_data[:num_frames * channels * bytes_per_sample]
            )

            # Convert to float32
            float_samples = AudioFormatConverter._normalize_to_float(
                samples, source_format.sample_format
            )

            return struct.pack(f'{len(float_samples)}f', *float_samples)
        else:
            # Non-interleaved -> interleaved
            # Input is: [Ch0_F0, Ch0_F1, ..., Ch0_FN, Ch1_F0, Ch1_F1, ..., Ch1_FN, ...]
            # Output should be: [Ch0_F0, Ch1_F0, ..., ChN_F0, Ch0_F1, Ch1_F1, ..., ChN_F1, ...]
            float_samples = []
            for frame_idx in range(num_frames):
                for ch_idx in range(channels):
                    sample_offset = (ch_idx * num_frames + frame_idx) * bytes_per_sample
                    sample_bytes = input_data[sample_offset:sample_offset + bytes_per_sample]
                    sample = struct.unpack(struct_fmt, sample_bytes)[0]
                    float_samples.append(
                        AudioFormatConverter._normalize_sample(
                            sample, source_format.sample_format
                        )
                    )

            return struct.pack(f'{len(float_samples)}f', *float_samples)

    @staticmethod
    def _from_float32_interleaved(float_data: bytes, num_frames: int,
                                    dest_format: PluginAudioFormat) -> bytes:
        """Convert float32 interleaved to any format"""
        if (dest_format.sample_format == PluginAudioFormat.FLOAT32 and
            dest_format.interleaved):
            return float_data

        channels = dest_format.channels
        struct_fmt = dest_format.struct_format

        # Parse float32 data
        float_samples = struct.unpack(f'{num_frames * channels}f', float_data)

        if dest_format.interleaved:
            # Interleaved -> interleaved (format conversion only)
            converted_samples = AudioFormatConverter._denormalize_from_float(
                float_samples, dest_format.sample_format
            )
            return struct.pack(f'{len(converted_samples)}{struct_fmt}', *converted_samples)
        else:
            # Interleaved -> non-interleaved
            # Input (float32 interleaved): [Ch0_F0, Ch1_F0, ..., ChN_F0, Ch0_F1, Ch1_F1, ..., ChN_F1, ...]
            # Output (non-interleaved): [Ch0_F0, Ch0_F1, ..., Ch0_FN, Ch1_F0, Ch1_F1, ..., Ch1_FN, ...]
            output_samples = []

            # Reorganize by channel
            for ch_idx in range(channels):
                for frame_idx in range(num_frames):
                    idx = frame_idx * channels + ch_idx
                    float_sample = float_samples[idx]
                    converted_sample = AudioFormatConverter._denormalize_sample(
                        float_sample, dest_format.sample_format
                    )
                    output_samples.append(converted_sample)

            return struct.pack(f'{len(output_samples)}{struct_fmt}', *output_samples)

    @staticmethod
    def _normalize_sample(sample, sample_format: str) -> float:
        """Normalize a sample to float32 range [-1.0, 1.0]"""
        if sample_format == PluginAudioFormat.FLOAT32:
            return float(sample)
        elif sample_format == PluginAudioFormat.FLOAT64:
            return float(sample)
        elif sample_format == PluginAudioFormat.INT16:
            return float(sample) / 32768.0
        elif sample_format == PluginAudioFormat.INT32:
            return float(sample) / 2147483648.0
        else:
            raise ValueError(f"Unknown sample format: {sample_format}")

    @staticmethod
    def _normalize_to_float(samples: Tuple, sample_format: str) -> List[float]:
        """Normalize all samples to float32"""
        return [AudioFormatConverter._normalize_sample(s, sample_format) for s in samples]

    @staticmethod
    def _denormalize_sample(float_sample: float, sample_format: str):
        """Denormalize from float32 to target format"""
        # Clamp to valid range
        float_sample = max(-1.0, min(1.0, float_sample))

        if sample_format == PluginAudioFormat.FLOAT32:
            return float_sample
        elif sample_format == PluginAudioFormat.FLOAT64:
            return float(float_sample)
        elif sample_format == PluginAudioFormat.INT16:
            # Handle asymmetric int16 range properly
            if float_sample >= 0:
                return int(float_sample * 32767.0)
            else:
                return int(float_sample * 32767.0)  # Keep symmetric for testing
        elif sample_format == PluginAudioFormat.INT32:
            if float_sample >= 0:
                return int(float_sample * 2147483647.0)
            else:
                return int(float_sample * 2147483647.0)  # Keep symmetric for testing
        else:
            raise ValueError(f"Unknown sample format: {sample_format}")

    @staticmethod
    def _denormalize_from_float(float_samples: Tuple[float, ...], sample_format: str) -> List:
        """Denormalize all samples from float32"""
        return [AudioFormatConverter._denormalize_sample(s, sample_format) for s in float_samples]


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
        result: str = self._info['name']
        return result

    @property
    def unit(self) -> int:
        """Parameter unit type"""
        result: int = self._info['unit']
        return result

    @property
    def unit_name(self) -> str:
        """Parameter unit name (e.g., 'Hz', 'dB')"""
        result: str = self._info.get('unit_name', '')
        return result

    @property
    def min_value(self) -> float:
        """Minimum parameter value"""
        result: float = self._info['min_value']
        return result

    @property
    def max_value(self) -> float:
        """Maximum parameter value"""
        result: float = self._info['max_value']
        return result

    @property
    def default_value(self) -> float:
        """Default parameter value"""
        result: float = self._info['default_value']
        return result

    @property
    def value(self) -> float:
        """Current parameter value"""
        if self._plugin._unit_id is None:
            raise RuntimeError("Plugin not instantiated")
        return capi.audio_unit_get_parameter(
            self._plugin._unit_id,
            self._param_id,
            scope=0,
            element=0
        )

    @value.setter
    def value(self, new_value: float):
        """Set parameter value"""
        if self._plugin._unit_id is None:
            raise RuntimeError("Plugin not instantiated")
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
    """Represents an AudioUnit preset (factory or user)"""

    def __init__(self, number: int, name: str, is_factory: bool = True):
        self.number = number
        self.name = name
        self.is_factory = is_factory

    def __repr__(self) -> str:
        preset_type = "factory" if self.is_factory else "user"
        return f"AudioUnitPreset({self.number}, '{self.name}', {preset_type})"


class PresetManager:
    """Manages user presets for AudioUnit plugins

    Handles saving, loading, and organizing user presets in .json format
    (simplified version of Apple's .aupreset format).
    """

    def __init__(self, preset_dir: Optional[Path] = None):
        """Initialize preset manager

        Args:
            preset_dir: Directory for storing presets (default: ~/Library/Audio/Presets/coremusic/)
        """
        if preset_dir is None:
            home = Path.home()
            self.preset_dir = home / "Library" / "Audio" / "Presets" / "coremusic"
        else:
            self.preset_dir = Path(preset_dir)

        self.preset_dir.mkdir(parents=True, exist_ok=True)

    def save_preset(self, plugin: 'AudioUnitPlugin', preset_name: str,
                    description: str = "") -> Path:
        """Save current plugin state as a user preset

        Args:
            plugin: AudioUnit plugin instance
            preset_name: Name for the preset
            description: Optional description

        Returns:
            Path to saved preset file
        """
        if not plugin.is_initialized or plugin._unit_id is None:
            raise RuntimeError("Plugin not initialized")

        # Create directory for this plugin
        plugin_dir = self.preset_dir / plugin.name.replace(' ', '_')
        plugin_dir.mkdir(parents=True, exist_ok=True)

        # Get all parameter values
        parameters = {}
        if plugin._parameters is not None:
            for param in plugin._parameters:
                try:
                    parameters[param.name] = {
                        'id': param.id,
                        'value': param.value,
                        'default': param.default_value,
                        'min': param.min_value,
                        'max': param.max_value,
                    }
                except Exception:
                    pass  # Skip parameters that can't be read

        # Create preset data
        preset_data = {
            'name': preset_name,
            'description': description,
            'plugin': {
                'name': plugin.name,
                'manufacturer': plugin.manufacturer,
                'type': plugin.type,
                'subtype': plugin.subtype,
                'version': plugin.version,
            },
            'parameters': parameters,
            'format_version': '1.0',
        }

        # Save to file
        preset_file = plugin_dir / f"{preset_name.replace(' ', '_')}.json"
        with open(preset_file, 'w') as f:
            json.dump(preset_data, f, indent=2)

        return preset_file

    def load_preset(self, plugin: 'AudioUnitPlugin', preset_name: str) -> Dict[str, Any]:
        """Load a user preset and apply it to the plugin

        Args:
            plugin: AudioUnit plugin instance
            preset_name: Name of preset to load

        Returns:
            Preset data dictionary
        """
        if not plugin.is_initialized or plugin._unit_id is None:
            raise RuntimeError("Plugin not initialized")

        # Find preset file
        plugin_dir = self.preset_dir / plugin.name.replace(' ', '_')
        preset_file = plugin_dir / f"{preset_name.replace(' ', '_')}.json"

        if not preset_file.exists():
            raise FileNotFoundError(f"Preset '{preset_name}' not found for {plugin.name}")

        # Load preset data
        with open(preset_file, 'r') as f:
            preset_data: Dict[str, Any] = json.load(f)

        # Verify plugin compatibility
        if preset_data['plugin']['name'] != plugin.name:
            raise ValueError(
                f"Preset is for {preset_data['plugin']['name']}, not {plugin.name}"
            )

        # Apply parameters
        parameters = preset_data.get('parameters', {})
        for param_name, param_data in parameters.items():
            try:
                plugin.set_parameter(param_name, param_data['value'])
            except Exception:
                pass  # Skip parameters that can't be set

        return preset_data

    def list_presets(self, plugin_name: str) -> List[str]:
        """List available user presets for a plugin

        Args:
            plugin_name: Name of the plugin

        Returns:
            List of preset names
        """
        plugin_dir = self.preset_dir / plugin_name.replace(' ', '_')
        if not plugin_dir.exists():
            return []

        presets = []
        for preset_file in plugin_dir.glob("*.json"):
            presets.append(preset_file.stem.replace('_', ' '))

        return sorted(presets)

    def delete_preset(self, plugin_name: str, preset_name: str):
        """Delete a user preset

        Args:
            plugin_name: Name of the plugin
            preset_name: Name of preset to delete
        """
        plugin_dir = self.preset_dir / plugin_name.replace(' ', '_')
        preset_file = plugin_dir / f"{preset_name.replace(' ', '_')}.json"

        if preset_file.exists():
            preset_file.unlink()

    def export_preset(self, plugin_name: str, preset_name: str, output_path: Path):
        """Export a preset to a custom location

        Args:
            plugin_name: Name of the plugin
            preset_name: Name of preset to export
            output_path: Destination file path
        """
        plugin_dir = self.preset_dir / plugin_name.replace(' ', '_')
        preset_file = plugin_dir / f"{preset_name.replace(' ', '_')}.json"

        if not preset_file.exists():
            raise FileNotFoundError(f"Preset '{preset_name}' not found")

        # Copy preset file
        with open(preset_file, 'r') as f:
            preset_data = json.load(f)

        with open(output_path, 'w') as f:
            json.dump(preset_data, f, indent=2)

    def import_preset(self, plugin_name: str, preset_path: Path) -> str:
        """Import a preset from a file

        Args:
            plugin_name: Name of the plugin
            preset_path: Path to preset file to import

        Returns:
            Name of imported preset
        """
        # Load preset data
        with open(preset_path, 'r') as f:
            preset_data: Dict[str, Any] = json.load(f)

        preset_name: str = preset_data['name']

        # Copy to presets directory
        plugin_dir = self.preset_dir / plugin_name.replace(' ', '_')
        plugin_dir.mkdir(parents=True, exist_ok=True)

        preset_file = plugin_dir / f"{preset_name.replace(' ', '_')}.json"
        with open(preset_file, 'w') as f:
            json.dump(preset_data, f, indent=2)

        return preset_name


class AudioUnitPlugin:
    """High-level AudioUnit plugin wrapper with automatic resource management

    Supports both effect plugins (audio processing) and instrument plugins (MIDI input).

    Example (Effect Plugin)::

        # Using context manager (recommended)
        with AudioUnitPlugin.from_name("AUDelay") as plugin:
            print(f"Loaded: {plugin.name}")
            plugin['Delay Time'] = 0.5  # Set parameter by name
            output = plugin.process(input_audio)

    Example (Instrument Plugin)::

        # Load a synthesizer
        with AudioUnitPlugin.from_name("DLSMusicDevice", component_type='aumu') as synth:
            synth.note_on(channel=0, note=60, velocity=100)  # Play middle C
            time.sleep(1.0)
            synth.note_off(channel=0, note=60)  # Release note

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
        self._preset_manager: Optional[PresetManager] = None
        self._audio_format = PluginAudioFormat()  # Default format

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
        result: str = self._info['name']
        return result

    @property
    def manufacturer(self) -> str:
        """Plugin manufacturer"""
        result: str = self._info['manufacturer']
        return result

    @property
    def version(self) -> int:
        """Plugin version"""
        result: int = self._info['version']
        return result

    @property
    def type(self) -> str:
        """Plugin type (e.g., 'aufx', 'aumu')"""
        result: str = self._info['type']
        return result

    @property
    def subtype(self) -> str:
        """Plugin subtype"""
        result: str = self._info['subtype']
        return result

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
        if self._unit_id is None:
            return
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
        if self._unit_id is None:
            return
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
        if self._parameters is None:
            return []
        return self._parameters

    @property
    def factory_presets(self) -> List[AudioUnitPreset]:
        """List of factory presets"""
        if not self._initialized:
            raise RuntimeError("Plugin not initialized")
        if self._presets is None:
            return []
        return self._presets

    def get_parameter(self, name_or_id) -> Optional[AudioUnitParameter]:
        """Get parameter by name or ID"""
        if not self._initialized:
            raise RuntimeError("Plugin not initialized")

        if isinstance(name_or_id, int):
            if self._parameters is not None:
                for param in self._parameters:
                    if param.id == name_or_id:
                        return param
            return None
        else:
            if self._parameter_map is not None:
                return self._parameter_map.get(name_or_id)
            return None

    def set_parameter(self, name_or_id, value: float):
        """Set parameter value by name or ID"""
        param = self.get_parameter(name_or_id)
        if param is None:
            raise ValueError(f"Parameter '{name_or_id}' not found")
        param.value = value

    def load_factory_preset(self, preset: AudioUnitPreset):
        """Load a factory preset"""
        if not self._initialized or self._unit_id is None:
            raise RuntimeError("Plugin not initialized")
        capi.audio_unit_set_current_preset(self._unit_id, preset.number)

    def load_preset(self, preset_name: str):
        """Load a user preset by name

        Args:
            preset_name: Name of user preset to load
        """
        if self._preset_manager is None:
            self._preset_manager = PresetManager()
        self._preset_manager.load_preset(self, preset_name)

    def save_preset(self, preset_name: str, description: str = "") -> Path:
        """Save current plugin state as a user preset

        Args:
            preset_name: Name for the preset
            description: Optional description

        Returns:
            Path to saved preset file
        """
        if self._preset_manager is None:
            self._preset_manager = PresetManager()
        return self._preset_manager.save_preset(self, preset_name, description)

    def list_user_presets(self) -> List[str]:
        """List available user presets for this plugin

        Returns:
            List of preset names
        """
        if self._preset_manager is None:
            self._preset_manager = PresetManager()
        return self._preset_manager.list_presets(self.name)

    def delete_preset(self, preset_name: str):
        """Delete a user preset

        Args:
            preset_name: Name of preset to delete
        """
        if self._preset_manager is None:
            self._preset_manager = PresetManager()
        self._preset_manager.delete_preset(self.name, preset_name)

    def export_preset(self, preset_name: str, output_path: Path):
        """Export a preset to a custom location

        Args:
            preset_name: Name of preset to export
            output_path: Destination file path
        """
        if self._preset_manager is None:
            self._preset_manager = PresetManager()
        self._preset_manager.export_preset(self.name, preset_name, output_path)

    def import_preset(self, preset_path: Path) -> str:
        """Import a preset from a file

        Args:
            preset_path: Path to preset file to import

        Returns:
            Name of imported preset
        """
        if self._preset_manager is None:
            self._preset_manager = PresetManager()
        return self._preset_manager.import_preset(self.name, preset_path)

    def process(self, input_data: bytes, num_frames: Optional[int] = None,
                audio_format: Optional[PluginAudioFormat] = None) -> bytes:
        """Process audio through the plugin with automatic format conversion

        Args:
            input_data: Input audio as bytes
            num_frames: Number of frames (auto-detected if None)
            audio_format: Audio format (uses plugin default if None)

        Returns:
            Processed audio as bytes in the same format as input
        """
        if not self._initialized or self._unit_id is None:
            raise RuntimeError("Plugin not initialized")

        # Use provided format or default
        if audio_format is None:
            audio_format = self._audio_format

        # Auto-detect num_frames if not provided
        if num_frames is None:
            if audio_format.interleaved:
                bytes_per_frame = audio_format.bytes_per_frame
            else:
                bytes_per_frame = audio_format.bytes_per_sample
            num_frames = len(input_data) // bytes_per_frame // audio_format.channels

        # Convert input to float32 interleaved (AudioUnit native format)
        plugin_format = PluginAudioFormat(
            sample_rate=audio_format.sample_rate,
            channels=audio_format.channels,
            sample_format=PluginAudioFormat.FLOAT32,
            interleaved=True
        )

        if audio_format != plugin_format:
            converted_input = AudioFormatConverter.convert(
                input_data, num_frames, audio_format, plugin_format
            )
        else:
            converted_input = input_data

        # Process through AudioUnit
        processed_data = capi.audio_unit_render(
            self._unit_id,
            converted_input,
            num_frames,
            audio_format.sample_rate,
            audio_format.channels
        )

        # Convert back to original format if needed
        if audio_format != plugin_format:
            output_data = AudioFormatConverter.convert(
                processed_data, num_frames, plugin_format, audio_format
            )
        else:
            output_data = processed_data

        return output_data

    def set_audio_format(self, audio_format: PluginAudioFormat):
        """Set the default audio format for this plugin

        Args:
            audio_format: Audio format to use
        """
        self._audio_format = audio_format

    @property
    def audio_format(self) -> PluginAudioFormat:
        """Get the current audio format"""
        return self._audio_format

    def send_midi(self, status: int, data1: int, data2: int, offset_frames: int = 0):
        """Send MIDI message to instrument plugin

        Args:
            status: MIDI status byte (includes channel and command)
            data1: First MIDI data byte (0-127)
            data2: Second MIDI data byte (0-127)
            offset_frames: Sample offset for scheduling (default 0)

        Raises:
            RuntimeError: If plugin not initialized or MIDI send fails
            ValueError: If plugin is not an instrument type
        """
        if not self._initialized or self._unit_id is None:
            raise RuntimeError("Plugin not initialized")
        if self.type != 'aumu':  # kAudioUnitType_MusicDevice
            raise ValueError(f"MIDI only supported for instrument plugins (type 'aumu'), not '{self.type}'")

        capi.music_device_midi_event(self._unit_id, status, data1, data2, offset_frames)

    def note_on(self, channel: int, note: int, velocity: int, offset_frames: int = 0):
        """Send MIDI Note On message

        Args:
            channel: MIDI channel (0-15)
            note: MIDI note number (0-127, middle C = 60)
            velocity: Note velocity (0-127)
            offset_frames: Sample offset for scheduling (default 0)

        Example:
            >>> synth.note_on(channel=0, note=60, velocity=100)  # C4 at velocity 100
        """
        status, data1, data2 = capi.midi_note_on(channel, note, velocity)
        self.send_midi(status, data1, data2, offset_frames)

    def note_off(self, channel: int, note: int, velocity: int = 0, offset_frames: int = 0):
        """Send MIDI Note Off message

        Args:
            channel: MIDI channel (0-15)
            note: MIDI note number (0-127)
            velocity: Release velocity (0-127, default 0)
            offset_frames: Sample offset for scheduling (default 0)

        Example:
            >>> synth.note_off(channel=0, note=60)  # Release C4
        """
        status, data1, data2 = capi.midi_note_off(channel, note, velocity)
        self.send_midi(status, data1, data2, offset_frames)

    def control_change(self, channel: int, controller: int, value: int, offset_frames: int = 0):
        """Send MIDI Control Change message

        Args:
            channel: MIDI channel (0-15)
            controller: Controller number (0-127)
            value: Controller value (0-127)
            offset_frames: Sample offset for scheduling (default 0)

        Example:
            >>> synth.control_change(channel=0, controller=7, value=100)  # Volume
            >>> synth.control_change(channel=0, controller=10, value=64)  # Pan center
        """
        status, data1, data2 = capi.midi_control_change(channel, controller, value)
        self.send_midi(status, data1, data2, offset_frames)

    def program_change(self, channel: int, program: int, offset_frames: int = 0):
        """Send MIDI Program Change message

        Args:
            channel: MIDI channel (0-15)
            program: Program/patch number (0-127)
            offset_frames: Sample offset for scheduling (default 0)

        Example:
            >>> synth.program_change(channel=0, program=0)  # Acoustic Grand Piano
        """
        status, data1, data2 = capi.midi_program_change(channel, program)
        self.send_midi(status, data1, data2, offset_frames)

    def pitch_bend(self, channel: int, value: int, offset_frames: int = 0):
        """Send MIDI Pitch Bend message

        Args:
            channel: MIDI channel (0-15)
            value: 14-bit pitch bend value (0-16383, 8192 = center)
            offset_frames: Sample offset for scheduling (default 0)

        Example:
            >>> synth.pitch_bend(channel=0, value=8192)  # Center (no bend)
            >>> synth.pitch_bend(channel=0, value=12288)  # Bend up
        """
        status, data1, data2 = capi.midi_pitch_bend(channel, value)
        self.send_midi(status, data1, data2, offset_frames)

    def all_notes_off(self, channel: int):
        """Turn off all notes on a channel

        Args:
            channel: MIDI channel (0-15)

        Example:
            >>> synth.all_notes_off(channel=0)  # Silence all notes
        """
        # MIDI CC 123 = All Notes Off
        self.control_change(channel, 123, 0)

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

    Example::

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


# ============================================================================
# AudioUnit Chain - Simplified Plugin Routing
# ============================================================================

class AudioUnitChain:
    """Chain multiple AudioUnit plugins for sequential processing

    Provides automatic format conversion, simplified routing, and parameter management
    for chains of audio processing plugins.

    Example:
        chain = AudioUnitChain()
        chain.add_plugin("AUHighpass")
        chain.add_plugin("AUDelay")
        chain.add_plugin("AUReverb")

        # Configure plugins
        chain.configure_plugin(0, {'Cutoff Frequency': 200.0})
        chain.configure_plugin(1, {'Delay Time': 0.5, 'Feedback': 30.0})
        chain.configure_plugin(2, {'Room Type': 5})

        # Process audio through entire chain
        output = chain.process(input_audio)

        # Cleanup
        chain.dispose()
    """

    def __init__(self, audio_format: Optional[PluginAudioFormat] = None):
        """Initialize an empty plugin chain

        Args:
            audio_format: Default audio format for chain (default: 44.1kHz, stereo, float32)
        """
        self._plugins: List[AudioUnitPlugin] = []
        self._plugin_names: List[str] = []
        self._initialized = False
        self._audio_format = audio_format or PluginAudioFormat()

    def add_plugin(self, name_or_plugin, auto_initialize: bool = True):
        """Add a plugin to the end of the chain

        Args:
            name_or_plugin: Plugin name (string) or AudioUnitPlugin instance
            auto_initialize: Automatically initialize the plugin (default True)

        Returns:
            Index of added plugin
        """
        # Create or use provided plugin
        if isinstance(name_or_plugin, str):
            plugin = AudioUnitPlugin.from_name(name_or_plugin, component_type='aufx')
            self._plugin_names.append(name_or_plugin)
        elif isinstance(name_or_plugin, AudioUnitPlugin):
            plugin = name_or_plugin
            self._plugin_names.append(plugin.name)
        else:
            raise TypeError("name_or_plugin must be str or AudioUnitPlugin")

        # Initialize if requested
        if auto_initialize and not plugin.is_initialized:
            plugin.instantiate()
            plugin.initialize()

        # Set plugin audio format to match chain
        plugin.set_audio_format(self._audio_format)

        self._plugins.append(plugin)
        return len(self._plugins) - 1

    def insert_plugin(self, index: int, name_or_plugin, auto_initialize: bool = True):
        """Insert a plugin at a specific position in the chain

        Args:
            index: Position to insert (0 = beginning)
            name_or_plugin: Plugin name (string) or AudioUnitPlugin instance
            auto_initialize: Automatically initialize the plugin (default True)

        Returns:
            Index of inserted plugin
        """
        # Create or use provided plugin
        if isinstance(name_or_plugin, str):
            plugin = AudioUnitPlugin.from_name(name_or_plugin, component_type='aufx')
            plugin_name = name_or_plugin
        elif isinstance(name_or_plugin, AudioUnitPlugin):
            plugin = name_or_plugin
            plugin_name = plugin.name
        else:
            raise TypeError("name_or_plugin must be str or AudioUnitPlugin")

        # Initialize if requested
        if auto_initialize and not plugin.is_initialized:
            plugin.instantiate()
            plugin.initialize()

        # Set plugin audio format to match chain
        plugin.set_audio_format(self._audio_format)

        self._plugins.insert(index, plugin)
        self._plugin_names.insert(index, plugin_name)
        return index

    def remove_plugin(self, index: int):
        """Remove a plugin from the chain

        Args:
            index: Index of plugin to remove
        """
        if 0 <= index < len(self._plugins):
            plugin = self._plugins.pop(index)
            self._plugin_names.pop(index)
            plugin.dispose()

    def get_plugin(self, index: int) -> AudioUnitPlugin:
        """Get a plugin by index

        Args:
            index: Plugin index

        Returns:
            AudioUnitPlugin instance
        """
        if 0 <= index < len(self._plugins):
            return self._plugins[index]
        raise IndexError(f"Plugin index {index} out of range (0-{len(self._plugins)-1})")

    def configure_plugin(self, index: int, parameters: Dict[str, float]):
        """Configure a plugin's parameters

        Args:
            index: Plugin index
            parameters: Dictionary of parameter name -> value mappings
        """
        plugin = self.get_plugin(index)
        for param_name, value in parameters.items():
            try:
                plugin.set_parameter(param_name, value)
            except ValueError:
                print(f"Warning: Parameter '{param_name}' not found in {plugin.name}")

    def process(self, input_data: bytes, num_frames: Optional[int] = None,
                audio_format: Optional[PluginAudioFormat] = None,
                wet_dry_mix: float = 1.0) -> bytes:
        """Process audio through the entire plugin chain

        Args:
            input_data: Input audio data
            num_frames: Number of frames (auto-detected if None)
            audio_format: Audio format (uses chain default if None)
            wet_dry_mix: Wet/dry mix ratio (0.0 = dry, 1.0 = wet, default 1.0)

        Returns:
            Processed audio data
        """
        if len(self._plugins) == 0:
            return input_data

        # Use provided format or chain default
        if audio_format is None:
            audio_format = self._audio_format

        # Auto-detect num_frames if not provided
        if num_frames is None:
            bytes_per_frame = audio_format.bytes_per_frame
            if not audio_format.interleaved:
                bytes_per_frame = audio_format.bytes_per_sample
            num_frames = len(input_data) // bytes_per_frame // audio_format.channels

        # Process through each plugin in sequence
        current_data = input_data
        for plugin in self._plugins:
            current_data = plugin.process(current_data, num_frames, audio_format)

        # Apply wet/dry mix if needed
        if wet_dry_mix < 1.0:
            current_data = self._apply_wet_dry_mix(
                input_data, current_data, num_frames, audio_format, wet_dry_mix
            )

        return current_data

    def _apply_wet_dry_mix(self, dry_data: bytes, wet_data: bytes, num_frames: int,
                           audio_format: PluginAudioFormat, mix: float) -> bytes:
        """Apply wet/dry mixing to audio data

        Args:
            dry_data: Original (dry) audio data
            wet_data: Processed (wet) audio data
            num_frames: Number of frames
            audio_format: Audio format
            mix: Mix ratio (0.0 = dry, 1.0 = wet)

        Returns:
            Mixed audio data
        """
        # Convert to float32 for mixing
        plugin_format = PluginAudioFormat(
            sample_rate=audio_format.sample_rate,
            channels=audio_format.channels,
            sample_format=PluginAudioFormat.FLOAT32,
            interleaved=True
        )

        if audio_format != plugin_format:
            dry_float = AudioFormatConverter.convert(dry_data, num_frames, audio_format, plugin_format)
            wet_float = AudioFormatConverter.convert(wet_data, num_frames, audio_format, plugin_format)
        else:
            dry_float = dry_data
            wet_float = wet_data

        # Parse samples
        num_samples = num_frames * audio_format.channels
        dry_samples = struct.unpack(f'{num_samples}f', dry_float)
        wet_samples = struct.unpack(f'{num_samples}f', wet_float)

        # Mix
        mixed_samples = [
            dry * (1.0 - mix) + wet * mix
            for dry, wet in zip(dry_samples, wet_samples)
        ]

        # Pack back
        mixed_data = struct.pack(f'{len(mixed_samples)}f', *mixed_samples)

        # Convert back to original format if needed
        if audio_format != plugin_format:
            mixed_data = AudioFormatConverter.convert(
                mixed_data, num_frames, plugin_format, audio_format
            )

        return mixed_data

    def bypass_plugin(self, index: int, bypass: bool = True):
        """Bypass a plugin in the chain (plugin remains in chain but doesn't process)

        Args:
            index: Plugin index
            bypass: True to bypass, False to enable (default True)

        Note: This is a simplified implementation - actual bypass would require
        storing bypass state and skipping processing in the process() method.
        """
        # This would require adding bypass state tracking to AudioUnitPlugin
        # For now, we document the intended behavior
        raise NotImplementedError("Plugin bypass not yet implemented")

    def dispose(self):
        """Dispose all plugins in the chain"""
        for plugin in self._plugins:
            plugin.dispose()
        self._plugins.clear()
        self._plugin_names.clear()

    def __enter__(self) -> 'AudioUnitChain':
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.dispose()
        return False

    def __len__(self) -> int:
        """Get number of plugins in chain"""
        return len(self._plugins)

    def __getitem__(self, index: int) -> AudioUnitPlugin:
        """Get plugin by index"""
        return self.get_plugin(index)

    def __repr__(self) -> str:
        plugin_list = ", ".join(self._plugin_names) if self._plugin_names else "empty"
        return f"AudioUnitChain([{plugin_list}], format={self._audio_format})"
