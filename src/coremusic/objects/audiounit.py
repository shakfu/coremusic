"""AudioUnit classes for coremusic.

This module provides classes for working with Audio Units:
- AudioComponentDescription: Describes an audio component
- AudioComponent: Audio component wrapper
- AudioUnit: Audio unit for real-time audio processing
"""

from __future__ import annotations

import struct
from typing import Any, Dict, List, Optional

from .. import capi
from .audio import AudioFormat
from .exceptions import AudioUnitError

__all__ = [
    "AudioComponentDescription",
    "AudioComponent",
    "AudioUnit",
]


class AudioComponentDescription:
    """Pythonic representation of AudioComponent description"""

    def __init__(
        self,
        type: str,
        subtype: str,
        manufacturer: str,
        flags: int = 0,
        flags_mask: int = 0,
    ):
        self.type = type
        self.subtype = subtype
        self.manufacturer = manufacturer
        self.flags = flags
        self.flags_mask = flags_mask

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for functional API"""
        type_int = (
            capi.fourchar_to_int(self.type) if isinstance(self.type, str) else self.type
        )
        subtype_int = (
            capi.fourchar_to_int(self.subtype)
            if isinstance(self.subtype, str)
            else self.subtype
        )
        manufacturer_int = (
            capi.fourchar_to_int(self.manufacturer)
            if isinstance(self.manufacturer, str)
            else self.manufacturer
        )

        return {
            "type": type_int,
            "subtype": subtype_int,
            "manufacturer": manufacturer_int,
            "flags": self.flags,
            "flags_mask": self.flags_mask,
        }

    def __repr__(self) -> str:
        return f"AudioComponentDescription({self.type!r}, {self.subtype!r}, {self.manufacturer!r})"


class AudioComponent(capi.CoreAudioObject):
    """Audio component wrapper"""

    def __init__(self, description: AudioComponentDescription):
        super().__init__()
        self._description = description

    @classmethod
    def find_next(
        cls, description: AudioComponentDescription
    ) -> Optional["AudioComponent"]:
        """Find the next matching audio component"""
        try:
            result = capi.audio_component_find_next(description.to_dict())
            if result is None or result == 0:
                return None
            component = cls(description)
            # Set the object_id using the Cython method
            component._set_object_id(result)
            return component
        except Exception:
            # If lookup fails, component doesn't exist
            return None

    def create_instance(self) -> "AudioUnit":
        """Create an AudioUnit instance from this component"""
        self._ensure_not_disposed()
        try:
            unit_id = capi.audio_component_instance_new(self.object_id)
            unit = AudioUnit(self._description)
            unit._set_object_id(unit_id)
            return unit
        except Exception as e:
            raise AudioUnitError(f"Failed to create instance: {e}")

    def __repr__(self) -> str:
        return f"AudioComponent({self._description.type!r}, {self._description.subtype!r})"


class AudioUnit(capi.CoreAudioObject):
    """Audio unit for real-time audio processing"""

    def __init__(self, description: AudioComponentDescription):
        super().__init__()
        self._description = description
        self._is_initialized = False

    @classmethod
    def default_output(cls) -> "AudioUnit":
        """Create a default output AudioUnit"""
        desc = AudioComponentDescription(
            type="auou",  # kAudioUnitType_Output
            subtype="def ",  # kAudioUnitSubType_DefaultOutput
            manufacturer="appl",  # kAudioUnitManufacturer_Apple
        )
        component = AudioComponent.find_next(desc)
        if component is None:
            raise AudioUnitError("Default output AudioUnit not found")
        return component.create_instance()

    def initialize(self) -> None:
        """Initialize the AudioUnit"""
        self._ensure_not_disposed()
        if not self._is_initialized:
            try:
                capi.audio_unit_initialize(self.object_id)
                self._is_initialized = True
            except Exception as e:
                raise AudioUnitError(f"Failed to initialize: {e}")

    def uninitialize(self) -> None:
        """Uninitialize the AudioUnit"""
        if self._is_initialized:
            try:
                capi.audio_unit_uninitialize(self.object_id)
            except Exception as e:
                raise AudioUnitError(f"Failed to uninitialize: {e}")
            finally:
                self._is_initialized = False

    def start(self) -> None:
        """Start the AudioUnit output"""
        self._ensure_not_disposed()
        if not self._is_initialized:
            raise AudioUnitError("AudioUnit not initialized")
        try:
            capi.audio_output_unit_start(self.object_id)
        except Exception as e:
            raise AudioUnitError(f"Failed to start: {e}")

    def stop(self) -> None:
        """Stop the AudioUnit output"""
        self._ensure_not_disposed()
        try:
            capi.audio_output_unit_stop(self.object_id)
        except Exception as e:
            raise AudioUnitError(f"Failed to stop: {e}")

    def get_property(self, property_id: int, scope: int, element: int) -> bytes:
        """Get a property from the AudioUnit"""
        self._ensure_not_disposed()
        try:
            return capi.audio_unit_get_property(
                self.object_id, property_id, scope, element
            )
        except Exception as e:
            raise AudioUnitError(f"Failed to get property: {e}")

    def set_property(
        self, property_id: int, scope: int, element: int, data: bytes
    ) -> None:
        """Set a property on the AudioUnit"""
        self._ensure_not_disposed()
        try:
            capi.audio_unit_set_property(
                self.object_id, property_id, scope, element, data
            )
        except Exception as e:
            raise AudioUnitError(f"Failed to set property: {e}")

    # ========================================================================
    # Advanced AudioUnit Features
    # ========================================================================

    def get_stream_format(self, scope: str = "output", element: int = 0) -> AudioFormat:
        """Get the stream format for a specific scope and element

        Args:
            scope: 'input', 'output', or 'global' (default: 'output')
            element: Element index (default: 0)

        Returns:
            AudioFormat object with the current stream format

        Raises:
            AudioUnitError: If scope is invalid or getting format fails

        Example::

            # Get the output format of an AudioUnit
            with AudioUnit(component) as unit:
                format = unit.get_stream_format('output', 0)
                print(f"Sample rate: {format.sample_rate}")
                print(f"Channels: {format.channels_per_frame}")
                print(f"Bits: {format.bits_per_channel}")
        """
        self._ensure_not_disposed()

        # Map scope name to constant
        scope_map = {
            "input": capi.get_audio_unit_scope_input(),
            "output": capi.get_audio_unit_scope_output(),
            "global": capi.get_audio_unit_scope_global(),
        }
        scope_val = scope_map.get(scope.lower())
        if scope_val is None:
            raise AudioUnitError(f"Invalid scope: {scope}")

        try:
            asbd_data = self.get_property(
                capi.get_audio_unit_property_stream_format(), scope_val, element
            )

            if len(asbd_data) >= 40:
                asbd = struct.unpack("<dLLLLLLLL", asbd_data[:40])
                (
                    sample_rate,
                    format_id_int,
                    format_flags,
                    bytes_per_packet,
                    frames_per_packet,
                    bytes_per_frame,
                    channels_per_frame,
                    bits_per_channel,
                    reserved,
                ) = asbd

                format_id = capi.int_to_fourchar(format_id_int)

                return AudioFormat(
                    sample_rate=sample_rate,
                    format_id=format_id,
                    format_flags=format_flags,
                    bytes_per_packet=bytes_per_packet,
                    frames_per_packet=frames_per_packet,
                    bytes_per_frame=bytes_per_frame,
                    channels_per_frame=channels_per_frame,
                    bits_per_channel=bits_per_channel,
                )
            else:
                raise AudioUnitError(f"Invalid ASBD data size: {len(asbd_data)}")
        except Exception as e:
            raise AudioUnitError(f"Failed to get stream format: {e}")

    def set_stream_format(
        self, format: AudioFormat, scope: str = "output", element: int = 0
    ) -> None:
        """Set the stream format for a specific scope and element

        Args:
            format: AudioFormat object with desired format
            scope: 'input', 'output', or 'global' (default: 'output')
            element: Element index (must be non-negative, default: 0)

        Raises:
            TypeError: If format is not an AudioFormat
            ValueError: If scope is invalid or element is negative
            AudioUnitError: If setting format fails

        Example::

            import coremusic as cm

            # Create a stereo 44.1kHz 16-bit PCM format
            format = cm.AudioFormat(
                sample_rate=44100.0,
                format_id='lpcm',
                channels_per_frame=2,
                bits_per_channel=16
            )

            # Set the input format on an effect unit
            with cm.AudioUnit(effect_component) as effect:
                effect.set_stream_format(format, 'input', 0)
        """
        if not isinstance(format, AudioFormat):
            raise TypeError(f"format must be AudioFormat, got {type(format).__name__}")
        if element < 0:
            raise ValueError(f"element must be non-negative, got {element}")

        self._ensure_not_disposed()

        # Map scope name to constant
        scope_map = {
            "input": capi.get_audio_unit_scope_input(),
            "output": capi.get_audio_unit_scope_output(),
            "global": capi.get_audio_unit_scope_global(),
        }
        scope_val = scope_map.get(scope.lower())
        if scope_val is None:
            raise AudioUnitError(f"Invalid scope: {scope}")

        try:
            # Convert format_id to integer
            format_id_int = (
                capi.fourchar_to_int(format.format_id)
                if isinstance(format.format_id, str)
                else format.format_id
            )

            # Pack AudioStreamBasicDescription
            asbd_data = struct.pack(
                "<dLLLLLLLL",
                format.sample_rate,
                format_id_int,
                format.format_flags,
                format.bytes_per_packet,
                format.frames_per_packet,
                format.bytes_per_frame,
                format.channels_per_frame,
                format.bits_per_channel,
                0,  # reserved
            )

            self.set_property(
                capi.get_audio_unit_property_stream_format(),
                scope_val,
                element,
                asbd_data,
            )
        except Exception as e:
            raise AudioUnitError(f"Failed to set stream format: {e}")

    @property
    def sample_rate(self) -> float:
        """Get the sample rate (kAudioUnitProperty_SampleRate on global scope)"""
        self._ensure_not_disposed()
        try:
            data = self.get_property(
                2, capi.get_audio_unit_scope_global(), 0
            )  # kAudioUnitProperty_SampleRate = 2
            if len(data) >= 8:
                result: float = struct.unpack("<d", data[:8])[0]
                return result
            return 0.0
        except Exception:
            # Fallback to stream format sample rate
            try:
                return self.get_stream_format("output", 0).sample_rate
            except Exception:
                return 0.0

    @sample_rate.setter
    def sample_rate(self, rate: float) -> None:
        """Set the sample rate (kAudioUnitProperty_SampleRate)"""
        self._ensure_not_disposed()
        data = struct.pack("<d", rate)

        # Try input scope first (for output units, input scope is configurable before init)
        try:
            self.set_property(
                2, capi.get_audio_unit_scope_input(), 0, data
            )  # kAudioUnitProperty_SampleRate = 2
            return
        except Exception:
            pass

        # Try output scope (may work after initialization)
        try:
            self.set_property(
                2, capi.get_audio_unit_scope_output(), 0, data
            )
            return
        except Exception:
            pass

        # Try global scope as fallback
        try:
            self.set_property(
                2, capi.get_audio_unit_scope_global(), 0, data
            )
            return
        except Exception:
            pass

        # Last resort: try to set via stream format on input scope
        try:
            format_data = self.get_stream_format("input", 0)
            format_data.sample_rate = rate
            self.set_stream_format(format_data, "input", 0)
        except Exception as e:
            raise AudioUnitError(f"Failed to set sample rate: {e}")

    @property
    def latency(self) -> float:
        """Get the latency in seconds (kAudioUnitProperty_Latency)"""
        self._ensure_not_disposed()
        try:
            data = self.get_property(
                12, capi.get_audio_unit_scope_global(), 0
            )  # kAudioUnitProperty_Latency = 12
            if len(data) >= 8:
                result: float = struct.unpack("<d", data[:8])[0]
                return result
            return 0.0
        except Exception:
            return 0.0

    @property
    def cpu_load(self) -> float:
        """Get the CPU load as a fraction (0.0 to 1.0) (kAudioUnitProperty_CPULoad)"""
        self._ensure_not_disposed()
        try:
            data = self.get_property(
                6, capi.get_audio_unit_scope_global(), 0
            )  # kAudioUnitProperty_CPULoad = 6
            if len(data) >= 4:
                result: float = struct.unpack("<f", data[:4])[0]
                return result
            return 0.0
        except Exception:
            return 0.0

    @property
    def max_frames_per_slice(self) -> int:
        """Get the maximum frames per slice (kAudioUnitProperty_MaximumFramesPerSlice)"""
        self._ensure_not_disposed()
        try:
            data = self.get_property(
                14, capi.get_audio_unit_scope_global(), 0
            )  # kAudioUnitProperty_MaximumFramesPerSlice = 14
            if len(data) >= 4:
                result: int = struct.unpack("<L", data[:4])[0]
                return result
            return 0
        except Exception:
            return 0

    @max_frames_per_slice.setter
    def max_frames_per_slice(self, frames: int) -> None:
        """Set the maximum frames per slice (kAudioUnitProperty_MaximumFramesPerSlice)"""
        self._ensure_not_disposed()
        try:
            data = struct.pack("<L", frames)
            self.set_property(
                14, capi.get_audio_unit_scope_global(), 0, data
            )  # kAudioUnitProperty_MaximumFramesPerSlice = 14
        except Exception as e:
            raise AudioUnitError(f"Failed to set max frames per slice: {e}")

    def get_parameter_list(self, scope: str = "global") -> List[int]:
        """Get list of available parameter IDs (kAudioUnitProperty_ParameterList)

        Args:
            scope: 'input', 'output', or 'global' (default: 'global')

        Returns:
            List of parameter IDs
        """
        self._ensure_not_disposed()

        scope_map = {
            "input": capi.get_audio_unit_scope_input(),
            "output": capi.get_audio_unit_scope_output(),
            "global": capi.get_audio_unit_scope_global(),
        }
        scope_val = scope_map.get(scope.lower())
        if scope_val is None:
            raise AudioUnitError(f"Invalid scope: {scope}")

        try:
            data = self.get_property(
                3, scope_val, 0
            )  # kAudioUnitProperty_ParameterList = 3
            # Data is an array of UInt32 parameter IDs
            param_count = len(data) // 4
            if param_count > 0:
                return list(struct.unpack(f"<{param_count}L", data[: param_count * 4]))
            return []
        except Exception:
            return []

    def render(self, num_frames: int, timestamp: Optional[int] = None) -> bytes:
        """Render audio frames (for offline processing)

        Args:
            num_frames: Number of frames to render
            timestamp: Optional timestamp (default: None uses current time)

        Returns:
            Rendered audio data as bytes

        Note: This is a simplified render method for offline processing.
        For real-time audio, use render callbacks with the audio player infrastructure.
        """
        # This would require implementing AudioUnitRender which needs more infrastructure
        raise NotImplementedError(
            "Direct rendering not yet implemented. "
            "Use the audio player infrastructure with render callbacks for real-time audio."
        )

    def __repr__(self) -> str:
        if self.is_disposed:
            return f"AudioUnit({self._description.subtype!r}, disposed)"
        status = "initialized" if self._is_initialized else "uninitialized"
        return f"AudioUnit({self._description.subtype!r}, {status})"

    def __enter__(self) -> "AudioUnit":
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.uninitialize()
        self.dispose()

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def dispose(self) -> None:
        """Dispose of the AudioUnit"""
        if not self.is_disposed:
            if self._is_initialized:
                try:
                    capi.audio_unit_uninitialize(self.object_id)
                except Exception:
                    pass  # Best effort cleanup
                finally:
                    self._is_initialized = False

            if self.object_id != 0:
                try:
                    capi.audio_component_instance_dispose(self.object_id)
                except Exception:
                    pass  # Best effort cleanup

            super().dispose()
