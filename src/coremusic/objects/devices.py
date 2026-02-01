"""Audio device classes for coremusic.

This module provides classes for working with audio hardware:
- AudioDevice: Represents a hardware audio device
- AudioDeviceManager: Manager for discovering and accessing audio devices
"""

from __future__ import annotations

import struct
from typing import Any, Dict, List, Optional

from .. import capi
from .exceptions import AudioDeviceError

__all__ = [
    "AudioDevice",
    "AudioDeviceManager",
]


class AudioDevice(capi.CoreAudioObject):
    """Represents a hardware audio device with property access

    Provides Pythonic access to audio hardware devices including inputs,
    outputs, and their properties like name, sample rate, channels, etc.
    """

    def __init__(self, device_id: int):
        """Initialize AudioDevice with a device ID

        Args:
            device_id: The AudioObjectID for this device
        """
        super().__init__()
        self._set_object_id(device_id)

    def _get_property_string(
        self, property_id: int, scope: Optional[int] = None, element: int = 0
    ) -> str:
        """Get a string property from the device"""
        if scope is None:
            scope = capi.get_audio_object_property_scope_global()

        try:
            data = capi.audio_object_get_property_string(  # type: ignore[attr-defined]
                self.object_id, property_id, scope, element
            )
            if data:
                # Decode UTF-8 string from CoreFoundation
                # Remove any null terminators
                result: str = data.decode("utf-8", errors="ignore").strip("\x00")
                return result
            return ""
        except Exception:
            return ""

    def _get_property_uint32(
        self, property_id: int, scope: Optional[int] = None, element: int = 0
    ) -> int:
        """Get a UInt32 property from the device"""
        if scope is None:
            scope = capi.get_audio_object_property_scope_global()

        try:
            data = capi.audio_object_get_property_data(
                self.object_id, property_id, scope, element
            )
            if len(data) >= 4:
                result: int = struct.unpack("<L", data[:4])[0]
                return result
            return 0
        except Exception:
            return 0

    def _get_property_float64(
        self, property_id: int, scope: Optional[int] = None, element: int = 0
    ) -> float:
        """Get a Float64 property from the device"""
        if scope is None:
            scope = capi.get_audio_object_property_scope_global()

        try:
            data = capi.audio_object_get_property_data(
                self.object_id, property_id, scope, element
            )
            if len(data) >= 8:
                result: float = struct.unpack("<d", data[:8])[0]
                return result
            return 0.0
        except Exception:
            return 0.0

    @property
    def name(self) -> str:
        """Get the device name"""
        return self._get_property_string(capi.get_audio_object_property_name())

    @property
    def manufacturer(self) -> str:
        """Get the device manufacturer"""
        return self._get_property_string(capi.get_audio_object_property_manufacturer())

    @property
    def uid(self) -> str:
        """Get the device UID (unique identifier)"""
        return self._get_property_string(capi.get_audio_device_property_device_uid())

    @property
    def model_uid(self) -> str:
        """Get the device model UID"""
        return self._get_property_string(capi.get_audio_device_property_model_uid())

    @property
    def transport_type(self) -> int:
        """Get the transport type (USB, PCI, etc.)"""
        return self._get_property_uint32(
            capi.get_audio_device_property_transport_type()
        )

    @property
    def sample_rate(self) -> float:
        """Get the current nominal sample rate"""
        return self._get_property_float64(
            capi.get_audio_device_property_nominal_sample_rate()
        )

    @sample_rate.setter
    def sample_rate(self, rate: float) -> None:
        """Set the nominal sample rate (not all devices support this)"""
        # This would require implementing AudioObjectSetPropertyData
        raise NotImplementedError("Setting sample rate not yet implemented")

    @property
    def is_alive(self) -> bool:
        """Check if the device is alive/connected"""
        value = self._get_property_uint32(
            capi.get_audio_device_property_device_is_alive()
        )
        return bool(value)

    @property
    def is_hidden(self) -> bool:
        """Check if the device is hidden"""
        value = self._get_property_uint32(capi.get_audio_device_property_is_hidden())
        return bool(value)

    def get_stream_configuration(self, scope: str = "output") -> Dict[str, Any]:
        """Get stream configuration (channel layout)

        Args:
            scope: 'input' or 'output' (default: 'output')

        Returns:
            Dictionary with stream configuration information
        """
        scope_map = {
            "input": capi.get_audio_object_property_scope_input(),
            "output": capi.get_audio_object_property_scope_output(),
        }
        scope_val = scope_map.get(scope.lower())
        if scope_val is None:
            raise AudioDeviceError(f"Invalid scope: {scope}")

        try:
            data = capi.audio_object_get_property_data(
                self.object_id,
                capi.get_audio_device_property_stream_configuration(),
                scope_val,
                0,
            )
            # AudioBufferList structure - would need detailed parsing
            # For now, return basic info
            return {"raw_data_length": len(data)}
        except Exception as e:
            raise AudioDeviceError(f"Failed to get stream configuration: {e}")

    def get_volume(self, scope: str = "output", channel: int = 0) -> Optional[float]:
        """Get volume level for a channel

        Args:
            scope: 'input' or 'output' (default: 'output')
            channel: Channel index (0 = main/master)

        Returns:
            Volume level as float (0.0-1.0), or None if not available
        """
        scope_map = {
            "input": capi.get_audio_object_property_scope_input(),
            "output": capi.get_audio_object_property_scope_output(),
        }
        scope_val = scope_map.get(scope.lower())
        if scope_val is None:
            raise AudioDeviceError(f"Invalid scope: {scope}")

        try:
            data = capi.audio_object_get_property_data(
                self.object_id,
                capi.get_audio_device_property_volume_scalar(),  # type: ignore[attr-defined]
                scope_val,
                channel,
            )
            if len(data) >= 4:
                result: float = struct.unpack("<f", data[:4])[0]
                return result
            return None
        except Exception:
            return None

    def set_volume(self, level: float, scope: str = "output", channel: int = 0) -> None:
        """Set volume level for a channel

        Args:
            level: Volume level (0.0-1.0)
            scope: 'input' or 'output' (default: 'output')
            channel: Channel index (0 = main/master)

        Raises:
            AudioDeviceError: If setting volume fails or is not supported
        """
        if not 0.0 <= level <= 1.0:
            raise ValueError("Volume level must be between 0.0 and 1.0")

        scope_map = {
            "input": capi.get_audio_object_property_scope_input(),
            "output": capi.get_audio_object_property_scope_output(),
        }
        scope_val = scope_map.get(scope.lower())
        if scope_val is None:
            raise AudioDeviceError(f"Invalid scope: {scope}")

        # Check if property is settable
        if not capi.audio_object_is_property_settable(  # type: ignore[attr-defined]
            self.object_id,
            capi.get_audio_device_property_volume_scalar(),  # type: ignore[attr-defined]
            scope_val,
            channel,
        ):
            raise AudioDeviceError("Volume is not settable on this device/channel")

        try:
            data = struct.pack("<f", level)
            capi.audio_object_set_property_data(  # type: ignore[attr-defined]
                self.object_id,
                capi.get_audio_device_property_volume_scalar(),  # type: ignore[attr-defined]
                scope_val,
                channel,
                data,
            )
        except Exception as e:
            raise AudioDeviceError(f"Failed to set volume: {e}")

    def get_mute(self, scope: str = "output", channel: int = 0) -> Optional[bool]:
        """Get mute state for a channel

        Args:
            scope: 'input' or 'output' (default: 'output')
            channel: Channel index (0 = main/master)

        Returns:
            True if muted, False if not muted, None if not available
        """
        scope_map = {
            "input": capi.get_audio_object_property_scope_input(),
            "output": capi.get_audio_object_property_scope_output(),
        }
        scope_val = scope_map.get(scope.lower())
        if scope_val is None:
            raise AudioDeviceError(f"Invalid scope: {scope}")

        try:
            data = capi.audio_object_get_property_data(
                self.object_id,
                capi.get_audio_device_property_mute(),  # type: ignore[attr-defined]
                scope_val,
                channel,
            )
            if len(data) >= 4:
                return bool(struct.unpack("<I", data[:4])[0])
            return None
        except Exception:
            return None

    def set_mute(self, muted: bool, scope: str = "output", channel: int = 0) -> None:
        """Set mute state for a channel

        Args:
            muted: True to mute, False to unmute
            scope: 'input' or 'output' (default: 'output')
            channel: Channel index (0 = main/master)

        Raises:
            AudioDeviceError: If setting mute fails or is not supported
        """
        scope_map = {
            "input": capi.get_audio_object_property_scope_input(),
            "output": capi.get_audio_object_property_scope_output(),
        }
        scope_val = scope_map.get(scope.lower())
        if scope_val is None:
            raise AudioDeviceError(f"Invalid scope: {scope}")

        # Check if property is settable
        if not capi.audio_object_is_property_settable(  # type: ignore[attr-defined]
            self.object_id,
            capi.get_audio_device_property_mute(),  # type: ignore[attr-defined]
            scope_val,
            channel,
        ):
            raise AudioDeviceError("Mute is not settable on this device/channel")

        try:
            data = struct.pack("<I", 1 if muted else 0)
            capi.audio_object_set_property_data(  # type: ignore[attr-defined]
                self.object_id,
                capi.get_audio_device_property_mute(),  # type: ignore[attr-defined]
                scope_val,
                channel,
                data,
            )
        except Exception as e:
            raise AudioDeviceError(f"Failed to set mute: {e}")

    def __repr__(self) -> str:
        name = self.name or "Unknown"
        return f"AudioDevice(id={self.object_id}, name='{name}')"

    def __str__(self) -> str:
        return f"{self.name} ({self.manufacturer})"


class AudioDeviceManager:
    """Manager for discovering and accessing audio devices

    Provides static methods for device discovery and retrieval.
    """

    def __repr__(self) -> str:
        devices = self.get_devices()
        return f"AudioDeviceManager({len(devices)} devices)"

    @staticmethod
    def get_all_devices() -> List[AudioDevice]:
        """Get all available audio devices (alias for get_devices)

        Returns:
            List of AudioDevice objects
        """
        return AudioDeviceManager.get_devices()

    @staticmethod
    def get_devices() -> List[AudioDevice]:
        """Get all available audio devices

        Returns:
            List of AudioDevice objects
        """
        device_ids = capi.audio_hardware_get_devices()
        return [AudioDevice(device_id) for device_id in device_ids]

    @staticmethod
    def get_default_output_device() -> Optional[AudioDevice]:
        """Get the default output device

        Returns:
            AudioDevice object or None if no default
        """
        device_id = capi.audio_hardware_get_default_output_device()
        if device_id == 0:
            return None
        return AudioDevice(device_id)

    @staticmethod
    def get_default_input_device() -> Optional[AudioDevice]:
        """Get the default input device

        Returns:
            AudioDevice object or None if no default
        """
        device_id = capi.audio_hardware_get_default_input_device()
        if device_id == 0:
            return None
        return AudioDevice(device_id)

    @staticmethod
    def get_output_devices() -> List[AudioDevice]:
        """Get all output devices

        Returns:
            List of AudioDevice objects that have output capability
        """
        # For now, return all devices - would need to filter by checking
        # stream configuration for output scope
        return AudioDeviceManager.get_devices()

    @staticmethod
    def get_input_devices() -> List[AudioDevice]:
        """Get all input devices

        Returns:
            List of AudioDevice objects that have input capability
        """
        # For now, return all devices - would need to filter by checking
        # stream configuration for input scope
        return AudioDeviceManager.get_devices()

    @staticmethod
    def find_device_by_name(name: str) -> Optional[AudioDevice]:
        """Find a device by name

        Args:
            name: Device name to search for (case-insensitive)

        Returns:
            AudioDevice object or None if not found
        """
        for device in AudioDeviceManager.get_devices():
            if device.name.lower() == name.lower():
                return device
        return None

    @staticmethod
    def find_device_by_uid(uid: str) -> Optional[AudioDevice]:
        """Find a device by UID

        Args:
            uid: Device UID to search for

        Returns:
            AudioDevice object or None if not found
        """
        for device in AudioDeviceManager.get_devices():
            try:
                device_uid = device.uid
                if device_uid and device_uid == uid:
                    return device
            except Exception:
                # Some devices may not have UID property accessible
                continue
        return None

    @staticmethod
    def set_default_output_device(device: AudioDevice) -> None:
        """Set the default output device

        Args:
            device: AudioDevice to set as default output

        Raises:
            AudioDeviceError: If setting fails

        Note:
            This requires the device to support being a default device.
            Not all devices (like aggregate devices) can be set as default.
        """
        try:
            data = struct.pack("<I", device.object_id)
            capi.audio_object_set_property_data(  # type: ignore[attr-defined]
                1,  # kAudioObjectSystemObject
                capi.get_audio_hardware_property_default_output_device(),  # type: ignore[attr-defined]
                capi.get_audio_object_property_scope_global(),
                0,
                data,
            )
        except Exception as e:
            raise AudioDeviceError(f"Failed to set default output device: {e}")

    @staticmethod
    def set_default_input_device(device: AudioDevice) -> None:
        """Set the default input device

        Args:
            device: AudioDevice to set as default input

        Raises:
            AudioDeviceError: If setting fails

        Note:
            This requires the device to support being a default device.
            Not all devices (like aggregate devices) can be set as default.
        """
        try:
            data = struct.pack("<I", device.object_id)
            capi.audio_object_set_property_data(  # type: ignore[attr-defined]
                1,  # kAudioObjectSystemObject
                capi.get_audio_hardware_property_default_input_device(),  # type: ignore[attr-defined]
                capi.get_audio_object_property_scope_global(),
                0,
                data,
            )
        except Exception as e:
            raise AudioDeviceError(f"Failed to set default input device: {e}")
