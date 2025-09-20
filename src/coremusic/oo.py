#!/usr/bin/env python3
"""Object-oriented Python classes for coremusic.

This module provides Pythonic, object-oriented wrappers around the CoreAudio
functional API. These classes handle resource management automatically and
provide a more intuitive interface for CoreAudio development.

All classes inherit from CoreAudioObject (Cython extension class) for automatic
resource cleanup, but are implemented as pure Python classes for simplicity.
"""

from typing import Optional, Union, List, Dict, Any
from pathlib import Path

from . import capi
from .objects import CoreAudioObject

# ============================================================================
# Exception Hierarchy
# ============================================================================

class CoreAudioError(Exception):
    """Base exception for CoreAudio errors"""
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code

class AudioFileError(CoreAudioError):
    """Exception for AudioFile operations"""
    pass

class AudioQueueError(CoreAudioError):
    """Exception for AudioQueue operations"""
    pass

class AudioUnitError(CoreAudioError):
    """Exception for AudioUnit operations"""
    pass

class MIDIError(CoreAudioError):
    """Exception for MIDI operations"""
    pass

class MusicPlayerError(CoreAudioError):
    """Exception for MusicPlayer operations"""
    pass

# ============================================================================
# Audio Format
# ============================================================================

class AudioFormat:
    """Pythonic representation of AudioStreamBasicDescription"""

    def __init__(self, sample_rate: float, format_id: str,
                 format_flags: int = 0, bytes_per_packet: int = 0,
                 frames_per_packet: int = 0, bytes_per_frame: int = 0,
                 channels_per_frame: int = 2, bits_per_channel: int = 16):
        self.sample_rate = sample_rate
        self.format_id = format_id
        self.format_flags = format_flags
        self.bytes_per_packet = bytes_per_packet
        self.frames_per_packet = frames_per_packet
        self.bytes_per_frame = bytes_per_frame
        self.channels_per_frame = channels_per_frame
        self.bits_per_channel = bits_per_channel

    @property
    def is_pcm(self) -> bool:
        """Check if this is a PCM format"""
        return self.format_id == 'lpcm'

    @property
    def is_stereo(self) -> bool:
        """Check if this is stereo (2 channels)"""
        return self.channels_per_frame == 2

    @property
    def is_mono(self) -> bool:
        """Check if this is mono (1 channel)"""
        return self.channels_per_frame == 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for functional API"""
        # Convert format_id string to integer using fourcc_to_int
        from . import capi
        format_id_int = capi.fourchar_to_int(self.format_id) if isinstance(self.format_id, str) else self.format_id

        return {
            'sample_rate': self.sample_rate,
            'format_id': format_id_int,
            'format_flags': self.format_flags,
            'bytes_per_packet': self.bytes_per_packet,
            'frames_per_packet': self.frames_per_packet,
            'bytes_per_frame': self.bytes_per_frame,
            'channels_per_frame': self.channels_per_frame,
            'bits_per_channel': self.bits_per_channel
        }

    def __repr__(self) -> str:
        return (f"AudioFormat({self.sample_rate}Hz, {self.format_id}, "
                f"channels={self.channels_per_frame}, bits={self.bits_per_channel})")

# ============================================================================
# Audio File Operations
# ============================================================================

class AudioFile(CoreAudioObject):
    """High-level audio file operations with automatic resource management"""

    def __init__(self, path: Union[str, Path]):
        super().__init__()
        self._path = str(path)
        self._format: Optional[AudioFormat] = None
        self._is_open = False

    def open(self):
        """Open the audio file"""
        self._ensure_not_disposed()
        if not self._is_open:
            try:
                file_id = capi.audio_file_open_url(self._path)
                self._set_object_id(file_id)
                self._is_open = True
            except Exception as e:
                raise AudioFileError(f"Failed to open file {self._path}: {e}")
        return self

    def close(self) -> None:
        """Close the audio file"""
        if self._is_open:
            try:
                capi.audio_file_close(self.object_id)
            except Exception as e:
                raise AudioFileError(f"Failed to close file: {e}")
            finally:
                self._is_open = False
                self.dispose()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def format(self) -> AudioFormat:
        """Get the audio format of the file"""
        self._ensure_not_disposed()
        if not self._is_open:
            self.open()

        if self._format is None:
            try:
                format_data = capi.audio_file_get_property(
                    self.object_id,
                    capi.get_audio_file_property_data_format()
                )
                # Parse format data into AudioFormat
                # This would need implementation based on the format_data structure
                self._format = AudioFormat(44100.0, 'lpcm')  # Placeholder
            except Exception as e:
                raise AudioFileError(f"Failed to get format: {e}")

        return self._format

    def read_packets(self, start_packet: int, packet_count: int):
        """Read audio packets from the file"""
        self._ensure_not_disposed()
        if not self._is_open:
            self.open()

        try:
            return capi.audio_file_read_packets(self.object_id, start_packet, packet_count)
        except Exception as e:
            raise AudioFileError(f"Failed to read packets: {e}")

    def get_property(self, property_id: int):
        """Get a property from the audio file"""
        self._ensure_not_disposed()
        if not self._is_open:
            self.open()

        try:
            return capi.audio_file_get_property(self.object_id, property_id)
        except Exception as e:
            raise AudioFileError(f"Failed to get property: {e}")

    @property
    def duration(self) -> float:
        """Duration in seconds (placeholder implementation)"""
        # This would need proper implementation based on frame count and sample rate
        return 0.0

    def __repr__(self) -> str:
        status = "open" if self._is_open else "closed"
        return f"AudioFile({self._path}, {status})"

    def dispose(self) -> None:
        """Dispose of the audio file"""
        if not self.is_disposed:
            if self._is_open:
                try:
                    capi.audio_file_close(self.object_id)
                except:
                    pass  # Best effort cleanup
                finally:
                    self._is_open = False
            super().dispose()

class AudioFileStream(CoreAudioObject):
    """Audio file stream for parsing audio data"""

    def __init__(self, file_type_hint: int = 0):
        super().__init__()
        self._file_type_hint = file_type_hint
        self._is_open = False

    def open(self):
        """Open the audio file stream"""
        self._ensure_not_disposed()
        if not self._is_open:
            try:
                stream_id = capi.audio_file_stream_open()
                self._set_object_id(stream_id)
                self._is_open = True
            except Exception as e:
                raise AudioFileError(f"Failed to open stream: {e}")
        return self

    def close(self) -> None:
        """Close the audio file stream"""
        if self._is_open:
            try:
                capi.audio_file_stream_close(self.object_id)
            except Exception as e:
                raise AudioFileError(f"Failed to close stream: {e}")
            finally:
                self._is_open = False
                self.dispose()

    @property
    def ready_to_produce_packets(self) -> bool:
        """Check if the stream is ready to produce packets"""
        self._ensure_not_disposed()
        if not self._is_open:
            return False
        try:
            return capi.audio_file_stream_get_property_ready_to_produce_packets(self.object_id)
        except Exception:
            return False

    def parse_bytes(self, data: bytes) -> None:
        """Parse audio data bytes"""
        self._ensure_not_disposed()
        if not self._is_open:
            self.open()
        try:
            capi.audio_file_stream_parse_bytes(self.object_id, data)
        except Exception as e:
            raise AudioFileError(f"Failed to parse bytes: {e}")

    def seek(self, packet_offset: int) -> None:
        """Seek to packet offset"""
        self._ensure_not_disposed()
        if not self._is_open:
            raise AudioFileError("Stream not open")
        try:
            capi.audio_file_stream_seek(self.object_id, packet_offset)
        except Exception as e:
            raise AudioFileError(f"Failed to seek: {e}")

    def get_property(self, property_id: int):
        """Get a property from the audio file stream"""
        self._ensure_not_disposed()
        if not self._is_open:
            raise AudioFileError("Stream not open")
        try:
            return capi.audio_file_stream_get_property(self.object_id, property_id)
        except Exception as e:
            raise AudioFileError(f"Failed to get property: {e}")

    def dispose(self) -> None:
        """Dispose of the audio file stream"""
        if not self.is_disposed:
            if self._is_open:
                try:
                    capi.audio_file_stream_close(self.object_id)
                except:
                    pass  # Best effort cleanup
                finally:
                    self._is_open = False
            super().dispose()

# ============================================================================
# Audio Queue Framework
# ============================================================================

class AudioBuffer(CoreAudioObject):
    """Audio buffer for queue operations"""

    def __init__(self, queue_id: int, buffer_size: int):
        super().__init__()
        self._queue_id = queue_id
        self._buffer_size = buffer_size

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

class AudioQueue(CoreAudioObject):
    """Audio queue for buffered playback and recording"""

    def __init__(self, audio_format: AudioFormat):
        super().__init__()
        self._format = audio_format
        self._buffers: List[AudioBuffer] = []

    @classmethod
    def new_output(cls, audio_format: AudioFormat) -> 'AudioQueue':
        """Create a new output audio queue"""
        queue = cls(audio_format)
        try:
            queue_id = capi.audio_queue_new_output(audio_format.to_dict())
            queue._set_object_id(queue_id)
        except Exception as e:
            raise AudioQueueError(f"Failed to create output queue: {e}")
        return queue

    def allocate_buffer(self, buffer_size: int) -> AudioBuffer:
        """Allocate an audio buffer"""
        self._ensure_not_disposed()
        try:
            buffer_id = capi.audio_queue_allocate_buffer(self.object_id, buffer_size)
            buffer = AudioBuffer(self.object_id, buffer_size)
            buffer._set_object_id(buffer_id)
            self._buffers.append(buffer)
            return buffer
        except Exception as e:
            raise AudioQueueError(f"Failed to allocate buffer: {e}")

    def enqueue_buffer(self, buffer: AudioBuffer) -> None:
        """Enqueue an audio buffer"""
        self._ensure_not_disposed()
        try:
            capi.audio_queue_enqueue_buffer(self.object_id, buffer.object_id)
        except Exception as e:
            raise AudioQueueError(f"Failed to enqueue buffer: {e}")

    def start(self) -> None:
        """Start the audio queue"""
        self._ensure_not_disposed()
        try:
            capi.audio_queue_start(self.object_id)
        except Exception as e:
            raise AudioQueueError(f"Failed to start queue: {e}")

    def stop(self, immediate: bool = True) -> None:
        """Stop the audio queue"""
        self._ensure_not_disposed()
        try:
            capi.audio_queue_stop(self.object_id, immediate)
        except Exception as e:
            raise AudioQueueError(f"Failed to stop queue: {e}")

    def dispose(self, immediate: bool = True) -> None:
        """Dispose of the audio queue"""
        if not self.is_disposed:
            try:
                capi.audio_queue_dispose(self.object_id, immediate)
            except Exception as e:
                raise AudioQueueError(f"Failed to dispose queue: {e}")
            finally:
                # Clear buffer references and call base dispose
                self._buffers.clear()
                super().dispose()

# ============================================================================
# Audio Component & AudioUnit Framework
# ============================================================================

class AudioComponentDescription:
    """Pythonic representation of AudioComponent description"""

    def __init__(self, type: str, subtype: str, manufacturer: str,
                 flags: int = 0, flags_mask: int = 0):
        self.type = type
        self.subtype = subtype
        self.manufacturer = manufacturer
        self.flags = flags
        self.flags_mask = flags_mask

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for functional API"""
        # Convert fourcc strings to integers
        from . import capi
        type_int = capi.fourchar_to_int(self.type) if isinstance(self.type, str) else self.type
        subtype_int = capi.fourchar_to_int(self.subtype) if isinstance(self.subtype, str) else self.subtype
        manufacturer_int = capi.fourchar_to_int(self.manufacturer) if isinstance(self.manufacturer, str) else self.manufacturer

        return {
            'type': type_int,
            'subtype': subtype_int,
            'manufacturer': manufacturer_int,
            'flags': self.flags,
            'flags_mask': self.flags_mask
        }

class AudioComponent(CoreAudioObject):
    """Audio component wrapper"""

    def __init__(self, description: AudioComponentDescription):
        super().__init__()
        self._description = description

    @classmethod
    def find_next(cls, description: AudioComponentDescription) -> Optional['AudioComponent']:
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

    def create_instance(self) -> 'AudioUnit':
        """Create an AudioUnit instance from this component"""
        self._ensure_not_disposed()
        try:
            unit_id = capi.audio_component_instance_new(self.object_id)
            unit = AudioUnit(self._description)
            unit._set_object_id(unit_id)
            return unit
        except Exception as e:
            raise AudioUnitError(f"Failed to create instance: {e}")

class AudioUnit(CoreAudioObject):
    """Audio unit for real-time audio processing"""

    def __init__(self, description: AudioComponentDescription):
        super().__init__()
        self._description = description
        self._is_initialized = False

    @classmethod
    def default_output(cls) -> 'AudioUnit':
        """Create a default output AudioUnit"""
        desc = AudioComponentDescription(
            type='auou',  # kAudioUnitType_Output
            subtype='def ',  # kAudioUnitSubType_DefaultOutput
            manufacturer='appl'  # kAudioUnitManufacturer_Apple
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
            return capi.audio_unit_get_property(self.object_id, property_id, scope, element)
        except Exception as e:
            raise AudioUnitError(f"Failed to get property: {e}")

    def set_property(self, property_id: int, scope: int, element: int, data: bytes) -> None:
        """Set a property on the AudioUnit"""
        self._ensure_not_disposed()
        try:
            capi.audio_unit_set_property(self.object_id, property_id, scope, element, data)
        except Exception as e:
            raise AudioUnitError(f"Failed to set property: {e}")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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
                except:
                    pass  # Best effort cleanup
                finally:
                    self._is_initialized = False

            if self.object_id != 0:
                try:
                    capi.audio_component_instance_dispose(self.object_id)
                except:
                    pass  # Best effort cleanup

            super().dispose()

# ============================================================================
# MIDI Framework
# ============================================================================

class MIDIPort(CoreAudioObject):
    """Base class for MIDI ports"""

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._client = None  # Reference to parent MIDIClient

    @property
    def name(self) -> str:
        return self._name

    def dispose(self) -> None:
        """Dispose of the MIDI port"""
        if not self.is_disposed:
            try:
                capi.midi_port_dispose(self.object_id)
            except Exception:
                # Best effort disposal - some MIDI operations may fail in test environments
                pass
            finally:
                # Remove from client's port list if we have a client reference
                if self._client and hasattr(self._client, '_ports'):
                    try:
                        self._client._ports.remove(self)
                    except ValueError:
                        pass  # Already removed
                super().dispose()

class MIDIInputPort(MIDIPort):
    """MIDI input port for receiving MIDI data"""

    def connect_source(self, source) -> None:
        """Connect to a MIDI source"""
        self._ensure_not_disposed()
        try:
            capi.midi_port_connect_source(self.object_id, source.object_id)
        except Exception as e:
            raise MIDIError(f"Failed to connect source: {e}")

    def disconnect_source(self, source) -> None:
        """Disconnect from a MIDI source"""
        self._ensure_not_disposed()
        try:
            capi.midi_port_disconnect_source(self.object_id, source.object_id)
        except Exception as e:
            raise MIDIError(f"Failed to disconnect source: {e}")

class MIDIOutputPort(MIDIPort):
    """MIDI output port for sending MIDI data"""

    def send_data(self, destination, data: bytes, timestamp: int = 0) -> None:
        """Send MIDI data to a destination"""
        self._ensure_not_disposed()
        try:
            capi.midi_send(self.object_id, destination.object_id, data, timestamp)
        except Exception as e:
            raise MIDIError(f"Failed to send data: {e}")

class MIDIClient(CoreAudioObject):
    """MIDI client for managing MIDI operations"""

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._ports: List[MIDIPort] = []
        try:
            client_id = capi.midi_client_create(name)
            self._set_object_id(client_id)
        except Exception as e:
            raise MIDIError(f"Failed to create MIDI client: {e}")

    @property
    def name(self) -> str:
        return self._name

    def create_input_port(self, name: str) -> MIDIInputPort:
        """Create a MIDI input port"""
        self._ensure_not_disposed()
        try:
            port_id = capi.midi_input_port_create(self.object_id, name)
            port = MIDIInputPort(name)
            port._set_object_id(port_id)
            port._client = self
            self._ports.append(port)
            return port
        except Exception as e:
            raise MIDIError(f"Failed to create input port: {e}")

    def create_output_port(self, name: str) -> MIDIOutputPort:
        """Create a MIDI output port"""
        self._ensure_not_disposed()
        try:
            port_id = capi.midi_output_port_create(self.object_id, name)
            port = MIDIOutputPort(name)
            port._set_object_id(port_id)
            port._client = self
            self._ports.append(port)
            return port
        except Exception as e:
            raise MIDIError(f"Failed to create output port: {e}")

    def dispose(self) -> None:
        """Dispose of the MIDI client and all its ports"""
        if not self.is_disposed:
            # Dispose all ports first
            for port in self._ports[:]:  # Copy list to avoid modification during iteration
                if not port.is_disposed:
                    try:
                        port.dispose()
                    except:
                        pass  # Best effort cleanup

            try:
                capi.midi_client_dispose(self.object_id)
            except Exception:
                # Best effort disposal - some MIDI operations may fail in test environments
                pass
            finally:
                # Clear port references and call base dispose
                self._ports.clear()
                super().dispose()