## API Reference

**Note:** This document covers the **functional C API** (`coremusic.capi`). These functions require explicit import:

```python
import coremusic.capi as capi

# Example usage
file_id = capi.audio_file_open_url("file.wav", capi.get_audio_file_read_permission(), 0)
```

For the recommended **object-oriented API**, see the main README.md.

---

### Audio File Operations

#### `audio_file_open_url(file_path, permissions, file_type_hint)`

Open an audio file for reading or writing.

**Parameters:**

- `file_path` (str): Path to the audio file
- `permissions` (int): File access permissions (use `get_audio_file_read_permission()`)
- `file_type_hint` (int): File type hint (use `get_audio_file_wave_type()` for WAV files)

**Returns:** Audio file ID (int)

#### `audio_file_get_property(audio_file_id, property_id)`

Get a property from an audio file.

**Parameters:**

- `audio_file_id` (int): Audio file ID from `audio_file_open_url()`
- `property_id` (int): Property ID (use `get_audio_file_property_data_format()` for format info)

**Returns:** Property data as bytes

#### `audio_file_read_packets(audio_file_id, start_packet, num_packets)`

Read audio packets from a file.

**Parameters:**

- `audio_file_id` (int): Audio file ID
- `start_packet` (int): Starting packet number
- `num_packets` (int): Number of packets to read

**Returns:** Tuple of (packet_data, packets_read)

#### `audio_file_close(audio_file_id)`

Close an audio file.

### AudioUnit Operations

#### `audio_component_find_next(description)`

Find an audio component matching the description.

**Parameters:**

- `description` (dict): Component description with keys: type, subtype, manufacturer, flags, flags_mask

**Returns:** Component ID (int) or None

#### `audio_component_instance_new(component_id)`

Create a new instance of an audio component.

**Parameters:**

- `component_id` (int): Component ID from `audio_component_find_next()`

**Returns:** AudioUnit instance ID (int)

#### `audio_unit_initialize(audio_unit_id)`

Initialize an AudioUnit.

#### `audio_output_unit_start(audio_unit_id)`

Start audio output.

#### `audio_output_unit_stop(audio_unit_id)`

Stop audio output.

### AudioQueue Operations

#### `audio_queue_new_output(audio_format)`

Create a new output audio queue.

**Parameters:**

- `audio_format` (dict): Audio format specification

**Returns:** AudioQueue ID (int)

#### `audio_queue_allocate_buffer(queue_id, buffer_size)`

Allocate a buffer for an audio queue.

#### `audio_queue_start(queue_id)`

Start an audio queue.

#### `audio_queue_stop(queue_id, immediate)`

Stop an audio queue.

### CoreMIDI Operations

#### MIDI Message Creation

##### `midi1_channel_voice_message(group, status, channel, data1, data2)`

Create a MIDI 1.0 Universal Packet for channel voice messages.

**Parameters:**
- `group` (int): MIDI group (0-15)
- `status` (int): MIDI status byte (use `get_midi_status_*()` functions)
- `channel` (int): MIDI channel (0-15)
- `data1` (int): First data byte (0-127)
- `data2` (int): Second data byte (0-127)

**Returns:** 32-bit Universal MIDI Packet

##### `midi2_channel_voice_message(group, status, channel, index, value)`

Create a MIDI 2.0 Universal Packet for enhanced channel voice messages.

**Parameters:**
- `group` (int): MIDI group (0-15)
- `status` (int): MIDI 2.0 status
- `channel` (int): MIDI channel (0-15)
- `index` (int): Parameter index
- `value` (long): 32-bit parameter value

**Returns:** 32-bit Universal MIDI Packet

#### MIDI Device Management

##### `midi_get_number_of_devices()`

Get the number of MIDI devices in the system.

**Returns:** Number of devices (int)

##### `midi_get_number_of_sources()`

Get the number of MIDI sources in the system.

**Returns:** Number of sources (int)

##### `midi_device_create(name)`

Create a new virtual MIDI device.

**Parameters:**
- `name` (str): Device name

**Returns:** Device reference (int)

##### `midi_device_dispose(device)`

Dispose of a MIDI device.

**Parameters:**
- `device` (int): Device reference from `midi_device_create()`

#### MIDI Thru Connections

##### `midi_thru_connection_params_initialize()`

Initialize MIDI thru connection parameters with default values.

**Returns:** Parameter dictionary with routing and transformation settings

##### `midi_thru_connection_create()`

Create a basic MIDI thru connection.

**Returns:** Thru connection reference (int)

##### `midi_thru_connection_create_with_params(params)`

Create a MIDI thru connection with custom parameters.

**Parameters:**
- `params` (dict): Connection parameters from `midi_thru_connection_params_initialize()`

**Returns:** Thru connection reference (int)

##### `midi_thru_connection_dispose(connection)`

Dispose of a MIDI thru connection.

**Parameters:**
- `connection` (int): Connection reference

### Utility Functions

#### `fourchar_to_int(code)`

Convert a four-character code string to integer.

**Parameters:**

- `code` (str): Four-character code (e.g., 'WAVE', 'TEXT')

**Returns:** Integer representation

#### `int_to_fourchar(n)`

Convert an integer to a four-character code string.

**Parameters:**

- `n` (int): Integer value

**Returns:** Four-character code string

#### MIDI Constants

Access MIDI constants through getter functions:

- `get_midi_status_note_on()`: Note On status (0x90)
- `get_midi_status_note_off()`: Note Off status (0x80)
- `get_midi_status_control_change()`: Control Change status (0xB0)
- `get_midi_transform_none()`: No transformation
- `get_midi_transform_add()`: Add value transformation
- `get_midi_transform_scale()`: Scale value transformation

### AudioPlayer Class

The `AudioPlayer` class provides a high-level interface for audio playback:

#### `AudioPlayer()`

Create a new AudioPlayer instance.

#### `load_file(file_path)`

Load an audio file for playback.

#### `setup_output()`

Setup the audio output unit.

#### `start()`

Start audio playback.

#### `stop()`

Stop audio playback.

#### `set_looping(loop)`

Enable or disable looping playback.

#### `is_playing()`

Check if audio is currently playing.

#### `get_progress()`

Get current playback progress (0.0 to 1.0).

#### `reset_playback()`

Reset playback to the beginning.
