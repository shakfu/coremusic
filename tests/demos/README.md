# CoreMusic Demos

Focused, single-purpose examples demonstrating coremusic functionality.

## Directory Structure

### `analysis/` - Audio Analysis
| File | Description |
|------|-------------|
| `file_info.py` | Extract audio file information |
| `peak_rms.py` | Calculate peak and RMS levels |
| `silence_detection.py` | Detect silence regions |

### `audiounit/` - AudioUnit Plugins
| File | Description |
|------|-------------|
| `list_plugins.py` | List plugins by category |
| `plugin_info.py` | Show detailed plugin info |
| `parameter_control.py` | Control plugin parameters |
| `factory_presets.py` | List factory presets |
| `discover_plugins.py` | Discover plugins (high-level API) |
| `stream_format.py` | Show stream format configuration |

### `conversion/` - Format Conversion
| File | Description |
|------|-------------|
| `stereo_to_mono.py` | Convert stereo to mono |
| `format_presets.py` | Show available format presets |

### `devices/` - Audio Device Management
| File | Description |
|------|-------------|
| `list_devices.py` | List all audio devices |
| `default_devices.py` | Show default input/output |
| `find_device.py` | Find device by name or UID |

### `effects/` - Audio Effects
| File | Description |
|------|-------------|
| `create_chain.py` | Create effects chain with AUGraph |
| `find_by_name.py` | Find AudioUnits by name |
| `fourcc_reference.py` | FourCC codes reference |

### `link/` - Ableton Link
| File | Description |
|------|-------------|
| `session.py` | Create a Link session |
| `beat_tracking.py` | Track beats using Link |

### `midi/` - MIDI
| File | Description |
|------|-------------|
| `create_sequence.py` | Create a MIDI sequence |
| `multi_track.py` | Multi-track composition |
| `routing.py` | MIDI routing with transforms |

### `numpy/` - NumPy Integration
| File | Description |
|------|-------------|
| `read_audio.py` | Read audio as NumPy array |
| `channel_analysis.py` | Analyze individual channels |
| `format_dtypes.py` | AudioFormat to NumPy dtype |

### `slicing/` - Audio Slicing
| File | Description |
|------|-------------|
| `onset_slicing.py` | Slice using onset detection |
| `grid_slicing.py` | Slice into equal divisions |
| `recombine.py` | Recombine slices |

### `streaming/` - Real-Time Streaming
| File | Description |
|------|-------------|
| `input_stream.py` | Audio input stream |
| `output_stream.py` | Audio output stream |
| `latency_comparison.py` | Compare buffer latencies |

### `visualization/` - Audio Visualization
| File | Description |
|------|-------------|
| `waveform.py` | Plot waveform |
| `spectrogram.py` | Plot spectrogram |
| `spectrum.py` | Plot frequency spectrum |

### Root Directory
| File | Description |
|------|-------------|
| `unified_audio.py` | Comprehensive demo |
| `daw.py` | DAW framework demo |

## Usage

```bash
# Run any focused example
uv run python tests/demos/devices/list_devices.py
uv run python tests/demos/analysis/file_info.py tests/amen.wav
uv run python tests/demos/audiounit/list_plugins.py

# Run comprehensive demo
uv run python tests/demos/unified_audio.py
```

## Requirements

- macOS (CoreAudio framework)
- Python 3.11+
- Built coremusic module (`make`)

Optional dependencies:
- **NumPy**: analysis/, numpy/, slicing/, visualization/
- **SciPy**: slicing/
- **matplotlib**: visualization/
