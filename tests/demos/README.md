# CoreMusic Demos

This directory contains demonstrations of the coremusic wrapper functionality.

## Demo Files

Files are organized by prefix for easy discovery:

### Audio Processing
| File | Description |
|------|-------------|
| `audio_analysis.py` | Beat detection, pitch tracking, spectral analysis, MFCCs |
| `audio_slicing.py` | Audio slicing, time-stretching, segment extraction |
| `audio_visualization.py` | Waveform and spectral visualization |

### AudioUnit
| File | Description |
|------|-------------|
| `audiounit_advanced.py` | Advanced AudioUnit features and techniques |
| `audiounit_browser.py` | Discover and list available AudioUnit plugins |
| `audiounit_highlevel.py` | High-level AudioUnit API usage |
| `audiounit_instrument.py` | Software instrument plugins (DLSMusicDevice, etc.) |

### Link (Ableton Link)
| File | Description |
|------|-------------|
| `link_audio.py` | Synchronize audio playback with Link |
| `link_highlevel.py` | High-level Link session management |
| `link_midi.py` | Synchronize MIDI clock with Link tempo |

### MIDI
| File | Description |
|------|-------------|
| `midi_render.py` | Render MIDI to audio using software instruments |
| `midi_to_audiounit.py` | Control AudioUnit instruments with MIDI |
| `midi_utilities.py` | MIDI utilities, device listing, message parsing |

### Core/Utilities
| File | Description |
|------|-------------|
| `async_io.py` | Async/await patterns for non-blocking I/O |
| `batch_processing.py` | Process multiple audio files efficiently |
| `daw.py` | DAW framework: timeline, tracks, clips, automation |
| `oo_api.py` | Object-oriented API with AudioDevice discovery |
| `os_status_errors.py` | OSStatus error handling and translation |
| `scipy_integration.py` | Signal processing with SciPy (FFT, filters) |
| `streaming.py` | Stream processing with chunked I/O |
| `unified_audio.py` | Comprehensive demo combining all functionality |
| `utilities.py` | Audio utilities and helper functions |

## Main Demo

**`unified_audio.py`** - The comprehensive demonstration that combines all functionality:

- CoreAudio constants and utilities
- Audio file loading and format detection
- AudioUnit infrastructure testing
- AudioQueue infrastructure testing
- **Actual audio playback using `coremusic.AudioPlayer`**
- Advanced CoreAudio features
- Comprehensive error handling and user feedback

## Usage

```bash
# From the project root directory
python3 tests/demos/unified_audio.py

# Run specific demos
python3 tests/demos/audiounit_browser.py
python3 tests/demos/audio_analysis.py
python3 tests/demos/link_highlevel.py
```

## Requirements

- macOS (CoreAudio framework)
- Python 3.11+
- Built coremusic module (`make`)
- `tests/amen.wav` audio file (for demos that use it)

Some demos have additional requirements:
- **NumPy**: `audio_analysis.py`, `audio_visualization.py`, `scipy_integration.py`
- **SciPy**: `audio_analysis.py`, `scipy_integration.py`

## What the Demos Show

1. **Complete CoreAudio API Access** - All major CoreAudio frameworks accessible from Python
2. **Audio File Operations** - Loading, format detection, and data extraction
3. **AudioUnit Infrastructure** - Component discovery, instantiation, and lifecycle management
4. **AudioQueue System** - Buffer management and audio output infrastructure
5. **Real Audio Playback** - Using the `coremusic.AudioPlayer` class for actual sound output
6. **Hardware Control** - Direct access to audio hardware and system objects
7. **MIDI Processing** - Device discovery, message handling, software synthesis
8. **Ableton Link** - Tempo synchronization across applications
9. **DAW Framework** - Timeline, tracks, clips, and automation
