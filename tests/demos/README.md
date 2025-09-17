# CoreAudio Demos

This directory contains demonstrations of the coremusic wrapper functionality.

## Main Demo

**`unified_audio_demo.py`** - The comprehensive demonstration that combines all functionality:

- CoreAudio constants and utilities
- Audio file loading and format detection  
- AudioUnit infrastructure testing
- AudioQueue infrastructure testing
- **Actual audio playback using `coreaudio.AudioPlayer`**
- Advanced CoreAudio features
- Comprehensive error handling and user feedback

This is the definitive demo that showcases the complete capabilities of the coremusic wrapper.

## Usage

```bash
# From the project root directory
python3 tests/demos/unified_audio_demo.py
```

## Requirements

- macOS (CoreAudio framework)
- Python 3.x
- Built coremusic module (`make coreaudio`)
- `tests/amen.wav` audio file

## What the Demo Shows

The unified demo demonstrates:

1. **Complete CoreAudio API Access** - All major CoreAudio frameworks are accessible from Python
2. **Audio File Operations** - Loading, format detection, and data extraction
3. **AudioUnit Infrastructure** - Component discovery, instantiation, and lifecycle management
4. **AudioQueue System** - Buffer management and audio output infrastructure
5. **Real Audio Playback** - Using the `coreaudio.AudioPlayer` class for actual sound output
6. **Hardware Control** - Direct access to audio hardware and system objects
7. **Professional Features** - Everything needed for professional audio development

## Results

When run successfully, the unified demo shows:

- ✓ CoreAudio constants and utilities working
- ✓ Audio file operations working
- ✓ AudioUnit infrastructure working
- ✓ AudioPlayer playback working (you should hear audio!)
- ✓ Advanced features working

The coremusic wrapper provides complete access to CoreAudio for professional audio development in Python.
