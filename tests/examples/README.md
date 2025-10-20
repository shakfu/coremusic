# coremusic Examples

Complete, working examples demonstrating coremusic capabilities.

## Quick Start

All examples are standalone Python scripts that can be run directly:

```bash
# From the project root
python examples/audio_inspector.py tests/amen.wav

# Or make executable and run
chmod +x examples/audio_inspector.py
./examples/audio_inspector.py tests/amen.wav
```

## Available Examples

### Basic Examples

#### Audio Inspector
**File:** `audio_inspector.py`

Comprehensive audio file inspection tool that displays:
- File information (size, path)
- Format details (sample rate, channels, bit depth)
- Duration and frame count
- Quality classification
- Bitrate calculations

**Usage:**
```bash
python examples/audio_inspector.py audio.wav
```

**Example Output:**
```
======================================================================
Audio File Inspector
======================================================================

FILE INFORMATION
----------------------------------------------------------------------
  Filename:     amen.wav
  File Size:    529.03 KB

FORMAT INFORMATION
----------------------------------------------------------------------
  Format ID:    lpcm
  Sample Rate:  44,100 Hz
  Channels:     2 (Stereo)
  Bit Depth:    16-bit

CLASSIFICATION
----------------------------------------------------------------------
  Quality:      CD Quality
  Bitrate:      1,411 kbps
```

#### Audio Converter
**File:** `audio_converter.py`

Convert audio files between different sample rates while preserving quality.

**Features:**
- Sample rate conversion (44.1kHz, 48kHz, 96kHz, etc.)
- Duration preservation verification
- Format validation
- Progress reporting

**Usage:**
```bash
# Convert to 48kHz
python examples/audio_converter.py input.wav output.wav --rate 48000

# Convert to 44.1kHz
python examples/audio_converter.py input.wav output.wav --rate 44100
```

**Example Output:**
```
Converting: input.wav
Output:     output.wav

Source Format:
  Sample Rate:  44100.0 Hz
  Channels:     2
  Duration:     2.74s

Target Format:
  Sample Rate:  48000.0 Hz
  Channels:     2

Converting...
Conversion complete!

Verifying output...
  Output Sample Rate: 48000.0 Hz
  Output Duration:    2.74s
  Duration Preserved: Yes
```

## Coming Soon

### Audio Processing Examples
- **Real-time Processor**: Process audio in real-time
- **Batch Converter**: Convert multiple files
- **Audio Analyzer**: Analyze audio characteristics
- **Waveform Generator**: Generate audio waveforms

### AudioUnit Examples
- **AudioUnit Explorer**: Discover available AudioUnits
- **Effect Chain**: Chain multiple audio effects
- **Custom Processor**: Create custom AudioUnit
- **Parameter Controller**: Automate parameters

### MIDI Examples
- **MIDI Monitor**: Monitor MIDI input
- **MIDI Router**: Route MIDI between devices
- **MIDI Transformer**: Transform MIDI messages
- **Virtual Keyboard**: Create virtual MIDI keyboard

### Advanced Examples
- **Multi-channel Processor**: Handle surround audio
- **Low Latency Streamer**: Minimal latency streaming
- **Audio Visualizer**: Real-time visualization
- **SciPy Integration**: Signal processing with SciPy

## Prerequisites

All examples require:
- coremusic installed and built
- macOS with CoreAudio
- Python 3.6+

Some examples have additional requirements:
- **NumPy**: For audio analysis examples
- **SciPy**: For signal processing examples
- **Matplotlib**: For visualization examples

Install optional dependencies:
```bash
pip install numpy scipy matplotlib
```

## Example Structure

Each example follows this structure:

1. **Docstring**: Description and usage
2. **Imports**: Required modules
3. **Helper Functions**: Utility functions
4. **Main Function**: Core implementation
5. **Argument Parsing**: Command-line handling
6. **Error Handling**: Robust error handling

## Contributing Examples

We welcome example contributions! To add an example:

1. Create a standalone, working script
2. Add comprehensive docstrings
3. Include usage instructions in the docstring
4. Handle errors gracefully
5. Add to this README
6. Test with various inputs

### Example Template

```python
#!/usr/bin/env python3
"""
Example Name

Brief description of what this example demonstrates.

Usage: python example_name.py <arguments>
"""

import coremusic as cm
import sys

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python example_name.py <audio_file>")
        sys.exit(1)

    filepath = sys.argv[1]

    try:
        # Your implementation here
        pass

    except cm.AudioFileError as e:
        print(f"Audio file error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## See Also

- [Tutorials](../sphinx/tutorials/index.rst) - Step-by-step tutorials
- [Cookbook](../sphinx/cookbook/index.rst) - Recipe collection
- [API Reference](../sphinx/api/index.rst) - Complete API documentation

## License

All examples are provided under the same MIT license as the coremusic project.
