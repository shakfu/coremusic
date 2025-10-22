# AudioUnit Host Implementation Plan

## Overview

Implement comprehensive AudioUnit plugin hosting capabilities in CoreMusic, enabling users to:
- Discover and enumerate available AudioUnit plugins
- Instantiate and manage plugin instances
- Control plugin parameters with full automation
- Save and load plugin presets
- Route audio through plugin chains
- Integrate with Ableton Link for tempo synchronization

## Architecture

### Component Hierarchy

```
AudioUnitHost (Python high-level API)
├── AudioUnitPlugin (per-plugin instance)
│   ├── Parameters (discovery and control)
│   ├── Presets (save/load)
│   └── Audio I/O (routing)
└── AudioUnitChain (multi-plugin routing)
```

### Core Classes

#### 1. **AudioUnitHost** - Main host controller
- Plugin discovery and enumeration
- System-wide plugin scanning
- Category-based filtering
- Manufacturer filtering

#### 2. **AudioUnitPlugin** - Individual plugin instance
- Plugin lifecycle (instantiate, initialize, uninitialize, dispose)
- Parameter discovery and control
- Preset management
- Stream format configuration
- Real-time audio processing

#### 3. **AudioUnitParameter** - Plugin parameter wrapper
- Parameter metadata (name, unit, min/max, default)
- Value get/set with validation
- String value formatting
- Automation support

#### 4. **AudioUnitPreset** - Preset management
- Factory presets enumeration
- User preset save/load
- ClassInfo serialization/deserialization
- Preset import/export

#### 5. **AudioUnitChain** - Multi-plugin routing
- Serial plugin chains
- Audio buffer management
- Bypass control
- Wet/dry mix

## Implementation Phases

### Phase 1: Foundation (Days 1-2)
**Goal**: Add missing C API declarations and basic plugin discovery

**Tasks**:
1. Add AudioUnitParameterInfo structure to `audiotoolbox.pxd`
2. Add AUPreset structure
3. Add missing property constants
4. Implement plugin discovery functions in `capi.pyx`
5. Create basic test for plugin enumeration

**Deliverables**:
- Extended `audiotoolbox.pxd` with parameter structures
- Python functions: `audio_unit_get_plugins()`, `audio_unit_get_plugin_info()`
- Test: List all available AudioUnit plugins

### Phase 2: Plugin Instantiation & Lifecycle (Days 3-4)
**Goal**: Manage plugin instances with proper lifecycle

**Tasks**:
1. Create `AudioUnitPlugin` class in `audio_unit_host.py`
2. Implement instantiation from component description
3. Implement initialization/uninitialization
4. Add stream format configuration
5. Add proper cleanup and disposal
6. Create tests for plugin lifecycle

**Deliverables**:
- `AudioUnitPlugin` class with context manager support
- Proper resource management
- Tests for plugin creation and disposal

### Phase 3: Parameter Discovery & Control (Days 5-7)
**Goal**: Full parameter automation support

**Tasks**:
1. Implement parameter list enumeration
2. Implement parameter info retrieval
3. Create `AudioUnitParameter` class
4. Add parameter get/set functions
5. Add parameter value string formatting
6. Add parameter value validation
7. Create comprehensive parameter tests

**Deliverables**:
- Complete parameter discovery API
- Parameter automation with validation
- Tests for parameter control

### Phase 4: Preset Management (Days 8-9)
**Goal**: Save and load plugin presets

**Tasks**:
1. Implement factory preset enumeration
2. Implement preset selection
3. Implement ClassInfo get/set for user presets
4. Add preset file I/O
5. Create preset management tests

**Deliverables**:
- Factory preset browsing
- User preset save/load
- Preset import/export
- Tests for preset functionality

### Phase 5: Audio Routing (Days 10-12)
**Goal**: Process audio through plugins

**Tasks**:
1. Implement audio buffer allocation
2. Implement render callback setup
3. Create `AudioUnitChain` class
4. Implement serial routing
5. Add bypass functionality
6. Create audio processing tests

**Deliverables**:
- Audio processing through plugins
- Multi-plugin chains
- Tests with actual audio processing

### Phase 6: Link Integration (Days 13-14)
**Goal**: Tempo-synchronized plugins

**Tasks**:
1. Add tempo sync callbacks
2. Implement beat/time conversion for plugins
3. Add host callback structure
4. Test with tempo-dependent plugins (delays, LFOs)

**Deliverables**:
- Link tempo synchronization
- Host callbacks for tempo info
- Tests with tempo-synced plugins

### Phase 7: High-Level API & Documentation (Days 15-16)
**Goal**: Pythonic API and comprehensive docs

**Tasks**:
1. Create high-level Python API
2. Add convenience methods
3. Create demo applications
4. Write comprehensive documentation
5. Create tutorial examples

**Deliverables**:
- Pythonic `AudioUnitHost` API
- Multiple demo applications
- Complete documentation
- Tutorial examples

## API Design

### Low-Level Functional API

```python
import coremusic.capi as capi

# Discover plugins
plugins = capi.audio_unit_get_plugins(type='aufx')  # Effects

for plugin in plugins:
    info = capi.audio_unit_get_plugin_info(plugin)
    print(f"{info['name']} by {info['manufacturer']}")

# Create plugin instance
component = capi.audio_component_find_next(description)
unit = capi.audio_component_instance_new(component)
capi.audio_unit_initialize(unit)

# Get parameters
param_list = capi.audio_unit_get_parameter_list(unit)
for param_id in param_list:
    info = capi.audio_unit_get_parameter_info(unit, param_id)
    print(f"Parameter: {info['name']}, Range: {info['min']}-{info['max']}")

# Set parameter
capi.audio_unit_set_parameter(unit, param_id, value, scope, element)

# Get factory presets
presets = capi.audio_unit_get_factory_presets(unit)
capi.audio_unit_set_current_preset(unit, presets[0])

# Cleanup
capi.audio_unit_uninitialize(unit)
capi.audio_component_instance_dispose(unit)
```

### High-Level Object-Oriented API

```python
import coremusic as cm

# Discover plugins
host = cm.AudioUnitHost()
effects = host.discover_plugins(type='effect')

print(f"Found {len(effects)} effect plugins:")
for plugin_info in effects[:10]:
    print(f"  - {plugin_info.name} ({plugin_info.manufacturer})")

# Load a plugin
with cm.AudioUnitPlugin.from_name("AUDelay") as plugin:
    print(f"Loaded: {plugin.name}")
    print(f"Manufacturer: {plugin.manufacturer}")
    print(f"Version: {plugin.version}")

    # List parameters
    print(f"\nParameters ({len(plugin.parameters)}):")
    for param in plugin.parameters:
        print(f"  - {param.name}: {param.value} {param.unit}")
        print(f"    Range: {param.min} to {param.max}")

    # Set parameter by name
    plugin.set_parameter("Delay Time", 0.5)  # 500ms
    plugin.set_parameter("Feedback", 0.3)
    plugin.set_parameter("Wet/Dry Mix", 1.0)

    # Or use dictionary access
    plugin['Delay Time'] = 0.25
    print(f"Delay: {plugin['Delay Time']}")

    # List factory presets
    print(f"\nFactory Presets ({len(plugin.factory_presets)}):")
    for preset in plugin.factory_presets:
        print(f"  - {preset.name}")

    # Apply preset
    plugin.load_preset(plugin.factory_presets[0])

    # Save user preset
    plugin.save_preset("/path/to/my-preset.aupreset")

    # Process audio
    output = plugin.process(input_audio, num_frames=512)

# Create plugin chain
with cm.AudioUnitChain() as chain:
    # Add plugins to chain
    chain.add_plugin("AUHighpass", cutoff=200)
    chain.add_plugin("AUDelay", delay_time=0.5, feedback=0.3)
    chain.add_plugin("AUReverb", room_type="Large Hall")

    # Process audio through chain
    output = chain.process(input_audio)

    # Bypass individual plugins
    chain[1].bypass = True

    # Get plugin by index
    reverb = chain[2]
    reverb['Room Size'] = 0.8

# Link integration
with cm.link.LinkSession(bpm=120.0) as session:
    with cm.AudioUnitPlugin.from_name("AUDelay", link_session=session) as delay:
        # Delay automatically syncs to Link tempo
        delay['Delay Time (Beats)'] = 0.25  # Quarter note delay

        # Plugin receives tempo changes from Link
        print(f"Plugin tempo: {delay.tempo} BPM")
```

## Required CoreAudio Structures

### AudioUnitParameterInfo
```c
typedef struct AudioUnitParameterInfo {
    char name[52];
    CFStringRef unitName;
    UInt32 clumpID;
    CFStringRef cfNameString;
    AudioUnitParameterUnit unit;
    AudioUnitParameterValue minValue;
    AudioUnitParameterValue maxValue;
    AudioUnitParameterValue defaultValue;
    UInt32 flags;
} AudioUnitParameterInfo;
```

### AUPreset
```c
typedef struct AUPreset {
    SInt32 presetNumber;
    CFStringRef presetName;
} AUPreset;
```

### HostCallbackInfo
```c
typedef struct HostCallbackInfo {
    void* hostUserData;
    OSStatus (*beatAndTempoProc)(void* inHostUserData, Float64* outCurrentBeat, Float64* outCurrentTempo);
    OSStatus (*musicalTimeLocationProc)(void* inHostUserData, UInt32* outDeltaSampleOffsetToNextBeat, Float32* outTimeSig_Numerator, UInt32* outTimeSig_Denominator, Float64* outCurrentMeasureDownBeat);
    OSStatus (*transportStateProc)(void* inHostUserData, Boolean* outIsPlaying, Boolean* outIsRecording, Float64* outCurrentSampleInTimeLine, Boolean* outIsCycling, Float64* outCycleStartBeat, Float64* outCycleEndBeat);
} HostCallbackInfo;
```

## Testing Strategy

### Unit Tests
1. **Plugin Discovery**
   - Test plugin enumeration
   - Test filtering by type
   - Test filtering by manufacturer

2. **Plugin Lifecycle**
   - Test instantiation
   - Test initialization
   - Test proper cleanup

3. **Parameters**
   - Test parameter discovery
   - Test parameter get/set
   - Test parameter validation
   - Test parameter strings

4. **Presets**
   - Test factory preset listing
   - Test preset application
   - Test preset save/load

5. **Audio Processing**
   - Test audio routing
   - Test buffer management
   - Test plugin chains

6. **Link Integration**
   - Test tempo synchronization
   - Test beat callbacks
   - Test transport state

### Integration Tests
1. **Complete Plugin Hosting**
   - Load plugin, set parameters, process audio
   - Save preset, load preset, verify state
   - Chain multiple plugins

2. **Real-World Scenarios**
   - Reverb + delay chain
   - EQ + compressor + limiter chain
   - Instrument plugin with effects

## Demo Applications

### 1. Plugin Browser
- List all available plugins
- Show plugin information
- Test plugin instantiation

### 2. Simple Plugin Host
- Load a single plugin
- Control parameters interactively
- Process audio file

### 3. Effects Chain
- Build multi-plugin chain
- Process audio through chain
- Save/load chain configuration

### 4. Link-Synced Delay
- Tempo-synced delay effect
- Synchronize with Link session
- Demonstrate host callbacks

## Documentation

### User Guides
1. **AudioUnit Host Basics**
   - Introduction to AudioUnit hosting
   - Plugin discovery
   - Basic plugin usage

2. **Parameter Control**
   - Parameter discovery
   - Parameter automation
   - Parameter value types

3. **Preset Management**
   - Factory presets
   - User presets
   - Preset import/export

4. **Audio Routing**
   - Single plugin processing
   - Plugin chains
   - Buffer management

5. **Link Integration**
   - Tempo synchronization
   - Beat callbacks
   - Transport state

### API Reference
- Complete class and method documentation
- Parameter types and ranges
- Error handling
- Best practices

## Success Criteria

✅ **Phase 1 Complete**: Can discover and list all AudioUnit plugins
✅ **Phase 2 Complete**: Can instantiate and manage plugin lifecycle
✅ **Phase 3 Complete**: Can discover and control all plugin parameters
✅ **Phase 4 Complete**: Can save and load presets
✅ **Phase 5 Complete**: Can process audio through plugins and chains
✅ **Phase 6 Complete**: Plugins sync to Link tempo
✅ **Phase 7 Complete**: Full documentation and demos available

## Timeline

- **Week 1** (Days 1-7): Foundation, instantiation, parameters
- **Week 2** (Days 8-14): Presets, routing, Link integration
- **Week 3** (Days 15-16): Documentation and polish

**Total estimated time**: 16 days (~2-3 weeks with testing and refinement)

## Notes

- Prioritize stability and proper resource management
- Test with both Apple and third-party AudioUnits
- Ensure all memory is properly allocated/deallocated
- Handle all error cases gracefully
- Provide comprehensive examples for common use cases
