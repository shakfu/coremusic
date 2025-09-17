# CoreAudio Python Bindings Coverage

This document tracks the coverage of CoreAudio APIs in the coremusic Python bindings.

## Coverage Summary

### ✅ Fully Wrapped (High Coverage)

- **CoreAudio Hardware APIs** - Device management, I/O, properties
- **AudioQueue** - Playback and recording
- **AudioFile** - File I/O operations
- **AudioComponent/AudioUnit** - Component discovery and management
- **AudioConverter** - Format conversion and sample rate conversion
- **ExtendedAudioFile** - High-level file I/O with format conversion
- **AudioFormat** - Channel layout and format utilities
- **CoreAudioTypes** - Basic audio data types

### ⚠️ Partially Wrapped

- **CoreFoundation** - Only basic types and URL creation

### ❌ Not Wrapped (High Priority)

- **CoreFoundation Collections** - CFArray, CFDictionary, CFData
- **CoreMIDI** - Complete MIDI support
- **AudioServices** - System sound services

### 🚫 Deprecated (Lower Priority)

- **AUGraph** - Audio unit graphs (use AVAudioEngine instead)
- **AudioSession** - iOS audio session (use AVAudioSession instead)

---

## Next Steps

1. **High Priority**: Implement AudioConverter and ExtendedAudioFile
2. **Medium Priority**: Add CoreFoundation data structures
3. **Long Term**: Complete CoreMIDI and AudioServices support
4. **Consider**: Whether to wrap deprecated APIs for legacy support

## Legend

- ✅ **Wrapped** - API is fully wrapped and available in Python
- ⚠️ **Partial** - Some functions wrapped but not complete
- ❌ **Not Wrapped** - API not yet wrapped
- 🚫 **Deprecated** - API is deprecated but may still be useful

---

## CoreFoundation Framework

### Basic Types

- ✅ CFBase types (CFIndex, CFAllocatorRef, etc.)
- ✅ CFURLRef (basic creation)
- ✅ CFStringRef (basic creation)

### Data Structures

- ❌ CFArray APIs
- ❌ CFDictionary APIs
- ❌ CFSet APIs
- ❌ CFData APIs
- ❌ CFNumber APIs
- ❌ CFBag APIs
- ❌ CFBinaryHeap APIs
- ❌ CFBitVector APIs
- ❌ CFTree APIs

### String & Text

- ❌ CFString extended APIs
- ❌ CFAttributedString APIs
- ❌ CFStringTokenizer APIs
- ❌ CFStringEncodingExt APIs

### Date & Time

- ❌ CFDate APIs
- ❌ CFDateFormatter APIs
- ❌ CFLocale APIs
- ❌ CFTimeZone APIs
- ❌ CFCalendar APIs

### File & URL

- ❌ CFURL extended APIs
- ❌ CFURLAccess APIs
- ❌ CFURLEnumerator APIs
- ❌ CFFileDescriptor APIs
- ❌ CFFileSecurity APIs

### System Services

- ❌ CFRunLoop APIs
- ❌ CFNotificationCenter APIs
- ❌ CFPreferences APIs
- ❌ CFBundle APIs
- ❌ CFPlugIn APIs
- ❌ CFMachPort APIs
- ❌ CFMessagePort APIs

### Data & Serialization

- ❌ CFPropertyList APIs
- ❌ CFError APIs
- ❌ CFXMLNode APIs
- ❌ CFXMLParser APIs

### Network & Communication

- ❌ CFSocket APIs
- ❌ CFStream APIs
- ❌ CFUserNotification APIs

### Utilities

- ❌ CFUtilities APIs
- ❌ CFUUID APIs
- ❌ CFByteOrder APIs
- ❌ CFAvailability APIs

---

## CoreAudio Framework

### AudioHardware

- ✅ AudioObjectShow
- ✅ AudioObjectHasProperty
- ✅ AudioObjectIsPropertySettable
- ✅ AudioObjectGetPropertyDataSize
- ✅ AudioObjectGetPropertyData
- ✅ AudioObjectSetPropertyData
- ✅ AudioObjectAddPropertyListener
- ✅ AudioObjectRemovePropertyListener
- ✅ AudioHardwareUnload
- ✅ AudioHardwareCreateAggregateDevice
- ✅ AudioHardwareDestroyAggregateDevice
- ✅ AudioDeviceCreateIOProcID
- ✅ AudioDeviceDestroyIOProcID
- ✅ AudioDeviceStart
- ✅ AudioDeviceStartAtTime
- ✅ AudioDeviceStop
- ✅ AudioDeviceGetCurrentTime
- ✅ AudioDeviceTranslateTime
- ✅ AudioDeviceGetNearestStartTime

### CoreAudioTypes

- ✅ SMPTETime structures and enums
- ✅ AudioTimeStamp structures and enums
- ✅ Basic audio types (Float32, Float64, UInt32, etc.)
- ✅ AudioValueTranslation
- ✅ AudioValueRange
- ✅ AudioBuffer/AudioBufferList
- ✅ AudioStreamBasicDescription
- ✅ AudioFormatID/AudioFormatFlags

---

## AudioToolbox Framework

### AudioQueue

- ✅ AudioQueueNewOutput
- ✅ AudioQueueNewInput
- ✅ AudioQueueDispose
- ✅ AudioQueueAllocateBuffer
- ✅ AudioQueueAllocateBufferWithPacketDescriptions
- ✅ AudioQueueFreeBuffer
- ✅ AudioQueueEnqueueBuffer
- ✅ AudioQueueEnqueueBufferWithParameters
- ✅ AudioQueueStart
- ✅ AudioQueuePrime
- ✅ AudioQueueStop
- ✅ AudioQueuePause
- ✅ AudioQueueFlush
- ✅ AudioQueueReset
- ✅ AudioQueueGetCurrentTime

### AudioFile

- ✅ AudioFileOpenURL
- ✅ AudioFileClose
- ✅ AudioFileOptimize
- ✅ AudioFileReadBytes
- ✅ AudioFileWriteBytes
- ✅ AudioFileReadPackets
- ✅ AudioFileReadPacketData
- ✅ AudioFileWritePackets
- ✅ AudioFileCountUserData
- ✅ AudioFileGetUserDataSize
- ✅ AudioFileGetUserData
- ✅ AudioFileSetUserData
- ✅ AudioFileRemoveUserData
- ✅ AudioFileGetPropertyInfo
- ✅ AudioFileGetProperty
- ✅ AudioFileSetProperty

### AudioComponent/AudioUnit

- ✅ AudioComponentFindNext
- ✅ AudioComponentCopyName
- ✅ AudioComponentGetDescription
- ✅ AudioComponentGetVersion
- ✅ AudioComponentInstanceNew
- ✅ AudioComponentInstanceDispose
- ✅ AudioComponentInstanceCanDo
- ✅ AudioUnitInitialize
- ✅ AudioUnitUninitialize
- ✅ AudioUnitGetPropertyInfo
- ✅ AudioUnitGetProperty
- ✅ AudioUnitSetProperty
- ✅ AudioUnitAddPropertyListener
- ✅ AudioUnitRemovePropertyListener
- ✅ AudioUnitRemovePropertyListenerWithUserData
- ✅ AudioUnitAddRenderNotify
- ✅ AudioUnitRemoveRenderNotify
- ✅ AudioUnitGetParameter
- ✅ AudioUnitSetParameter
- ✅ AudioUnitScheduleParameters
- ✅ AudioUnitRender
- ✅ AudioUnitProcess
- ✅ AudioUnitProcessMultiple
- ✅ AudioUnitReset

### AudioOutputUnit

- ✅ AudioOutputUnitStart
- ✅ AudioOutputUnitStop

### AudioConverter

- ✅ AudioConverterNew
- ✅ AudioConverterNewSpecific
- ✅ AudioConverterDispose
- ✅ AudioConverterReset
- ✅ AudioConverterGetPropertyInfo
- ✅ AudioConverterGetProperty
- ✅ AudioConverterSetProperty
- ✅ AudioConverterConvertBuffer
- ✅ AudioConverterFillComplexBuffer
- ✅ AudioConverterConvertComplexBuffer
- ✅ AudioConverterFillBuffer (deprecated)

### AudioFormat

- ✅ AudioFormatGetPropertyInfo
- ✅ AudioFormatGetProperty
- ✅ AudioFormatProperty constants (35+ properties)
- ✅ Channel layout utilities
- ✅ AudioPanningInfo structure
- ✅ AudioBalanceFade structure
- ✅ AudioFormatInfo structure
- ✅ ExtendedAudioFormatInfo structure
- ✅ Panning and balance fade types
- ✅ iOS-specific codec types and manufacturers

### AudioFileStream

- ✅ AudioFileStreamOpen
- ✅ AudioFileStreamClose
- ✅ AudioFileStreamParseBytes
- ✅ AudioFileStreamSeek
- ✅ AudioFileStreamGetPropertyInfo
- ✅ AudioFileStreamGetProperty
- ✅ AudioFileStreamSetProperty

### ExtendedAudioFile

- ✅ ExtAudioFileOpenURL
- ✅ ExtAudioFileWrapAudioFileID
- ✅ ExtAudioFileCreateWithURL
- ✅ ExtAudioFileDispose
- ✅ ExtAudioFileRead
- ✅ ExtAudioFileWrite
- ✅ ExtAudioFileWriteAsync
- ✅ ExtAudioFileSeek
- ✅ ExtAudioFileTell
- ✅ ExtAudioFileGetPropertyInfo
- ✅ ExtAudioFileGetProperty
- ✅ ExtAudioFileSetProperty
- ✅ ExtAudioFileOpen (deprecated)
- ✅ ExtAudioFileCreateNew (deprecated)

### AudioServices

- ✅ AudioServicesCreateSystemSoundID
- ✅ AudioServicesDisposeSystemSoundID
- ✅ AudioServicesPlaySystemSound
- ✅ AudioServicesPlaySystemSoundWithCompletion
- ✅ AudioServicesAddSystemSoundCompletion
- ✅ AudioServicesRemoveSystemSoundCompletion
- ✅ AudioServicesGetPropertyInfo
- ✅ AudioServicesGetProperty
- ✅ AudioServicesSetProperty

### MusicDevice

- ✅ MusicDeviceMIDIEvent
- ✅ MusicDeviceStartNote
- ✅ MusicDeviceStopNote
- ✅ MusicDeviceSysEx

### MusicPlayer

- ✅ NewMusicPlayer
- ✅ DisposeMusicPlayer
- ✅ MusicPlayerSetSequence
- ✅ MusicPlayerGetSequence
- ✅ MusicPlayerPreroll
- ✅ MusicPlayerStart
- ✅ MusicPlayerStop
- ✅ MusicPlayerIsPlaying
- ✅ MusicPlayerGetTime
- ✅ MusicPlayerSetTime
- ✅ MusicPlayerSetPlayRateScalar

### AUGraph (Deprecated)

- 🚫 NewAUGraph
- 🚫 DisposeAUGraph
- 🚫 AUGraphAddNode
- 🚫 AUGraphRemoveNode
- 🚫 AUGraphGetNodeCount
- 🚫 AUGraphGetIndNode
- 🚫 AUGraphNodeInfo
- 🚫 AUGraphConnectNodeInput
- 🚫 AUGraphDisconnectNodeInput
- 🚫 AUGraphClearConnections
- 🚫 AUGraphGetNumberOfConnections
- 🚫 AUGraphGetConnectionInfo
- 🚫 AUGraphUpdate
- 🚫 AUGraphOpen
- 🚫 AUGraphClose
- 🚫 AUGraphInitialize
- 🚫 AUGraphUninitialize
- 🚫 AUGraphStart
- 🚫 AUGraphStop
- 🚫 AUGraphIsRunning
- 🚫 AUGraphIsInitialized
- 🚫 AUGraphIsOpen

### AudioSession (iOS - Deprecated)

- 🚫 AudioSessionInitialize
- 🚫 AudioSessionSetActive
- 🚫 AudioSessionGetProperty
- 🚫 AudioSessionSetProperty
- 🚫 AudioSessionAddPropertyListener
- 🚫 AudioSessionRemovePropertyListener

### Other AudioToolbox APIs

- ❌ AudioCodec APIs
- ❌ AudioFileComponent APIs
- ❌ AudioHardwareService APIs
- ❌ CoreAudioClock APIs
- ❌ DefaultAudioOutput APIs
- ❌ CAShow/CAShowFile (debug utilities)
- ❌ CAFFile structures and utilities
- ❌ AudioWorkInterval APIs

---

## CoreMIDI Framework

### Core MIDI Services

- ✅ MIDIClientCreate
- ✅ MIDIClientDispose
- ✅ MIDIGetNumberOfDevices
- ✅ MIDIGetDevice
- ✅ MIDIDeviceAddEntity
- ✅ MIDIDeviceRemoveEntity
- ✅ MIDIEntityGetNumberOfSources
- ✅ MIDIEntityGetNumberOfDestinations
- ✅ MIDIEntityGetSource
- ✅ MIDIEntityGetDestination
- ✅ MIDIEndpointGetEntity
- ✅ MIDIEndpointGetDevice

### MIDI I/O

- ❌ MIDIInputPortCreate
- ❌ MIDIOutputPortCreate
- ❌ MIDIPortDispose
- ❌ MIDISend
- ❌ MIDISendSysex
- ❌ MIDIReceived
- ❌ MIDIReceivedMultiple

### MIDI Setup & Configuration

- ❌ MIDISetup APIs
- ❌ MIDIThruConnection APIs
- ❌ MIDIDriver APIs

### MIDI Messages

- ❌ MIDIMessage APIs
- ❌ MIDIBluetoothConnection APIs
- ❌ MIDINetworkSession APIs
- ❌ MIDICapabilityInquiry APIs

---

## CoreAudioKit Framework

### Audio Unit UI

- ❌ AUGenericView APIs
- ❌ AUPannerView APIs
- ❌ AUViewController APIs
- ❌ AUCustomViewPersistentData APIs

### MIDI UI

- ❌ CABTLEMIDIWindowController APIs
- ❌ CAInterDeviceAudioViewController APIs
- ❌ CANetworkBrowserWindowController APIs

---

## CoreServices Framework

### Component System

- ❌ Component APIs
- ❌ CoreServices APIs

---
