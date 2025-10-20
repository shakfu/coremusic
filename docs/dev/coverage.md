# CoreAudio Python Bindings Coverage

This document tracks the coverage of CoreAudio APIs in the coremusic Python bindings.

## Coverage Summary

### [x] Fully Wrapped (High Coverage)

- **CoreAudio Hardware APIs** - Device management, I/O, properties
- **AudioQueue** - Playback and recording
- **AudioFile** - File I/O operations
- **AudioComponent/AudioUnit** - Component discovery and management
- **AudioConverter** - Format conversion and sample rate conversion
- **ExtendedAudioFile** - High-level file I/O with format conversion
- **AudioFormat** - Channel layout and format utilities
- **CoreAudioTypes** - Basic audio data types

### [!] Partially Wrapped

- **CoreFoundation** - Only basic types and URL creation

### [X] Not Wrapped (High Priority)

- **CoreFoundation Collections** - CFArray, CFDictionary, CFData
- **CoreMIDI** - Complete MIDI support
- **AudioServices** - System sound services

###  Deprecated (Lower Priority)

- **AUGraph** - Audio unit graphs (use AVAudioEngine instead)
- **AudioSession** - iOS audio session (use AVAudioSession instead)

---

## Next Steps

1. **High Priority**: Implement AudioConverter and ExtendedAudioFile
2. **Medium Priority**: Add CoreFoundation data structures
3. **Long Term**: Complete CoreMIDI and AudioServices support
4. **Consider**: Whether to wrap deprecated APIs for legacy support

## Legend

- [x] **Wrapped** - API is fully wrapped and available in Python
- [!] **Partial** - Some functions wrapped but not complete
- [X] **Not Wrapped** - API not yet wrapped
-  **Deprecated** - API is deprecated but may still be useful

---

## CoreFoundation Framework

### Basic Types

- [x] CFBase types (CFIndex, CFAllocatorRef, etc.)
- [x] CFURLRef (basic creation)
- [x] CFStringRef (basic creation)

### Data Structures

- [X] CFArray APIs
- [X] CFDictionary APIs
- [X] CFSet APIs
- [X] CFData APIs
- [X] CFNumber APIs
- [X] CFBag APIs
- [X] CFBinaryHeap APIs
- [X] CFBitVector APIs
- [X] CFTree APIs

### String & Text

- [X] CFString extended APIs
- [X] CFAttributedString APIs
- [X] CFStringTokenizer APIs
- [X] CFStringEncodingExt APIs

### Date & Time

- [X] CFDate APIs
- [X] CFDateFormatter APIs
- [X] CFLocale APIs
- [X] CFTimeZone APIs
- [X] CFCalendar APIs

### File & URL

- [X] CFURL extended APIs
- [X] CFURLAccess APIs
- [X] CFURLEnumerator APIs
- [X] CFFileDescriptor APIs
- [X] CFFileSecurity APIs

### System Services

- [X] CFRunLoop APIs
- [X] CFNotificationCenter APIs
- [X] CFPreferences APIs
- [X] CFBundle APIs
- [X] CFPlugIn APIs
- [X] CFMachPort APIs
- [X] CFMessagePort APIs

### Data & Serialization

- [X] CFPropertyList APIs
- [X] CFError APIs
- [X] CFXMLNode APIs
- [X] CFXMLParser APIs

### Network & Communication

- [X] CFSocket APIs
- [X] CFStream APIs
- [X] CFUserNotification APIs

### Utilities

- [X] CFUtilities APIs
- [X] CFUUID APIs
- [X] CFByteOrder APIs
- [X] CFAvailability APIs

---

## CoreAudio Framework

### AudioHardware

- [x] AudioObjectShow
- [x] AudioObjectHasProperty
- [x] AudioObjectIsPropertySettable
- [x] AudioObjectGetPropertyDataSize
- [x] AudioObjectGetPropertyData
- [x] AudioObjectSetPropertyData
- [x] AudioObjectAddPropertyListener
- [x] AudioObjectRemovePropertyListener
- [x] AudioHardwareUnload
- [x] AudioHardwareCreateAggregateDevice
- [x] AudioHardwareDestroyAggregateDevice
- [x] AudioDeviceCreateIOProcID
- [x] AudioDeviceDestroyIOProcID
- [x] AudioDeviceStart
- [x] AudioDeviceStartAtTime
- [x] AudioDeviceStop
- [x] AudioDeviceGetCurrentTime
- [x] AudioDeviceTranslateTime
- [x] AudioDeviceGetNearestStartTime

### CoreAudioTypes

- [x] SMPTETime structures and enums
- [x] AudioTimeStamp structures and enums
- [x] Basic audio types (Float32, Float64, UInt32, etc.)
- [x] AudioValueTranslation
- [x] AudioValueRange
- [x] AudioBuffer/AudioBufferList
- [x] AudioStreamBasicDescription
- [x] AudioFormatID/AudioFormatFlags

---

## AudioToolbox Framework

### AudioQueue

- [x] AudioQueueNewOutput
- [x] AudioQueueNewInput
- [x] AudioQueueDispose
- [x] AudioQueueAllocateBuffer
- [x] AudioQueueAllocateBufferWithPacketDescriptions
- [x] AudioQueueFreeBuffer
- [x] AudioQueueEnqueueBuffer
- [x] AudioQueueEnqueueBufferWithParameters
- [x] AudioQueueStart
- [x] AudioQueuePrime
- [x] AudioQueueStop
- [x] AudioQueuePause
- [x] AudioQueueFlush
- [x] AudioQueueReset
- [x] AudioQueueGetCurrentTime

### AudioFile

- [x] AudioFileOpenURL
- [x] AudioFileClose
- [x] AudioFileOptimize
- [x] AudioFileReadBytes
- [x] AudioFileWriteBytes
- [x] AudioFileReadPackets
- [x] AudioFileReadPacketData
- [x] AudioFileWritePackets
- [x] AudioFileCountUserData
- [x] AudioFileGetUserDataSize
- [x] AudioFileGetUserData
- [x] AudioFileSetUserData
- [x] AudioFileRemoveUserData
- [x] AudioFileGetPropertyInfo
- [x] AudioFileGetProperty
- [x] AudioFileSetProperty

### AudioComponent/AudioUnit

- [x] AudioComponentFindNext
- [x] AudioComponentCopyName
- [x] AudioComponentGetDescription
- [x] AudioComponentGetVersion
- [x] AudioComponentInstanceNew
- [x] AudioComponentInstanceDispose
- [x] AudioComponentInstanceCanDo
- [x] AudioUnitInitialize
- [x] AudioUnitUninitialize
- [x] AudioUnitGetPropertyInfo
- [x] AudioUnitGetProperty
- [x] AudioUnitSetProperty
- [x] AudioUnitAddPropertyListener
- [x] AudioUnitRemovePropertyListener
- [x] AudioUnitRemovePropertyListenerWithUserData
- [x] AudioUnitAddRenderNotify
- [x] AudioUnitRemoveRenderNotify
- [x] AudioUnitGetParameter
- [x] AudioUnitSetParameter
- [x] AudioUnitScheduleParameters
- [x] AudioUnitRender
- [x] AudioUnitProcess
- [x] AudioUnitProcessMultiple
- [x] AudioUnitReset

### AudioOutputUnit

- [x] AudioOutputUnitStart
- [x] AudioOutputUnitStop

### AudioConverter

- [x] AudioConverterNew
- [x] AudioConverterNewSpecific
- [x] AudioConverterDispose
- [x] AudioConverterReset
- [x] AudioConverterGetPropertyInfo
- [x] AudioConverterGetProperty
- [x] AudioConverterSetProperty
- [x] AudioConverterConvertBuffer
- [x] AudioConverterFillComplexBuffer
- [x] AudioConverterConvertComplexBuffer
- [x] AudioConverterFillBuffer (deprecated)

### AudioFormat

- [x] AudioFormatGetPropertyInfo
- [x] AudioFormatGetProperty
- [x] AudioFormatProperty constants (35+ properties)
- [x] Channel layout utilities
- [x] AudioPanningInfo structure
- [x] AudioBalanceFade structure
- [x] AudioFormatInfo structure
- [x] ExtendedAudioFormatInfo structure
- [x] Panning and balance fade types
- [x] iOS-specific codec types and manufacturers

### AudioFileStream

- [x] AudioFileStreamOpen
- [x] AudioFileStreamClose
- [x] AudioFileStreamParseBytes
- [x] AudioFileStreamSeek
- [x] AudioFileStreamGetPropertyInfo
- [x] AudioFileStreamGetProperty
- [x] AudioFileStreamSetProperty

### ExtendedAudioFile

- [x] ExtAudioFileOpenURL
- [x] ExtAudioFileWrapAudioFileID
- [x] ExtAudioFileCreateWithURL
- [x] ExtAudioFileDispose
- [x] ExtAudioFileRead
- [x] ExtAudioFileWrite
- [x] ExtAudioFileWriteAsync
- [x] ExtAudioFileSeek
- [x] ExtAudioFileTell
- [x] ExtAudioFileGetPropertyInfo
- [x] ExtAudioFileGetProperty
- [x] ExtAudioFileSetProperty
- [x] ExtAudioFileOpen (deprecated)
- [x] ExtAudioFileCreateNew (deprecated)

### AudioServices

- [x] AudioServicesCreateSystemSoundID
- [x] AudioServicesDisposeSystemSoundID
- [x] AudioServicesPlaySystemSound
- [x] AudioServicesPlaySystemSoundWithCompletion
- [x] AudioServicesAddSystemSoundCompletion
- [x] AudioServicesRemoveSystemSoundCompletion
- [x] AudioServicesGetPropertyInfo
- [x] AudioServicesGetProperty
- [x] AudioServicesSetProperty

### MusicDevice

- [x] MusicDeviceMIDIEvent
- [x] MusicDeviceStartNote
- [x] MusicDeviceStopNote
- [x] MusicDeviceSysEx

### MusicPlayer

- [x] NewMusicPlayer
- [x] DisposeMusicPlayer
- [x] MusicPlayerSetSequence
- [x] MusicPlayerGetSequence
- [x] MusicPlayerPreroll
- [x] MusicPlayerStart
- [x] MusicPlayerStop
- [x] MusicPlayerIsPlaying
- [x] MusicPlayerGetTime
- [x] MusicPlayerSetTime
- [x] MusicPlayerSetPlayRateScalar

### AUGraph (Deprecated)

-  NewAUGraph
-  DisposeAUGraph
-  AUGraphAddNode
-  AUGraphRemoveNode
-  AUGraphGetNodeCount
-  AUGraphGetIndNode
-  AUGraphNodeInfo
-  AUGraphConnectNodeInput
-  AUGraphDisconnectNodeInput
-  AUGraphClearConnections
-  AUGraphGetNumberOfConnections
-  AUGraphGetConnectionInfo
-  AUGraphUpdate
-  AUGraphOpen
-  AUGraphClose
-  AUGraphInitialize
-  AUGraphUninitialize
-  AUGraphStart
-  AUGraphStop
-  AUGraphIsRunning
-  AUGraphIsInitialized
-  AUGraphIsOpen

### AudioSession (iOS - Deprecated)

-  AudioSessionInitialize
-  AudioSessionSetActive
-  AudioSessionGetProperty
-  AudioSessionSetProperty
-  AudioSessionAddPropertyListener
-  AudioSessionRemovePropertyListener

### Other AudioToolbox APIs

- [X] AudioCodec APIs
- [X] AudioFileComponent APIs
- [X] AudioHardwareService APIs
- [X] CoreAudioClock APIs
- [X] DefaultAudioOutput APIs
- [X] CAShow/CAShowFile (debug utilities)
- [X] CAFFile structures and utilities
- [X] AudioWorkInterval APIs

---

## CoreMIDI Framework

### Core MIDI Services

- [x] MIDIClientCreate
- [x] MIDIClientDispose
- [x] MIDIGetNumberOfDevices
- [x] MIDIGetDevice
- [x] MIDIDeviceAddEntity
- [x] MIDIDeviceRemoveEntity
- [x] MIDIEntityGetNumberOfSources
- [x] MIDIEntityGetNumberOfDestinations
- [x] MIDIEntityGetSource
- [x] MIDIEntityGetDestination
- [x] MIDIEndpointGetEntity
- [x] MIDIEndpointGetDevice

### MIDI I/O

- [x] MIDIInputPortCreate
- [x] MIDIOutputPortCreate
- [x] MIDIPortDispose
- [x] MIDISend
- [x] MIDISendSysex
- [x] MIDIReceived

### MIDI Setup & Configuration

- [x] MIDISetup APIs
- [x] MIDIThruConnection APIs
- [x] MIDIDriver APIs

### MIDI Messages

- [x] MIDIMessage APIs
- [X] MIDIBluetoothConnection APIs
- [X] MIDINetworkSession APIs
- [X] MIDICapabilityInquiry APIs

---

## CoreAudioKit Framework

### Audio Unit UI

- [X] AUGenericView APIs
- [X] AUPannerView APIs
- [X] AUViewController APIs
- [X] AUCustomViewPersistentData APIs

### MIDI UI

- [X] CABTLEMIDIWindowController APIs
- [X] CAInterDeviceAudioViewController APIs
- [X] CANetworkBrowserWindowController APIs

---

## CoreServices Framework

### Component System

- [X] Component APIs
- [X] CoreServices APIs

---
