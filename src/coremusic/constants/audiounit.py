"""AudioUnit constants."""

from enum import IntEnum

__all__ = [
    "AudioUnitProperty",
    "AudioUnitScope",
    "AudioUnitElement",
    "AudioUnitRenderActionFlags",
    "AudioUnitParameterUnit",
]


class AudioUnitProperty(IntEnum):
    """AudioUnit property IDs (AudioUnitPropertyID)"""

    SAMPLE_RATE = 2  # kAudioUnitProperty_SampleRate
    PARAMETER_LIST = 3  # kAudioUnitProperty_ParameterList
    PARAMETER_INFO = 4  # kAudioUnitProperty_ParameterInfo
    STREAM_FORMAT = 8  # kAudioUnitProperty_StreamFormat
    ELEMENT_COUNT = 11  # kAudioUnitProperty_ElementCount
    LATENCY = 12  # kAudioUnitProperty_Latency
    MAXIMUM_FRAMES_PER_SLICE = 14  # kAudioUnitProperty_MaximumFramesPerSlice
    SET_RENDER_CALLBACK = 23  # kAudioUnitProperty_SetRenderCallback
    FACTORY_PRESETS = 24  # kAudioUnitProperty_FactoryPresets
    RENDER_QUALITY = 26  # kAudioUnitProperty_RenderQuality
    HOST_CALLBACKS = 27  # kAudioUnitProperty_HostCallbacks
    IN_PLACE_PROCESSING = 29  # kAudioUnitProperty_InPlaceProcessing
    ELEMENT_NAME = 30  # kAudioUnitProperty_ElementName
    COCOAUI = 31  # kAudioUnitProperty_CocoaUI
    CHANNEL_MAP = 33  # kAudioUnitProperty_ChannelMap
    AUDIO_CHANNEL_LAYOUT = 19  # kAudioUnitProperty_AudioChannelLayout
    TAIL_TIME = 20  # kAudioUnitProperty_TailTime
    BYPASS_EFFECT = 21  # kAudioUnitProperty_BypassEffect
    LAST_RENDER_ERROR = 22  # kAudioUnitProperty_LastRenderError
    SET_EXTERNAL_BUFFER = 15  # kAudioUnitProperty_SetExternalBuffer
    GET_UI_COMPONENT_LIST = 18  # kAudioUnitProperty_GetUIComponentList
    METER_CLIPPING = 1634755187  # 'clip'
    PRESENT_PRESET = 1886547818  # 'pset'
    OFFLINE_RENDER = 55  # kAudioUnitProperty_OfflineRender
    PARAMETER_STRING_FROM_VALUE = 33  # kAudioUnitProperty_ParameterStringFromValue
    PARAMETER_CLUMP_NAME = 34  # kAudioUnitProperty_ParameterClumpName
    CLASS_INFO = 0  # kAudioUnitProperty_ClassInfo (for presets)


class AudioUnitScope(IntEnum):
    """AudioUnit scope identifiers (AudioUnitScope)"""

    GLOBAL = 0  # kAudioUnitScope_Global
    INPUT = 1  # kAudioUnitScope_Input
    OUTPUT = 2  # kAudioUnitScope_Output
    GROUP = 3  # kAudioUnitScope_Group
    PART = 4  # kAudioUnitScope_Part
    NOTE = 5  # kAudioUnitScope_Note
    LAYER = 6  # kAudioUnitScope_Layer
    LAYER_ITEM = 7  # kAudioUnitScope_LayerItem


class AudioUnitElement(IntEnum):
    """Common AudioUnit element indices"""

    OUTPUT = 0  # Output element
    INPUT = 1  # Input element


class AudioUnitRenderActionFlags(IntEnum):
    """AudioUnit render action flags (AudioUnitRenderActionFlags)"""

    PRE_RENDER = 4  # kAudioUnitRenderAction_PreRender
    POST_RENDER = 8  # kAudioUnitRenderAction_PostRender
    OUTPUT_IS_SILENCE = 16  # kAudioUnitRenderAction_OutputIsSilence
    OFFLINE_RENDER = 32  # kAudioOfflineUnitRenderAction_Render
    OFFLINE_COMPLETE = 64  # kAudioOfflineUnitRenderAction_Complete
    OFFLINE_PREFLIGHT = 128  # kAudioOfflineUnitRenderAction_Preflight
    POST_RENDER_ERROR = 256  # kAudioUnitRenderAction_PostRenderError
    DO_NOT_CHECK_RENDER_ARGS = 512  # kAudioUnitRenderAction_DoNotCheckRenderArgs


class AudioUnitParameterUnit(IntEnum):
    """AudioUnit parameter units (AudioUnitParameterUnit)"""

    GENERIC = 0  # kAudioUnitParameterUnit_Generic
    INDEXED = 1  # kAudioUnitParameterUnit_Indexed
    BOOLEAN = 2  # kAudioUnitParameterUnit_Boolean
    PERCENT = 3  # kAudioUnitParameterUnit_Percent
    SECONDS = 4  # kAudioUnitParameterUnit_Seconds
    SAMPLE_FRAMES = 5  # kAudioUnitParameterUnit_SampleFrames
    PHASE = 6  # kAudioUnitParameterUnit_Phase
    RATE = 7  # kAudioUnitParameterUnit_Rate
    HERTZ = 8  # kAudioUnitParameterUnit_Hertz
    CENTS = 9  # kAudioUnitParameterUnit_Cents
    RELATIVE_SEMITONES = 10  # kAudioUnitParameterUnit_RelativeSemiTones
    MIDI_NOTE_NUMBER = 11  # kAudioUnitParameterUnit_MIDINoteNumber
    MIDI_CONTROLLER = 12  # kAudioUnitParameterUnit_MIDIController
    DECIBELS = 13  # kAudioUnitParameterUnit_Decibels
    LINEAR_GAIN = 14  # kAudioUnitParameterUnit_LinearGain
    DEGREES = 15  # kAudioUnitParameterUnit_Degrees
    EQUAL_POWER_CROSSFADE = 16  # kAudioUnitParameterUnit_EqualPowerCrossfade
    MIXER_FADER_CURVE1 = 17  # kAudioUnitParameterUnit_MixerFaderCurve1
    PAN = 18  # kAudioUnitParameterUnit_Pan
    METERS = 19  # kAudioUnitParameterUnit_Meters
    ABSOLUTE_CENTS = 20  # kAudioUnitParameterUnit_AbsoluteCents
    OCTAVES = 21  # kAudioUnitParameterUnit_Octaves
    BPM = 22  # kAudioUnitParameterUnit_BPM
    BEATS = 23  # kAudioUnitParameterUnit_Beats
    MILLISECONDS = 24  # kAudioUnitParameterUnit_Milliseconds
    RATIO = 25  # kAudioUnitParameterUnit_Ratio
