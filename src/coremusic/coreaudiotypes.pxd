# coreaudiotypes.pxd
# CoreAudio and CoreAudioTypes framework declarations for coremusic

from .corefoundation cimport *

# -----------------------------------------------------------------------------

cdef extern from "CoreAudioTypes/CoreAudioBaseTypes.h":
    ctypedef UInt32 AudioChannelLayoutTag
    ctypedef UInt32 AudioChannelLabel

    ctypedef enum AudioChannelFlags:
        kAudioChannelFlags_AllOff                   = 0
        kAudioChannelFlags_RectangularCoordinates   = 1
        kAudioChannelFlags_SphericalCoordinates     = 2
        kAudioChannelFlags_Meters                   = 4

    ctypedef enum AudioChannelBitmap:
        kAudioChannelBit_Left                       = 1
        kAudioChannelBit_Right                      = 2
        kAudioChannelBit_Center                     = 4
        kAudioChannelBit_LFEScreen                  = 8
        kAudioChannelBit_LeftSurround               = 16
        kAudioChannelBit_RightSurround              = 32
        kAudioChannelBit_LeftCenter                 = 64
        kAudioChannelBit_RightCenter                = 128
        kAudioChannelBit_CenterSurround             = 256      # WAVE: "Back Center"
        kAudioChannelBit_LeftSurroundDirect         = 512
        kAudioChannelBit_RightSurroundDirect        = 1024
        kAudioChannelBit_TopCenterSurround          = 2048
        kAudioChannelBit_VerticalHeightLeft         = 4096     # WAVE: "Top Front Left"
        kAudioChannelBit_VerticalHeightCenter       = 8192     # WAVE: "Top Front Center"
        kAudioChannelBit_VerticalHeightRight        = 16384    # WAVE: "Top Front Right"
        kAudioChannelBit_TopBackLeft                = 32768
        kAudioChannelBit_TopBackCenter              = 65536
        kAudioChannelBit_TopBackRight               = 131072
        kAudioChannelBit_LeftTopFront               = 4096  # 262144
        kAudioChannelBit_CenterTopFront             = 8192  # 524288
        kAudioChannelBit_RightTopFront              = 16384 # 1048576
        kAudioChannelBit_LeftTopMiddle              = 2097152
        kAudioChannelBit_CenterTopMiddle            = 2048  # 4194304
        kAudioChannelBit_RightTopMiddle             = 8388608
        kAudioChannelBit_LeftTopRear                = 16777216
        kAudioChannelBit_CenterTopRear              = 33554432
        kAudioChannelBit_RightTopRear               = 67108864

    ctypedef struct AudioChannelDescription:
        AudioChannelLabel   mChannelLabel
        AudioChannelFlags   mChannelFlags
        Float32             mCoordinates[3]

    ctypedef struct  AudioStreamPacketDescription:
        SInt64  mStartOffset
        UInt32  mVariableFramesInPacket
        UInt32  mDataByteSize

    ctypedef struct AudioClassDescription:
        OSType  mType
        OSType  mSubType
        OSType  mManufacturer

    ctypedef struct AudioChannelLayout:
        AudioChannelLayoutTag       mChannelLayoutTag
        AudioChannelBitmap          mChannelBitmap
        UInt32                      mNumberChannelDescriptions
        # this is a variable length array of mNumberChannelDescriptions elements
        AudioChannelDescription     mChannelDescriptions[1]

