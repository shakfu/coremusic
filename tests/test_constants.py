#!/usr/bin/env python3
"""Tests for Constant Enum classes"""

import pytest
import tempfile
import os
from pathlib import Path

import coremusic as cm
from coremusic import constants


# ============================================================================
# Constant Enum Classes Tests
# ============================================================================

class TestConstantEnums:
    """Test constant Enum classes"""

    def test_audio_file_property_enum(self):
        """Test AudioFileProperty enum"""
        assert constants.AudioFileProperty.DATA_FORMAT == 1684434292  # 'dfmt'
        assert constants.AudioFileProperty.FILE_FORMAT == 1717988724
        assert constants.AudioFileProperty.MAXIMUM_PACKET_SIZE == 1886616165

        # Can convert to int
        prop = constants.AudioFileProperty.DATA_FORMAT
        assert int(prop) == 1684434292

        # Can compare with int
        assert prop == 1684434292

    def test_audio_format_id_enum(self):
        """Test AudioFormatID enum"""
        assert constants.AudioFormatID.LINEAR_PCM == 1819304813
        assert constants.AudioFormatID.MPEG4_AAC == 1633772320
        assert constants.AudioFormatID.APPLE_LOSSLESS == 1634492771

    def test_linear_pcm_format_flag_enum(self):
        """Test LinearPCMFormatFlag enum"""
        assert constants.LinearPCMFormatFlag.IS_FLOAT == 1
        assert constants.LinearPCMFormatFlag.IS_BIG_ENDIAN == 2
        assert constants.LinearPCMFormatFlag.IS_SIGNED_INTEGER == 4
        assert constants.LinearPCMFormatFlag.IS_PACKED == 8

        # Test flag combination
        flags = (constants.LinearPCMFormatFlag.IS_FLOAT |
                 constants.LinearPCMFormatFlag.IS_PACKED)
        assert flags == 9

    def test_audio_converter_quality_enum(self):
        """Test AudioConverterQuality enum"""
        assert constants.AudioConverterQuality.MAX == 127
        assert constants.AudioConverterQuality.HIGH == 96
        assert constants.AudioConverterQuality.MEDIUM == 64
        assert constants.AudioConverterQuality.LOW == 32
        assert constants.AudioConverterQuality.MIN == 0

    def test_audio_unit_property_enum(self):
        """Test AudioUnitProperty enum"""
        assert constants.AudioUnitProperty.SAMPLE_RATE == 2
        assert constants.AudioUnitProperty.STREAM_FORMAT == 8
        assert constants.AudioUnitProperty.LATENCY == 12
        assert constants.AudioUnitProperty.MAXIMUM_FRAMES_PER_SLICE == 14

    def test_audio_unit_scope_enum(self):
        """Test AudioUnitScope enum"""
        assert constants.AudioUnitScope.GLOBAL == 0
        assert constants.AudioUnitScope.INPUT == 1
        assert constants.AudioUnitScope.OUTPUT == 2
        assert constants.AudioUnitScope.GROUP == 3

    def test_midi_status_enum(self):
        """Test MIDIStatus enum"""
        assert constants.MIDIStatus.NOTE_OFF == 0x80
        assert constants.MIDIStatus.NOTE_ON == 0x90
        assert constants.MIDIStatus.CONTROL_CHANGE == 0xB0
        assert constants.MIDIStatus.PROGRAM_CHANGE == 0xC0
        assert constants.MIDIStatus.PITCH_BEND == 0xE0

    def test_midi_control_change_enum(self):
        """Test MIDIControlChange enum"""
        assert constants.MIDIControlChange.VOLUME == 7
        assert constants.MIDIControlChange.PAN == 10
        assert constants.MIDIControlChange.EXPRESSION == 11
        assert constants.MIDIControlChange.SUSTAIN_PEDAL == 64
        assert constants.MIDIControlChange.ALL_NOTES_OFF == 123

    def test_enum_can_be_used_in_comparisons(self):
        """Test enums can be used in comparisons"""
        prop = constants.AudioFileProperty.DATA_FORMAT

        # Compare with another enum
        assert prop == constants.AudioFileProperty.DATA_FORMAT
        assert prop != constants.AudioFileProperty.FILE_FORMAT

        # Compare with int
        assert prop == 1684434292
        assert prop != 1717988724

    def test_enum_can_be_used_in_sets_and_dicts(self):
        """Test enums can be used in sets and dicts"""
        # Sets
        props = {
            constants.AudioFileProperty.DATA_FORMAT,
            constants.AudioFileProperty.FILE_FORMAT,
            constants.AudioFileProperty.DATA_FORMAT,  # Duplicate
        }
        assert len(props) == 2

        # Dicts
        prop_map = {
            constants.AudioFileProperty.DATA_FORMAT: "Audio format",
            constants.AudioFileProperty.FILE_FORMAT: "File format",
        }
        assert len(prop_map) == 2
        assert prop_map[constants.AudioFileProperty.DATA_FORMAT] == "Audio format"

    def test_backward_compatibility_with_getters(self):
        """Test that enum values match getter function values"""
        # Import capi module
        from coremusic import capi

        # Test a few key constants match between enum and getter
        assert (constants.AudioFileProperty.DATA_FORMAT ==
                capi.get_audio_file_property_data_format())

        assert (constants.AudioUnitScope.GLOBAL ==
                capi.get_audio_unit_scope_global())

        assert (constants.AudioUnitScope.INPUT ==
                capi.get_audio_unit_scope_input())

        assert (constants.AudioUnitScope.OUTPUT ==
                capi.get_audio_unit_scope_output())

    def test_enum_iteration(self):
        """Test that enums can be iterated"""
        # Get all audio file properties
        all_props = list(constants.AudioFileProperty)
        assert len(all_props) > 0
        assert all(isinstance(p, constants.AudioFileProperty) for p in all_props)

        # Get all MIDI status bytes
        all_status = list(constants.MIDIStatus)
        assert len(all_status) == 8  # 7 channel messages + system

    def test_enum_name_access(self):
        """Test enum member name access"""
        prop = constants.AudioFileProperty.DATA_FORMAT
        assert prop.name == 'DATA_FORMAT'
        assert prop.value == 1684434292

    def test_enum_value_lookup(self):
        """Test looking up enum by value"""
        prop = constants.AudioFileProperty(1684434292)
        assert prop == constants.AudioFileProperty.DATA_FORMAT
        assert prop.name == 'DATA_FORMAT'
