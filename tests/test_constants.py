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
    """Test constant Enum classes behavior (not hardcoded values)"""

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
