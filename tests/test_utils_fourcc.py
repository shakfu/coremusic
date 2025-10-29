#!/usr/bin/env python3
"""Tests for FourCC utilities"""

import pytest
import tempfile
import os
from pathlib import Path

from coremusic.utils.fourcc import (
	ensure_fourcc_int,
	ensure_fourcc_str,
	fourcc_to_int,
	fourcc_to_str,
	FourCCValue,
)



# ============================================================================
# FourCC Utilities Tests
# ============================================================================

class TestFourCCUtilities:
    """Test FourCC utility functions and classes"""

    def test_ensure_fourcc_int_from_string(self):
        """Test ensure_fourcc_int with string input"""
        result = ensure_fourcc_int('lpcm')
        assert isinstance(result, int)
        assert result == 1819304813

    def test_ensure_fourcc_int_from_int(self):
        """Test ensure_fourcc_int with int input"""
        result = ensure_fourcc_int(1819304813)
        assert isinstance(result, int)
        assert result == 1819304813

    def test_ensure_fourcc_int_invalid_string_length(self):
        """Test ensure_fourcc_int with invalid string length"""
        with pytest.raises(ValueError):
            ensure_fourcc_int('abc')  # Too short

        with pytest.raises(ValueError):
            ensure_fourcc_int('abcdef')  # Too long

    def test_ensure_fourcc_int_invalid_type(self):
        """Test ensure_fourcc_int with invalid type"""
        with pytest.raises(TypeError):
            ensure_fourcc_int(12.5)  # Float not allowed

        with pytest.raises(TypeError):
            ensure_fourcc_int(None)

    def test_ensure_fourcc_str_from_int(self):
        """Test ensure_fourcc_str with int input"""
        result = ensure_fourcc_str(1819304813)
        assert isinstance(result, str)
        assert result == 'lpcm'
        assert len(result) == 4

    def test_ensure_fourcc_str_from_string(self):
        """Test ensure_fourcc_str with string input"""
        result = ensure_fourcc_str('lpcm')
        assert isinstance(result, str)
        assert result == 'lpcm'

    def test_ensure_fourcc_str_invalid_type(self):
        """Test ensure_fourcc_str with invalid type"""
        with pytest.raises(TypeError):
            ensure_fourcc_str(12.5)

    def test_fourcc_to_int(self):
        """Test fourcc_to_int conversion"""
        assert fourcc_to_int('lpcm') == 1819304813
        assert fourcc_to_int('aac ') == 1633772320
        assert fourcc_to_int('WAVE') == 1463899717

    def test_fourcc_to_str(self):
        """Test fourcc_to_str conversion"""
        assert fourcc_to_str(1819304813) == 'lpcm'
        assert fourcc_to_str(1633772320) == 'aac '
        assert fourcc_to_str(1463899717) == 'WAVE'

    def test_fourcc_value_class_from_string(self):
        """Test FourCCValue class with string input"""
        fourcc = FourCCValue('lpcm')
        assert str(fourcc) == 'lpcm'
        assert int(fourcc) == 1819304813
        assert fourcc.str_value == 'lpcm'
        assert fourcc.int_value == 1819304813

    def test_fourcc_value_class_from_int(self):
        """Test FourCCValue class with int input"""
        fourcc = FourCCValue(1819304813)
        assert str(fourcc) == 'lpcm'
        assert int(fourcc) == 1819304813

    def test_fourcc_value_equality(self):
        """Test FourCCValue equality comparisons"""
        fourcc1 = FourCCValue('lpcm')
        fourcc2 = FourCCValue(1819304813)
        fourcc3 = FourCCValue('aac ')

        # Equal FourCCValue objects
        assert fourcc1 == fourcc2

        # Compare with string
        assert fourcc1 == 'lpcm'

        # Compare with int
        assert fourcc1 == 1819304813

        # Not equal
        assert fourcc1 != fourcc3
        assert fourcc1 != 'aac '

    def test_fourcc_value_hash(self):
        """Test FourCCValue can be hashed"""
        fourcc1 = FourCCValue('lpcm')
        fourcc2 = FourCCValue(1819304813)

        # Can be used in sets
        fourcc_set = {fourcc1, fourcc2}
        assert len(fourcc_set) == 1  # Same FourCC

        # Can be used as dict keys
        fourcc_dict = {fourcc1: "PCM"}
        assert fourcc_dict[fourcc2] == "PCM"

    def test_fourcc_value_format(self):
        """Test FourCCValue formatting"""
        fourcc = FourCCValue('lpcm')

        # Default format (string)
        assert f"{fourcc}" == 'lpcm'
        assert f"{fourcc:s}" == 'lpcm'

        # Integer formats
        assert f"{fourcc:d}" == '1819304813'
        assert f"{fourcc:08X}" == '6C70636D'

    def test_fourcc_value_repr(self):
        """Test FourCCValue repr"""
        fourcc = FourCCValue('lpcm')
        repr_str = repr(fourcc)
        assert 'FourCCValue' in repr_str
        assert 'lpcm' in repr_str
        assert '6C70636D' in repr_str

