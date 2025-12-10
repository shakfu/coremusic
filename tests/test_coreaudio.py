"""pytest test suite for coremusic wrapper functionality."""

import os
import pytest
import wave
import struct
import time
import coremusic as cm
import coremusic.capi as capi


class TestFourCCConversion:
    """Test FourCC conversion utilities"""

    def test_fourchar_to_int_conversion(self):
        """Test string to int FourCC conversion"""
        test_codes = ["WAVE", "TEXT", "AIFF", "mp4f", "RIFF"]
        for code in test_codes:
            int_val = capi.fourchar_to_int(code)
            assert isinstance(int_val, int)
            assert int_val > 0

    def test_int_to_fourchar_conversion(self):
        """Test int to string FourCC conversion"""
        test_codes = ["WAVE", "TEXT", "AIFF", "mp4f", "RIFF"]
        for code in test_codes:
            int_val = capi.fourchar_to_int(code)
            back_to_str = capi.int_to_fourchar(int_val)
            assert back_to_str == code

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion maintains data integrity"""
        test_codes = ["WAVE", "TEXT", "AIFF", "mp4f", "RIFF", "CAFF", "caff"]
        for code in test_codes:
            int_val = capi.fourchar_to_int(code)
            back_to_str = capi.int_to_fourchar(int_val)
            assert back_to_str == code, (
                f"Roundtrip failed for '{code}': {code} -> {int_val} -> {back_to_str}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
