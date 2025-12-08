#!/usr/bin/env python3
"""FourCC (Four-Character Code) utilities for CoreMusic.

FourCC codes are 32-bit identifiers used throughout CoreAudio APIs to identify
formats, properties, and other entities. They can be represented as either:
- 4-character strings (e.g., 'lpcm', 'aac ', 'dfmt')
- 32-bit integers (e.g., 1819304813, 1633969526)

This module provides utilities to convert between representations and work
with FourCC values consistently.
"""

from typing import Union

# Type alias for FourCC values
FourCC = Union[str, int]

__all__ = [
    "FourCC",
    "ensure_fourcc_int",
    "ensure_fourcc_str",
    "FourCCValue",
    "fourcc_to_str",
    "fourcc_to_int",
]


def ensure_fourcc_int(value: FourCC) -> int:
    """Ensure FourCC value is in integer form.

    Args:
        value: FourCC as string or integer

    Returns:
        FourCC as 32-bit integer

    Raises:
        ValueError: If string is not exactly 4 characters
        TypeError: If value is neither string nor integer

    Examples:
        >>> ensure_fourcc_int('lpcm')
        1819304813
        >>> ensure_fourcc_int(1819304813)
        1819304813
    """
    if isinstance(value, int):
        return value
    elif isinstance(value, str):
        return fourcc_to_int(value)
    else:
        raise TypeError(f"FourCC must be str or int, not {type(value).__name__}")


def ensure_fourcc_str(value: FourCC) -> str:
    """Ensure FourCC value is in string form.

    Args:
        value: FourCC as string or integer

    Returns:
        FourCC as 4-character string

    Raises:
        TypeError: If value is neither string nor integer

    Examples:
        >>> ensure_fourcc_str(1819304813)
        'lpcm'
        >>> ensure_fourcc_str('lpcm')
        'lpcm'
    """
    if isinstance(value, str):
        if len(value) != 4:
            raise ValueError(f"FourCC string must be exactly 4 characters, got {len(value)}")
        return value
    elif isinstance(value, int):
        return fourcc_to_str(value)
    else:
        raise TypeError(f"FourCC must be str or int, not {type(value).__name__}")


# def int_to_fourcc(value: int) -> str:
#     """Convert integer to four-character code string.
#     Args:
#         value: 32-bit integer

#     Returns:
#         4-character string representation

#     Examples:
#         >>> int_to_fourcc(1819304813)
#         'lpcm'
#     """
#     if isinstance(value, str) and len(value) == 4:
#         return value
#     chars = [
#         chr((value >> 24) & 0xFF),
#         chr((value >> 16) & 0xFF),
#         chr((value >> 8) & 0xFF),
#         chr(value & 0xFF)
#     ]
#     return ''.join(chars)


def fourcc_to_int(fourcc_str: str) -> int:
    """Convert FourCC string to integer.

    Args:
        fourcc_str: 4-character string

    Returns:
        32-bit integer representation

    Raises:
        ValueError: If string is not exactly 4 characters

    Examples:
        >>> fourcc_to_int('lpcm')
        1819304813
        >>> fourcc_to_int('aac ')
        1633969526
    """
    if len(fourcc_str) != 4:
        raise ValueError(f"FourCC string must be exactly 4 characters, got {len(fourcc_str)}")

    # Convert to big-endian 32-bit integer
    return int.from_bytes(fourcc_str.encode('latin-1'), byteorder='big')


def fourcc_to_str(fourcc_int: int) -> str:
    """Convert FourCC integer to string.

    Args:
        fourcc_int: 32-bit integer

    Returns:
        4-character string representation

    Raises:
        ValueError: If integer is out of range for 32-bit value

    Examples:
        >>> fourcc_to_str(1819304813)
        'lpcm'
        >>> fourcc_to_str(1633969526)
        'aac '
    """
    if fourcc_int < 0 or fourcc_int > 0xFFFFFFFF:
        raise ValueError(f"FourCC integer must be in range 0-4294967295, got {fourcc_int}")

    # Convert from big-endian 32-bit integer
    return fourcc_int.to_bytes(4, byteorder='big').decode('latin-1')


class FourCCValue:
    """FourCC value that can be used as both string and integer.

    This class provides a dual representation of FourCC codes, allowing
    seamless conversion between string and integer forms. It's useful when
    you need to pass FourCC values to APIs that accept either form.

    Attributes:
        str_value: 4-character string representation
        int_value: 32-bit integer representation

    Examples:
        >>> fourcc = FourCCValue('lpcm')
        >>> print(fourcc)
        'lpcm'
        >>> int(fourcc)
        1819304813
        >>> fourcc == 'lpcm'
        True
        >>> fourcc == 1819304813
        True

        >>> fourcc = FourCCValue(1633969526)
        >>> str(fourcc)
        'aac '
        >>> int(fourcc)
        1633969526
    """

    def __init__(self, value: FourCC):
        """Initialize FourCC value.

        Args:
            value: FourCC as string or integer

        Raises:
            ValueError: If string is not exactly 4 characters
            TypeError: If value is neither string nor integer
        """
        if isinstance(value, str):
            if len(value) != 4:
                raise ValueError(f"FourCC string must be exactly 4 characters, got {len(value)}")
            self._str = value
            self._int = fourcc_to_int(value)
        elif isinstance(value, int):
            self._int = value
            self._str = fourcc_to_str(value)
        else:
            raise TypeError(f"FourCC must be str or int, not {type(value).__name__}")

    @property
    def str_value(self) -> str:
        """Get string representation."""
        return self._str

    @property
    def int_value(self) -> int:
        """Get integer representation."""
        return self._int

    def __str__(self) -> str:
        """Return string representation."""
        return self._str

    def __int__(self) -> int:
        """Return integer representation."""
        return self._int

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"FourCCValue('{self._str}', 0x{self._int:08X})"

    def __eq__(self, other) -> bool:
        """Compare with another FourCC value (string or integer)."""
        if isinstance(other, FourCCValue):
            return self._int == other._int
        elif isinstance(other, str):
            return self._str == other
        elif isinstance(other, int):
            return self._int == other
        return False

    def __hash__(self) -> int:
        """Return hash for use in sets/dicts."""
        return hash(self._int)

    def __format__(self, format_spec: str) -> str:
        """Format the FourCC value.

        Format spec can be:
        - 's' or empty: return string representation
        - 'd', 'x', 'X', etc.: return integer in that format

        Examples:
            >>> fourcc = FourCCValue('lpcm')
            >>> f"{fourcc}"
            'lpcm'
            >>> f"{fourcc:s}"
            'lpcm'
            >>> f"{fourcc:d}"
            '1819304813'
            >>> f"{fourcc:08X}"
            '6C70636D'
        """
        if not format_spec or format_spec == 's':
            return self._str
        else:
            # Treat as integer format
            return format(self._int, format_spec)


# Convenience function for backward compatibility with capi module
def convert_fourcc(value: FourCC, to_type: type) -> Union[str, int]:
    """Convert FourCC to specified type.

    Args:
        value: FourCC value (str or int)
        to_type: Target type (str or int)

    Returns:
        FourCC in target type

    Examples:
        >>> convert_fourcc('lpcm', int)
        1819304813
        >>> convert_fourcc(1819304813, str)
        'lpcm'
    """
    if to_type is str:
        return ensure_fourcc_str(value)
    elif to_type is int:
        return ensure_fourcc_int(value)
    else:
        raise TypeError(f"Target type must be str or int, not {to_type.__name__}")
