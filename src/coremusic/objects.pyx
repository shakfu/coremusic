#!/usr/bin/env python3
"""Cython extension class for automatic resource management in coremusic.

This module provides the CoreAudioObject base class which handles automatic
resource cleanup via __dealloc__. All other classes are implemented as pure
Python classes in the objects module.
"""

# ============================================================================
# Base Infrastructure - Cython Extension Class
# ============================================================================

cdef class CoreAudioObject:
    """Base class for all CoreAudio objects with automatic resource management

    This is the only Cython extension class, providing __dealloc__ for automatic
    cleanup. All other classes are implemented as pure Python classes.
    """

    cdef long _object_id
    cdef bint _is_disposed

    def __cinit__(self):
        self._object_id = 0
        self._is_disposed = False

    def __dealloc__(self):
        """Automatic cleanup when object is garbage collected"""
        if not self._is_disposed:
            self._dispose_internal()

    @property
    def is_disposed(self) -> bool:
        """Check if the object has been disposed"""
        return self._is_disposed

    def dispose(self) -> None:
        """Explicitly dispose of the object's resources"""
        if not self._is_disposed:
            self._dispose_internal()

    cdef void _dispose_internal(self):
        """Internal disposal implementation"""
        self._is_disposed = True
        self._object_id = 0

    def _ensure_not_disposed(self) -> None:
        """Ensure the object has not been disposed"""
        if self._is_disposed:
            raise RuntimeError(f"{self.__class__.__name__} has been disposed")

    @property
    def object_id(self) -> int:
        """Expose object ID for testing purposes"""
        return self._object_id

    def _set_object_id(self, object_id: int) -> None:
        """Set object ID (for internal use)"""
        self._object_id = object_id