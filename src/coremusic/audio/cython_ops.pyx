"""Cython-optimized audio processing operations.

This module provides high-performance audio processing functions using Cython
optimizations including typed memoryviews, nogil operations, and inline functions.
"""

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

cimport cython
from libc.math cimport fabs, sqrt, log10
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset


# Type definitions for clarity
ctypedef double float64_t
ctypedef float float32_t
ctypedef short int16_t


# =============================================================================
# Inline utility functions (nogil for maximum performance)
# =============================================================================

cdef inline float32_t clip_float32(float32_t value, float32_t min_val, float32_t max_val) nogil:
    """Clip float32 value to range [min_val, max_val]."""
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value


cdef inline int16_t clip_int16(int value) nogil:
    """Clip integer value to int16 range."""
    if value < -32768:
        return -32768
    if value > 32767:
        return 32767
    return <int16_t>value


cdef inline float64_t db_to_linear(float64_t db) nogil:
    """Convert decibels to linear gain."""
    return 10.0 ** (db / 20.0)


cdef inline float64_t linear_to_db(float64_t linear) nogil:
    """Convert linear gain to decibels."""
    if linear <= 0.0:
        return -100.0  # Effectively -inf
    return 20.0 * log10(linear)


# =============================================================================
# Audio buffer operations (with GIL release)
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def normalize_audio_float32(float32_t[:, ::1] audio_data, float32_t target_peak=0.9):
    """Normalize audio to target peak level.

    Args:
        audio_data: 2D array (frames, channels) of float32 audio data
        target_peak: Target peak level (0.0 to 1.0)

    Returns:
        None (modifies audio_data in-place)

    Note:
        This function releases the GIL for parallel processing.
    """
    cdef int frames = audio_data.shape[0]
    cdef int channels = audio_data.shape[1]
    cdef float32_t max_val = 0.0
    cdef float32_t gain
    cdef int i, j

    # Find peak value (with GIL released)
    with nogil:
        for i in range(frames):
            for j in range(channels):
                if fabs(audio_data[i, j]) > max_val:
                    max_val = fabs(audio_data[i, j])

        # Calculate and apply gain
        if max_val > 0.0:
            gain = target_peak / max_val
            for i in range(frames):
                for j in range(channels):
                    audio_data[i, j] *= gain


@cython.boundscheck(False)
@cython.wraparound(False)
def apply_gain_float32(float32_t[:, ::1] audio_data, float32_t gain_db):
    """Apply gain in decibels to audio data.

    Args:
        audio_data: 2D array (frames, channels) of float32 audio data
        gain_db: Gain in decibels

    Returns:
        None (modifies audio_data in-place)
    """
    cdef int frames = audio_data.shape[0]
    cdef int channels = audio_data.shape[1]
    cdef float32_t linear_gain = <float32_t>db_to_linear(gain_db)
    cdef int i, j

    with nogil:
        for i in range(frames):
            for j in range(channels):
                audio_data[i, j] *= linear_gain


@cython.boundscheck(False)
@cython.wraparound(False)
def mix_audio_float32(float32_t[:, ::1] output, float32_t[:, ::1] input1,
                      float32_t[:, ::1] input2, float32_t mix_ratio=0.5):
    """Mix two audio buffers.

    Args:
        output: Output buffer (frames, channels)
        input1: First input buffer (frames, channels)
        input2: Second input buffer (frames, channels)
        mix_ratio: Mix ratio (0.0 = all input1, 1.0 = all input2)

    Returns:
        None (writes to output buffer)
    """
    cdef int frames = output.shape[0]
    cdef int channels = output.shape[1]
    cdef float32_t ratio1 = 1.0 - mix_ratio
    cdef float32_t ratio2 = mix_ratio
    cdef int i, j

    with nogil:
        for i in range(frames):
            for j in range(channels):
                output[i, j] = input1[i, j] * ratio1 + input2[i, j] * ratio2


@cython.boundscheck(False)
@cython.wraparound(False)
def convert_float32_to_int16(float32_t[:, ::1] float_data, int16_t[:, ::1] int_data):
    """Convert float32 audio to int16.

    Args:
        float_data: Input float32 data (frames, channels) in range [-1.0, 1.0]
        int_data: Output int16 data (frames, channels)

    Returns:
        None (writes to int_data buffer)
    """
    cdef int frames = float_data.shape[0]
    cdef int channels = float_data.shape[1]
    cdef int i, j
    cdef int value

    with nogil:
        for i in range(frames):
            for j in range(channels):
                value = <int>(float_data[i, j] * 32767.0)
                int_data[i, j] = clip_int16(value)


@cython.boundscheck(False)
@cython.wraparound(False)
def convert_int16_to_float32(int16_t[:, ::1] int_data, float32_t[:, ::1] float_data):
    """Convert int16 audio to float32.

    Args:
        int_data: Input int16 data (frames, channels)
        float_data: Output float32 data (frames, channels) in range [-1.0, 1.0]

    Returns:
        None (writes to float_data buffer)
    """
    cdef int frames = int_data.shape[0]
    cdef int channels = int_data.shape[1]
    cdef int i, j

    with nogil:
        for i in range(frames):
            for j in range(channels):
                float_data[i, j] = <float32_t>int_data[i, j] / 32768.0


# =============================================================================
# Stereo/Mono conversions
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def stereo_to_mono_float32(float32_t[:, ::1] stereo_data, float32_t[::1] mono_data):
    """Convert stereo audio to mono by averaging channels.

    Args:
        stereo_data: Input stereo data (frames, 2)
        mono_data: Output mono data (frames,)

    Returns:
        None (writes to mono_data buffer)
    """
    cdef int frames = stereo_data.shape[0]
    cdef int i

    with nogil:
        for i in range(frames):
            mono_data[i] = (stereo_data[i, 0] + stereo_data[i, 1]) * 0.5


@cython.boundscheck(False)
@cython.wraparound(False)
def mono_to_stereo_float32(float32_t[::1] mono_data, float32_t[:, ::1] stereo_data):
    """Convert mono audio to stereo by duplicating channel.

    Args:
        mono_data: Input mono data (frames,)
        stereo_data: Output stereo data (frames, 2)

    Returns:
        None (writes to stereo_data buffer)
    """
    cdef int frames = mono_data.shape[0]
    cdef int i

    with nogil:
        for i in range(frames):
            stereo_data[i, 0] = mono_data[i]
            stereo_data[i, 1] = mono_data[i]


# =============================================================================
# Fade operations
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_fade_in_float32(float32_t[:, ::1] audio_data, int fade_frames):
    """Apply linear fade-in to audio data.

    Args:
        audio_data: Audio data (frames, channels)
        fade_frames: Number of frames for fade-in

    Returns:
        None (modifies audio_data in-place)
    """
    cdef int frames = audio_data.shape[0]
    cdef int channels = audio_data.shape[1]
    cdef int actual_fade = min(fade_frames, frames)
    cdef float32_t gain
    cdef int i, j

    with nogil:
        for i in range(actual_fade):
            gain = <float32_t>i / <float32_t>actual_fade
            for j in range(channels):
                audio_data[i, j] *= gain


@cython.boundscheck(False)
@cython.wraparound(False)
def apply_fade_out_float32(float32_t[:, ::1] audio_data, int fade_frames):
    """Apply linear fade-out to audio data.

    Args:
        audio_data: Audio data (frames, channels)
        fade_frames: Number of frames for fade-out

    Returns:
        None (modifies audio_data in-place)
    """
    cdef int frames = audio_data.shape[0]
    cdef int channels = audio_data.shape[1]
    cdef int actual_fade = min(fade_frames, frames)
    cdef int start_frame = frames - actual_fade
    cdef float32_t gain
    cdef int i, j

    with nogil:
        for i in range(actual_fade):
            gain = 1.0 - (<float32_t>i / <float32_t>actual_fade)
            for j in range(channels):
                audio_data[start_frame + i, j] *= gain


# =============================================================================
# Signal analysis
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_rms_float32(float32_t[:, ::1] audio_data):
    """Calculate RMS (Root Mean Square) level of audio.

    Args:
        audio_data: Audio data (frames, channels)

    Returns:
        RMS level as float
    """
    cdef int frames = audio_data.shape[0]
    cdef int channels = audio_data.shape[1]
    cdef float64_t sum_squares = 0.0
    cdef int i, j
    cdef float32_t sample

    with nogil:
        for i in range(frames):
            for j in range(channels):
                sample = audio_data[i, j]
                sum_squares += sample * sample

    return sqrt(sum_squares / (frames * channels))


@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_peak_float32(float32_t[:, ::1] audio_data):
    """Calculate peak level of audio.

    Args:
        audio_data: Audio data (frames, channels)

    Returns:
        Peak level as float
    """
    cdef int frames = audio_data.shape[0]
    cdef int channels = audio_data.shape[1]
    cdef float32_t peak = 0.0
    cdef int i, j
    cdef float32_t abs_val

    with nogil:
        for i in range(frames):
            for j in range(channels):
                abs_val = fabs(audio_data[i, j])
                if abs_val > peak:
                    peak = abs_val

    return peak


# =============================================================================
# Python wrapper functions with NumPy array creation
# =============================================================================

def normalize_audio(audio_data, float target_peak=0.9):
    """Normalize audio to target peak level (NumPy wrapper).

    Args:
        audio_data: NumPy array of float32 audio data
        target_peak: Target peak level (0.0 to 1.0)

    Returns:
        Normalized audio array

    Example:
        >>> import numpy as np
        >>> from coremusic.audio.cython_ops import normalize_audio
        >>> audio = np.random.randn(44100, 2).astype(np.float32)
        >>> normalized = normalize_audio(audio, target_peak=0.8)
    """
    import numpy as np

    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(-1, 1)

    result = audio_data.copy()
    normalize_audio_float32(result, target_peak)
    return result


def apply_gain(audio_data, float gain_db):
    """Apply gain in decibels (NumPy wrapper).

    Args:
        audio_data: NumPy array of float32 audio data
        gain_db: Gain in decibels

    Returns:
        Audio with gain applied

    Example:
        >>> import numpy as np
        >>> from coremusic.audio.cython_ops import apply_gain
        >>> audio = np.random.randn(44100, 2).astype(np.float32)
        >>> louder = apply_gain(audio, gain_db=6.0)  # +6dB
    """
    import numpy as np

    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(-1, 1)

    result = audio_data.copy()
    apply_gain_float32(result, gain_db)
    return result


def calculate_rms(audio_data):
    """Calculate RMS level (NumPy wrapper).

    Args:
        audio_data: NumPy array of float32 audio data

    Returns:
        RMS level as float

    Example:
        >>> import numpy as np
        >>> from coremusic.audio.cython_ops import calculate_rms
        >>> audio = np.random.randn(44100, 2).astype(np.float32) * 0.5
        >>> rms = calculate_rms(audio)
        >>> print(f"RMS level: {rms:.4f}")
    """
    import numpy as np

    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(-1, 1)

    return calculate_rms_float32(audio_data)


def calculate_peak(audio_data):
    """Calculate peak level (NumPy wrapper).

    Args:
        audio_data: NumPy array of float32 audio data

    Returns:
        Peak level as float

    Example:
        >>> import numpy as np
        >>> from coremusic.audio.cython_ops import calculate_peak
        >>> audio = np.random.randn(44100, 2).astype(np.float32) * 0.5
        >>> peak = calculate_peak(audio)
        >>> print(f"Peak level: {peak:.4f}")
    """
    import numpy as np

    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(-1, 1)

    return calculate_peak_float32(audio_data)
