#!/usr/bin/env python3
"""Base classes and mixins for audio modules.

This module provides shared functionality to reduce code duplication across
audio processing classes that need to load audio files.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

# Type checking imports
if TYPE_CHECKING:
    try:
        from numpy.typing import NDArray as NDArray_
    except ImportError:
        NDArray_ = Any  # type: ignore[misc,assignment]
    NDArray = NDArray_


class AudioFileLoaderMixin:
    """Mixin providing audio file loading functionality.

    This mixin provides common audio file loading patterns used by
    visualization and analysis classes.

    Attributes:
        audio_file: Path to the audio file
        _audio_data: Cached audio data as NumPy array
        _sample_rate: Cached sample rate

    Subclasses should:
        1. Call _init_audio_loader(audio_file) in __init__
        2. Override _check_dependencies() to verify required libraries
        3. Use _load_audio() to get (audio_data, sample_rate)
    """

    audio_file: Path
    _audio_data: Optional["NDArray"]
    _sample_rate: Optional[float]

    def _init_audio_loader(self, audio_file: str) -> None:
        """Initialize audio loader state.

        Args:
            audio_file: Path to audio file

        Raises:
            ImportError: If required dependencies are missing
        """
        self._check_dependencies()
        self.audio_file = Path(audio_file)
        self._audio_data = None
        self._sample_rate = None

    def _check_dependencies(self) -> None:
        """Check that required dependencies are available.

        Override in subclasses to check for numpy, scipy, matplotlib, etc.

        Raises:
            ImportError: If required dependencies are missing
        """
        pass  # Default: no dependencies required

    def _load_audio(self) -> Tuple["NDArray", float]:
        """Load audio data from file.

        Returns:
            Tuple of (audio_data, sample_rate)

        Note:
            Audio data is cached after first load.
        """
        if self._audio_data is None:
            import coremusic as cm

            with cm.AudioFile(str(self.audio_file)) as af:
                self._audio_data = af.read_as_numpy()
                self._sample_rate = af.format.sample_rate

        assert self._audio_data is not None
        assert self._sample_rate is not None
        return self._audio_data, self._sample_rate

    def _load_audio_mono(self) -> Tuple["NDArray", float]:
        """Load audio data and convert to mono if stereo.

        Returns:
            Tuple of (mono_audio_data, sample_rate)

        Note:
            Audio data is cached after first load.
        """
        import numpy as np

        # Use base implementation directly to avoid recursion
        audio_data, sample_rate = AudioFileLoaderMixin._load_audio(self)

        # Convert to mono if stereo
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
            self._audio_data = audio_data

        return audio_data, sample_rate


def check_numpy_available() -> None:
    """Check that NumPy is available.

    Raises:
        ImportError: If NumPy is not available
    """
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        raise ImportError(
            "NumPy is required. Install with: pip install numpy"
        )


def check_matplotlib_available() -> None:
    """Check that matplotlib is available.

    Raises:
        ImportError: If matplotlib is not available
    """
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def check_scipy_available() -> None:
    """Check that SciPy is available.

    Raises:
        ImportError: If SciPy is not available
    """
    try:
        from scipy import signal  # noqa: F401
    except ImportError:
        raise ImportError(
            "SciPy is required for audio analysis. Install with: pip install scipy"
        )
