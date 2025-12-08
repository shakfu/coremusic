#!/usr/bin/env python3
"""Tests for audio analysis module."""

import pytest
from conftest import AMEN_WAV_PATH
import coremusic as cm
from coremusic.audio.analysis import (
    AudioAnalyzer,
    BeatInfo,
    PitchInfo,
    LivePitchDetector,
    NUMPY_AVAILABLE,
    SCIPY_AVAILABLE,
)

# Skip all tests if NumPy or SciPy not available
pytestmark = pytest.mark.skipif(
    not (NUMPY_AVAILABLE and SCIPY_AVAILABLE),
    reason="NumPy and SciPy required for audio analysis",
)

if NUMPY_AVAILABLE:
    import numpy as np


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_audio_file():
    """Path to test audio file."""
    return AMEN_WAV_PATH


@pytest.fixture
def analyzer(test_audio_file):
    """AudioAnalyzer instance."""
    return AudioAnalyzer(test_audio_file)


# ============================================================================
# Test Data Classes
# ============================================================================


class TestDataClasses:
    """Test data classes."""

    def test_beat_info_creation(self):
        """Test BeatInfo creation."""
        beat_info = BeatInfo(
            tempo=120.0, beats=[0.0, 0.5, 1.0], downbeats=[0.0, 2.0], confidence=0.9
        )

        assert beat_info.tempo == 120.0
        assert len(beat_info.beats) == 3
        assert len(beat_info.downbeats) == 2
        assert beat_info.confidence == 0.9

    def test_pitch_info_creation(self):
        """Test PitchInfo creation."""
        pitch_info = PitchInfo(
            frequency=440.0, midi_note=69, cents_offset=0.0, confidence=0.95
        )

        assert pitch_info.frequency == 440.0
        assert pitch_info.midi_note == 69
        assert pitch_info.cents_offset == 0.0
        assert pitch_info.confidence == 0.95


# ============================================================================
# Test AudioAnalyzer
# ============================================================================


class TestAudioAnalyzer:
    """Test AudioAnalyzer class."""

    def test_create_analyzer(self, test_audio_file):
        """Test creating analyzer."""
        analyzer = AudioAnalyzer(test_audio_file)

        assert analyzer.audio_file.exists()
        assert analyzer._audio_data is None
        assert analyzer._sample_rate is None

    def test_create_analyzer_without_numpy_raises_error(self, monkeypatch):
        """Test that creating analyzer without NumPy raises error."""
        # Temporarily disable NumPy
        import coremusic.audio.analysis as analysis_module

        monkeypatch.setattr(analysis_module, "NUMPY_AVAILABLE", False)

        with pytest.raises(ImportError, match="NumPy is required"):
            AudioAnalyzer(AMEN_WAV_PATH)

    def test_load_audio(self, analyzer):
        """Test loading audio file."""
        data, sr = analyzer._load_audio()

        assert data is not None
        assert sr > 0
        assert len(data.shape) == 1  # Mono

    def test_load_audio_caches_data(self, analyzer):
        """Test that audio loading is cached."""
        data1, sr1 = analyzer._load_audio()
        data2, sr2 = analyzer._load_audio()

        assert data1 is data2
        assert sr1 == sr2

    # ========================================================================
    # Spectral Analysis Tests
    # ========================================================================

    def test_compute_fft(self, analyzer):
        """Test FFT computation."""
        data, sr = analyzer._load_audio()

        # Take short segment
        segment = data[:2048]
        freqs, mags = analyzer._compute_fft(segment, sr)

        assert len(freqs) > 0
        assert len(mags) == len(freqs)
        assert np.all(freqs >= 0)
        assert np.all(mags >= 0)

    def test_spectral_centroid(self, analyzer):
        """Test spectral centroid calculation."""
        # Simple test with known spectrum
        freqs = np.array([100, 200, 300, 400, 500])
        mags = np.array([1, 2, 3, 2, 1])

        centroid = analyzer._spectral_centroid(freqs, mags)

        # Centroid should be near 300 (peak)
        assert 250 < centroid < 350

    def test_spectral_centroid_zero_magnitude(self, analyzer):
        """Test spectral centroid with zero magnitude."""
        freqs = np.array([100, 200, 300])
        mags = np.array([0, 0, 0])

        centroid = analyzer._spectral_centroid(freqs, mags)
        assert centroid == 0.0

    def test_spectral_rolloff(self, analyzer):
        """Test spectral rolloff calculation."""
        freqs = np.linspace(0, 1000, 100)
        mags = np.exp(-freqs / 500)  # Decaying spectrum

        rolloff = analyzer._spectral_rolloff(freqs, mags)

        assert 0 < rolloff < 1000

    def test_find_spectral_peaks(self, analyzer):
        """Test spectral peak finding."""
        freqs = np.linspace(0, 1000, 1000)
        # Create spectrum with peaks at 100, 300, 500 Hz
        mags = np.zeros(1000)
        mags[100] = 1.0
        mags[300] = 0.8
        mags[500] = 0.6

        peaks = analyzer._find_spectral_peaks(freqs, mags, num_peaks=3)

        assert len(peaks) <= 3
        # Check that we found some peaks
        if len(peaks) > 0:
            peak_freqs = [f for f, m in peaks]
            assert all(0 <= f <= 1000 for f in peak_freqs)

    def test_analyze_spectrum(self, analyzer):
        """Test spectrum analysis at specific time."""
        result = analyzer.analyze_spectrum(time=0.5, window_size=0.1)

        assert "frequencies" in result
        assert "magnitudes" in result
        assert "peaks" in result
        assert "centroid" in result
        assert "rolloff" in result

        assert len(result["frequencies"]) > 0
        assert len(result["magnitudes"]) == len(result["frequencies"])
        assert result["centroid"] >= 0
        assert result["rolloff"] >= 0

    def test_analyze_spectrum_at_start(self, analyzer):
        """Test spectrum analysis at file start."""
        result = analyzer.analyze_spectrum(time=0.0, window_size=0.05)

        assert result["centroid"] >= 0
        assert result["rolloff"] >= 0

    # ========================================================================
    # Onset Detection Tests
    # ========================================================================

    def test_detect_onsets(self, analyzer):
        """Test onset detection."""
        data, sr = analyzer._load_audio()

        onsets = analyzer._detect_onsets(data, sr)

        assert len(onsets) > 0
        assert np.all(onsets >= 0)
        assert np.all(onsets <= len(data) / sr)

    def test_detect_onsets_increasing(self, analyzer):
        """Test that onsets are in increasing order."""
        data, sr = analyzer._load_audio()

        onsets = analyzer._detect_onsets(data, sr)

        if len(onsets) > 1:
            assert np.all(np.diff(onsets) > 0)

    # ========================================================================
    # Beat Detection Tests
    # ========================================================================

    def test_estimate_tempo(self, analyzer):
        """Test tempo estimation."""
        # Create artificial onsets at 120 BPM (0.5 second intervals)
        onsets = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

        tempo, beats = analyzer._estimate_tempo(onsets)

        # Should detect 120 BPM
        assert 100 < tempo < 140  # Allow some tolerance

    def test_estimate_tempo_no_onsets(self, analyzer):
        """Test tempo estimation with no onsets."""
        onsets = np.array([])

        tempo, beats = analyzer._estimate_tempo(onsets)

        assert tempo == 60.0  # Default
        assert len(beats) == 0

    def test_detect_beats(self, analyzer):
        """Test beat detection."""
        beat_info = analyzer.detect_beats()

        assert isinstance(beat_info, BeatInfo)
        assert 40 <= beat_info.tempo <= 200
        assert len(beat_info.beats) > 0
        assert len(beat_info.downbeats) >= 0
        assert 0 <= beat_info.confidence <= 1

    def test_detect_beats_has_downbeats(self, analyzer):
        """Test that beat detection includes downbeats."""
        beat_info = analyzer.detect_beats()

        # Downbeats should be subset of beats
        if len(beat_info.downbeats) > 0:
            # First downbeat should align with a beat
            assert any(
                abs(beat_info.downbeats[0] - b) < 0.01 for b in beat_info.beats
            )

    # ========================================================================
    # Pitch Detection Tests
    # ========================================================================

    def test_autocorrelation_pitch(self, analyzer):
        """Test pitch detection with autocorrelation."""
        # Generate 440 Hz sine wave
        sr = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)

        pitch = analyzer._autocorrelation_pitch(audio, sr)

        # Should detect approximately 440 Hz
        if pitch:
            assert 400 < pitch < 480

    def test_autocorrelation_pitch_no_pitch(self, analyzer):
        """Test pitch detection with noise (no clear pitch)."""
        # Random noise
        sr = 44100
        audio = np.random.randn(2048) * 0.1

        pitch = analyzer._autocorrelation_pitch(audio, sr)

        # May or may not detect pitch in noise
        if pitch:
            assert 50 <= pitch <= 2000

    def test_freq_to_midi(self, analyzer):
        """Test frequency to MIDI conversion."""
        # A440 = MIDI note 69
        midi_note, cents = analyzer._freq_to_midi(440.0)

        assert midi_note == 69
        assert abs(cents) < 1  # Should be very close to 0

    def test_freq_to_midi_c4(self, analyzer):
        """Test frequency to MIDI for C4."""
        # C4 ≈ 261.63 Hz = MIDI note 60
        midi_note, cents = analyzer._freq_to_midi(261.63)

        assert midi_note == 60

    def test_detect_pitch(self, analyzer):
        """Test pitch detection over time."""
        pitch_track = analyzer.detect_pitch(time_range=(0.0, 0.5))

        # Should get some pitch detections
        assert isinstance(pitch_track, list)

        if len(pitch_track) > 0:
            for pitch_info in pitch_track:
                assert isinstance(pitch_info, PitchInfo)
                assert 50 <= pitch_info.frequency <= 2000
                assert 0 <= pitch_info.midi_note <= 127
                assert -50 <= pitch_info.cents_offset <= 50
                assert 0 <= pitch_info.confidence <= 1

    # ========================================================================
    # MFCC Tests
    # ========================================================================

    def test_mel_filterbank(self, analyzer):
        """Test mel filterbank creation."""
        n_fft = 2048
        n_mels = 40
        sr = 44100

        filterbank = analyzer._mel_filterbank(n_fft, n_mels, sr)

        assert filterbank.shape == (n_mels, n_fft // 2 + 1)
        assert np.all(filterbank >= 0)

    def test_extract_mfcc(self, analyzer):
        """Test MFCC extraction."""
        n_mfcc = 13

        mfcc = analyzer.extract_mfcc(n_mfcc=n_mfcc)

        assert mfcc.shape[0] == n_mfcc
        assert mfcc.shape[1] > 0  # Should have multiple frames

    def test_extract_mfcc_different_sizes(self, analyzer):
        """Test MFCC with different sizes."""
        mfcc_13 = analyzer.extract_mfcc(n_mfcc=13)
        mfcc_20 = analyzer.extract_mfcc(n_mfcc=20)

        assert mfcc_13.shape[0] == 13
        assert mfcc_20.shape[0] == 20

    # ========================================================================
    # Chroma and Key Detection Tests
    # ========================================================================

    def test_compute_chroma(self, analyzer):
        """Test chromagram computation."""
        data, sr = analyzer._load_audio()

        chroma = analyzer._compute_chroma(data, sr)

        assert chroma.shape[0] == 12  # 12 chroma bins
        assert chroma.shape[1] > 0  # Multiple frames
        assert np.all(chroma >= 0)

    def test_estimate_key(self, analyzer):
        """Test key estimation."""
        # Create artificial chroma emphasizing C major
        chroma = np.zeros((12, 100))
        # C major: C, E, G
        chroma[0, :] = 1.0  # C
        chroma[4, :] = 0.8  # E
        chroma[7, :] = 0.8  # G

        key, mode = analyzer._estimate_key(chroma)

        assert key in [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
        ]
        assert mode in ["major", "minor"]

    def test_detect_key(self, analyzer):
        """Test key detection."""
        key, mode = analyzer.detect_key()

        assert key in [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
        ]
        assert mode in ["major", "minor"]

    # ========================================================================
    # Fingerprinting Tests
    # ========================================================================

    def test_generate_fingerprint(self, analyzer):
        """Test fingerprint generation."""
        data, sr = analyzer._load_audio()

        fingerprint = analyzer._generate_fingerprint(data, sr)

        assert isinstance(fingerprint, str)
        assert len(fingerprint) > 0
        # Should be hex string
        assert all(c in "0123456789abcdef" for c in fingerprint)

    def test_get_audio_fingerprint(self, analyzer):
        """Test audio fingerprint API."""
        fingerprint = analyzer.get_audio_fingerprint()

        assert isinstance(fingerprint, str)
        assert len(fingerprint) > 0

    def test_fingerprint_consistency(self, analyzer):
        """Test that fingerprint is consistent."""
        fp1 = analyzer.get_audio_fingerprint()
        fp2 = analyzer.get_audio_fingerprint()

        assert fp1 == fp2


# ============================================================================
# Test LivePitchDetector
# ============================================================================


class TestLivePitchDetector:
    """Test LivePitchDetector class."""

    def test_create_detector(self):
        """Test creating detector."""
        detector = LivePitchDetector(44100, 2048)

        assert detector.sample_rate == 44100
        assert detector.buffer_size == 2048
        assert len(detector._buffer) == 2048

    def test_create_detector_without_numpy_raises_error(self, monkeypatch):
        """Test that creating detector without NumPy raises error."""
        import coremusic.audio.analysis as analysis_module

        monkeypatch.setattr(analysis_module, "NUMPY_AVAILABLE", False)

        with pytest.raises(ImportError, match="NumPy is required"):
            LivePitchDetector(44100, 2048)

    def test_process_sine_wave(self):
        """Test processing sine wave."""
        detector = LivePitchDetector(44100, 2048)

        # Generate 440 Hz sine wave
        sr = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)

        # Process in chunks
        chunk_size = 512
        pitch_info = None

        for i in range(0, len(audio) - chunk_size, chunk_size):
            chunk = audio[i : i + chunk_size]
            pitch_info = detector.process(chunk)
            if pitch_info:
                break

        # Should detect approximately 440 Hz
        if pitch_info:
            assert 400 < pitch_info.frequency < 480
            assert pitch_info.midi_note == 69  # A4

    def test_process_noise(self):
        """Test processing noise."""
        detector = LivePitchDetector(44100, 2048)

        # Random noise
        audio = np.random.randn(2048) * 0.1

        pitch_info = detector.process(audio)

        # May or may not detect pitch in noise
        if pitch_info:
            assert 50 <= pitch_info.frequency <= 2000

    def test_process_updates_buffer(self):
        """Test that process updates internal buffer."""
        detector = LivePitchDetector(44100, 2048)

        chunk1 = np.ones(512)
        chunk2 = np.zeros(512)

        detector.process(chunk1)
        buffer_after_1 = detector._buffer.copy()

        detector.process(chunk2)
        buffer_after_2 = detector._buffer.copy()

        # Buffer should be different after second chunk
        assert not np.array_equal(buffer_after_1, buffer_after_2)

    def test_freq_to_midi_conversion(self):
        """Test frequency to MIDI conversion."""
        detector = LivePitchDetector(44100, 2048)

        # Test A440
        midi_note, cents = detector._freq_to_midi(440.0)
        assert midi_note == 69
        assert abs(cents) < 1

        # Test C4
        midi_note, cents = detector._freq_to_midi(261.63)
        assert midi_note == 60


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests."""

    def test_full_analysis_workflow(self, test_audio_file):
        """Test complete analysis workflow."""
        analyzer = AudioAnalyzer(test_audio_file)

        # Beat detection
        beat_info = analyzer.detect_beats()
        assert isinstance(beat_info, BeatInfo)

        # Key detection
        key, mode = analyzer.detect_key()
        assert isinstance(key, str)
        assert isinstance(mode, str)

        # Spectrum analysis
        spectrum = analyzer.analyze_spectrum(time=0.5)
        assert "centroid" in spectrum

        # Fingerprint
        fingerprint = analyzer.get_audio_fingerprint()
        assert len(fingerprint) > 0

    def test_pitch_detection_with_live_detector(self):
        """Test live pitch detection workflow."""
        # Generate test tone
        sr = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)

        # Process with live detector
        detector = LivePitchDetector(sr, 2048)

        detections = []
        chunk_size = 512

        for i in range(0, len(audio) - chunk_size, chunk_size):
            chunk = audio[i : i + chunk_size]
            pitch_info = detector.process(chunk)
            if pitch_info:
                detections.append(pitch_info)

        # Should get some detections
        assert len(detections) > 0

        # Average frequency should be close to 440 Hz
        avg_freq = np.mean([d.frequency for d in detections])
        assert 400 < avg_freq < 480
