#!/usr/bin/env python3
"""Tests for high-level audio utilities."""

import os
import pytest
import tempfile
from pathlib import Path

import coremusic as cm


class TestAudioAnalyzer:
    """Test AudioAnalyzer utilities"""

    @pytest.fixture
    def amen_wav_path(self):
        """Fixture providing path to amen.wav test file"""
        path = os.path.join("tests", "amen.wav")
        if not os.path.exists(path):
            pytest.skip(f"Test audio file not found: {path}")
        return path

    @pytest.mark.skipif(not cm.NUMPY_AVAILABLE, reason="NumPy not available")
    def test_detect_silence(self, amen_wav_path):
        """Test silence detection"""
        silence_regions = cm.AudioAnalyzer.detect_silence(
            amen_wav_path,
            threshold_db=-50,
            min_duration=0.1
        )

        # Should return a list of tuples
        assert isinstance(silence_regions, list)
        for start, end in silence_regions:
            assert isinstance(start, float)
            assert isinstance(end, float)
            assert end > start
            assert start >= 0

    @pytest.mark.skipif(not cm.NUMPY_AVAILABLE, reason="NumPy not available")
    def test_detect_silence_with_audio_file_object(self, amen_wav_path):
        """Test silence detection with AudioFile object"""
        with cm.AudioFile(amen_wav_path) as audio:
            silence_regions = cm.AudioAnalyzer.detect_silence(
                audio,
                threshold_db=-50,
                min_duration=0.1
            )
            assert isinstance(silence_regions, list)

    @pytest.mark.skipif(not cm.NUMPY_AVAILABLE, reason="NumPy not available")
    def test_get_peak_amplitude(self, amen_wav_path):
        """Test peak amplitude detection"""
        peak = cm.AudioAnalyzer.get_peak_amplitude(amen_wav_path)

        assert isinstance(peak, float)
        assert 0 <= peak <= 1.5  # Allow some headroom for int16

    @pytest.mark.skipif(not cm.NUMPY_AVAILABLE, reason="NumPy not available")
    def test_get_peak_amplitude_with_audio_file_object(self, amen_wav_path):
        """Test peak amplitude with AudioFile object"""
        with cm.AudioFile(amen_wav_path) as audio:
            peak = cm.AudioAnalyzer.get_peak_amplitude(audio)
            assert isinstance(peak, float)
            assert peak > 0

    @pytest.mark.skipif(not cm.NUMPY_AVAILABLE, reason="NumPy not available")
    def test_calculate_rms(self, amen_wav_path):
        """Test RMS calculation"""
        rms = cm.AudioAnalyzer.calculate_rms(amen_wav_path)

        assert isinstance(rms, float)
        assert 0 < rms < 1.0
        # RMS should be less than peak
        peak = cm.AudioAnalyzer.get_peak_amplitude(amen_wav_path)
        assert rms <= peak

    @pytest.mark.skipif(not cm.NUMPY_AVAILABLE, reason="NumPy not available")
    def test_calculate_rms_with_audio_file_object(self, amen_wav_path):
        """Test RMS with AudioFile object"""
        with cm.AudioFile(amen_wav_path) as audio:
            rms = cm.AudioAnalyzer.calculate_rms(audio)
            assert isinstance(rms, float)
            assert rms > 0

    def test_get_file_info(self, amen_wav_path):
        """Test file info extraction"""
        info = cm.AudioAnalyzer.get_file_info(amen_wav_path)

        # Check required fields
        assert 'path' in info
        assert 'duration' in info
        assert 'sample_rate' in info
        assert 'format_id' in info
        assert 'channels' in info
        assert 'bits_per_channel' in info

        # Check values
        assert info['duration'] > 0
        assert info['sample_rate'] == 44100.0
        assert info['format_id'] == 'lpcm'
        assert info['channels'] == 2
        assert info['is_stereo'] is True

        # NumPy fields should be present if available
        if cm.NUMPY_AVAILABLE:
            assert 'peak_amplitude' in info
            assert 'rms' in info


class TestBatchProcessing:
    """Test batch processing utilities"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def amen_wav_path(self):
        """Fixture providing path to amen.wav test file"""
        path = os.path.join("tests", "amen.wav")
        if not os.path.exists(path):
            pytest.skip(f"Test audio file not found: {path}")
        return path

    def test_convert_audio_file_stereo_to_mono(self, amen_wav_path, temp_dir):
        """Test stereo to mono conversion"""
        output_path = os.path.join(temp_dir, "output.wav")
        output_format = cm.AudioFormatPresets.wav_44100_mono()

        cm.convert_audio_file(amen_wav_path, output_path, output_format)

        # Verify output file was created
        assert os.path.exists(output_path)

        # Verify format
        with cm.AudioFile(output_path) as audio:
            format = audio.format
            assert format.sample_rate == 44100.0
            assert format.channels_per_frame == 1  # Mono

    def test_convert_audio_file_unsupported_raises(self, amen_wav_path, temp_dir):
        """Test that unsupported conversions raise NotImplementedError"""
        output_path = os.path.join(temp_dir, "output.wav")
        # Try to convert sample rate (not yet supported in utilities)
        output_format = cm.AudioFormatPresets.wav_48000_stereo()

        with pytest.raises(NotImplementedError):
            cm.convert_audio_file(amen_wav_path, output_path, output_format)

    def test_batch_convert_single_file(self, amen_wav_path, temp_dir):
        """Test batch conversion with single file"""
        # Copy test file to temp dir
        import shutil
        input_file = os.path.join(temp_dir, "input.wav")
        shutil.copy(amen_wav_path, input_file)

        output_format = cm.AudioFormatPresets.wav_44100_mono()
        output_dir = os.path.join(temp_dir, "output")

        converted = cm.batch_convert(
            input_pattern=f"{temp_dir}/input.wav",
            output_format=output_format,
            output_dir=output_dir,
            output_extension="wav"
        )

        assert len(converted) == 1
        assert os.path.exists(converted[0])

        # Verify output format
        with cm.AudioFile(converted[0]) as audio:
            assert audio.format.channels_per_frame == 1  # Mono

    def test_batch_convert_with_progress_callback(self, amen_wav_path, temp_dir):
        """Test batch conversion with progress callback"""
        import shutil
        input_file = os.path.join(temp_dir, "input.wav")
        shutil.copy(amen_wav_path, input_file)

        callback_called = []

        def progress_callback(filename, current, total):
            callback_called.append((filename, current, total))

        output_format = cm.AudioFormatPresets.wav_44100_stereo()

        cm.batch_convert(
            input_pattern=f"{temp_dir}/*.wav",
            output_format=output_format,
            output_dir=temp_dir,
            output_extension="converted.wav",
            progress_callback=progress_callback
        )

        # Verify callback was called
        assert len(callback_called) > 0
        filename, current, total = callback_called[0]
        assert current == 1
        assert total == 1


class TestFormatPresets:
    """Test AudioFormatPresets"""

    def test_wav_44100_stereo(self):
        """Test 44.1kHz stereo preset"""
        format = cm.AudioFormatPresets.wav_44100_stereo()

        assert format.sample_rate == 44100.0
        assert format.channels_per_frame == 2
        assert format.bits_per_channel == 16
        assert format.format_id == 'lpcm'

    def test_wav_44100_mono(self):
        """Test 44.1kHz mono preset"""
        format = cm.AudioFormatPresets.wav_44100_mono()

        assert format.sample_rate == 44100.0
        assert format.channels_per_frame == 1
        assert format.bits_per_channel == 16

    def test_wav_48000_stereo(self):
        """Test 48kHz stereo preset"""
        format = cm.AudioFormatPresets.wav_48000_stereo()

        assert format.sample_rate == 48000.0
        assert format.channels_per_frame == 2
        assert format.bits_per_channel == 16

    def test_wav_96000_stereo(self):
        """Test 96kHz stereo preset"""
        format = cm.AudioFormatPresets.wav_96000_stereo()

        assert format.sample_rate == 96000.0
        assert format.channels_per_frame == 2
        assert format.bits_per_channel == 24


class TestAudioFileOperations:
    """Test audio file operations"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def amen_wav_path(self):
        """Fixture providing path to amen.wav test file"""
        path = os.path.join("tests", "amen.wav")
        if not os.path.exists(path):
            pytest.skip(f"Test audio file not found: {path}")
        return path

    @pytest.mark.skip(reason="trim_audio requires ExtendedAudioFile.write() support - TODO")
    def test_trim_audio_first_segment(self, amen_wav_path, temp_dir):
        """Test trimming first N seconds"""
        output_path = os.path.join(temp_dir, "trimmed.wav")

        # Trim first 1 second
        cm.trim_audio(amen_wav_path, output_path, start_time=0.0, end_time=1.0)

        assert os.path.exists(output_path)

        # Verify duration is approximately 1 second
        with cm.AudioFile(output_path) as audio:
            duration = audio.duration
            assert 0.9 < duration < 1.1  # Allow small tolerance

    @pytest.mark.skip(reason="trim_audio requires ExtendedAudioFile.write() support - TODO")
    def test_trim_audio_middle_segment(self, amen_wav_path, temp_dir):
        """Test trimming middle segment"""
        output_path = os.path.join(temp_dir, "trimmed.wav")

        # Trim from 0.5s to 1.5s
        cm.trim_audio(amen_wav_path, output_path, start_time=0.5, end_time=1.5)

        assert os.path.exists(output_path)

        with cm.AudioFile(output_path) as audio:
            duration = audio.duration
            assert 0.9 < duration < 1.1  # Should be ~1 second

    @pytest.mark.skip(reason="trim_audio requires ExtendedAudioFile.write() support - TODO")
    def test_trim_audio_skip_beginning(self, amen_wav_path, temp_dir):
        """Test trimming from start time to end"""
        output_path = os.path.join(temp_dir, "trimmed.wav")

        # Get original duration
        with cm.AudioFile(amen_wav_path) as audio:
            original_duration = audio.duration

        # Trim from 1 second to end
        cm.trim_audio(amen_wav_path, output_path, start_time=1.0)

        assert os.path.exists(output_path)

        with cm.AudioFile(output_path) as audio:
            trimmed_duration = audio.duration
            # Should be approximately 1 second shorter
            assert abs((original_duration - 1.0) - trimmed_duration) < 0.1


class TestUtilitiesIntegration:
    """Integration tests for utilities"""

    @pytest.fixture
    def amen_wav_path(self):
        """Fixture providing path to amen.wav test file"""
        path = os.path.join("tests", "amen.wav")
        if not os.path.exists(path):
            pytest.skip(f"Test audio file not found: {path}")
        return path

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.skipif(not cm.NUMPY_AVAILABLE, reason="NumPy not available")
    def test_workflow_analyze_and_convert(self, amen_wav_path, temp_dir):
        """Test complete workflow: analyze then convert"""
        # 1. Analyze original file
        info = cm.AudioAnalyzer.get_file_info(amen_wav_path)
        assert info['sample_rate'] == 44100.0

        # 2. Convert to mono
        output_path = os.path.join(temp_dir, "mono.wav")
        cm.convert_audio_file(
            amen_wav_path,
            output_path,
            cm.AudioFormatPresets.wav_44100_mono()
        )

        # 3. Analyze converted file
        mono_info = cm.AudioAnalyzer.get_file_info(output_path)
        assert mono_info['channels'] == 1
        assert mono_info['is_mono'] is True

        # 4. Check peak is preserved
        original_peak = info['peak_amplitude']
        mono_peak = mono_info['peak_amplitude']
        # Peaks should be similar (within 20% due to mono conversion)
        assert abs(original_peak - mono_peak) / original_peak < 0.2

    @pytest.mark.skip(reason="Requires trim_audio and sample rate conversion - TODO")
    def test_workflow_trim_and_convert(self, amen_wav_path, temp_dir):
        """Test workflow: trim then convert format"""
        # 1. Trim to first second
        trimmed_path = os.path.join(temp_dir, "trimmed.wav")
        cm.trim_audio(amen_wav_path, trimmed_path, start_time=0.0, end_time=1.0)

        # 2. Convert to 48kHz
        final_path = os.path.join(temp_dir, "final.wav")
        cm.convert_audio_file(
            trimmed_path,
            final_path,
            cm.AudioFormatPresets.wav_48000_stereo()
        )

        # 3. Verify result
        info = cm.AudioAnalyzer.get_file_info(final_path)
        assert 0.9 < info['duration'] < 1.1
        assert info['sample_rate'] == 48000.0
