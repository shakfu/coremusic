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

    @pytest.mark.skip(reason="AudioConverter sample rate conversion requires callback-based API - TODO")
    def test_convert_audio_file_sample_rate(self, amen_wav_path, temp_dir):
        """Test sample rate conversion (44.1kHz -> 48kHz)"""
        # Note: Sample rate conversion with AudioConverter requires the callback-based
        # AudioConverterFillComplexBuffer API, not the simple buffer conversion.
        # This is a known limitation that should be addressed in a future update.
        output_path = os.path.join(temp_dir, "output_48k.wav")
        output_format = cm.AudioFormatPresets.wav_48000_stereo()

        cm.convert_audio_file(amen_wav_path, output_path, output_format)

        # Verify output file was created
        assert os.path.exists(output_path)

        # Verify format
        with cm.AudioFile(output_path) as audio:
            format = audio.format
            assert format.sample_rate == 48000.0
            assert format.channels_per_frame == 2  # Still stereo

    @pytest.mark.skip(reason="AudioConverter bit depth conversion requires callback-based API - TODO")
    def test_convert_audio_file_bit_depth(self, amen_wav_path, temp_dir):
        """Test bit depth conversion (16-bit -> 24-bit)"""
        # Note: Bit depth conversion with AudioConverter may require the callback-based
        # AudioConverterFillComplexBuffer API for some format combinations.
        output_path = os.path.join(temp_dir, "output_24bit.wav")
        output_format = cm.AudioFormatPresets.wav_96000_stereo()  # 24-bit

        cm.convert_audio_file(amen_wav_path, output_path, output_format)

        # Verify output file was created
        assert os.path.exists(output_path)

        # Verify format
        with cm.AudioFile(output_path) as audio:
            format = audio.format
            assert format.sample_rate == 96000.0
            assert format.bits_per_channel == 24
            assert format.channels_per_frame == 2

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


class TestAudioEffectsChain:
    """Test AudioEffectsChain functionality"""

    def test_create_effects_chain(self):
        """Test creating an empty effects chain"""
        chain = cm.AudioEffectsChain()
        assert chain is not None
        assert chain.node_count == 0

    def test_add_output_node(self):
        """Test adding an output node"""
        chain = cm.AudioEffectsChain()
        output_node = chain.add_output()
        assert isinstance(output_node, int)
        assert chain.node_count == 1

    def test_add_effect_node(self):
        """Test adding an effect node"""
        chain = cm.AudioEffectsChain()
        # Add a mixer unit (common and should always exist)
        mixer_node = chain.add_effect('aumi', '3dem', 'appl')
        assert isinstance(mixer_node, int)
        assert chain.node_count == 1

    def test_connect_nodes(self):
        """Test connecting nodes in the chain"""
        chain = cm.AudioEffectsChain()
        mixer_node = chain.add_effect('aumi', '3dem', 'appl')
        output_node = chain.add_output()

        # Should not raise an exception
        chain.connect(mixer_node, output_node)

    def test_remove_node(self):
        """Test removing a node from the chain"""
        chain = cm.AudioEffectsChain()
        mixer_node = chain.add_effect('aumi', '3dem', 'appl')
        assert chain.node_count == 1

        chain.remove_node(mixer_node)
        assert chain.node_count == 0

    def test_chain_lifecycle(self):
        """Test complete chain lifecycle (open, initialize, start, stop)"""
        chain = cm.AudioEffectsChain()
        output_node = chain.add_output()

        # Open the graph
        chain.open()
        assert chain.is_open is True

        # Initialize
        chain.initialize()
        assert chain.is_initialized is True

        # Start (may require audio hardware)
        try:
            chain.start()
            assert chain.is_running is True

            # Stop
            chain.stop()
            assert chain.is_running is False
        except Exception:
            # Hardware may not be available in CI
            pass
        finally:
            chain.dispose()

    def test_context_manager(self):
        """Test using AudioEffectsChain as context manager"""
        with cm.AudioEffectsChain() as chain:
            output_node = chain.add_output()
            assert chain.node_count == 1
        # Chain should be disposed after context exit

    def test_create_simple_effect_chain(self):
        """Test creating a simple linear effects chain"""
        chain = cm.create_simple_effect_chain([
            ('aumi', '3dem', 'appl'),  # 3D Mixer
        ])

        # Should have 2 nodes: mixer + output
        assert chain.node_count == 2

    def test_create_multi_effect_chain(self):
        """Test creating a multi-effect chain"""
        chain = cm.create_simple_effect_chain([
            ('aumi', '3dem', 'appl'),  # 3D Mixer
            ('aumi', 'mxmx', 'appl'),  # Matrix Mixer
        ])

        # Should have 3 nodes: 2 effects + output
        assert chain.node_count == 3

    @pytest.mark.skip(reason="Requires specific AudioUnit availability")
    def test_reverb_eq_chain(self):
        """Test creating a reverb + EQ effect chain"""
        # Note: This test is skipped because AudioUnit availability varies
        # On different macOS versions and configurations
        chain = cm.create_simple_effect_chain([
            ('aumu', 'rvb2', 'appl'),  # Reverb
            ('aufx', 'eqal', 'appl'),  # EQ
        ])

        assert chain.node_count == 3  # reverb + eq + output

        # Try to initialize
        try:
            chain.open().initialize()
            assert chain.is_initialized
        finally:
            chain.dispose()


class TestAudioUnitDiscovery:
    """Test AudioUnit discovery by name"""

    def test_list_available_audio_units(self):
        """Test listing all available AudioUnits"""
        units = cm.list_available_audio_units()

        # Should find at least some AudioUnits on macOS
        assert len(units) > 0
        assert isinstance(units, list)

        # Check structure of first unit
        first_unit = units[0]
        assert 'name' in first_unit
        assert 'type' in first_unit
        assert 'subtype' in first_unit
        assert 'manufacturer' in first_unit

    def test_find_audio_unit_by_name_audelay(self):
        """Test finding AUDelay by name"""
        codes = cm.find_audio_unit_by_name('AUDelay')

        # AUDelay should always be available on macOS
        assert codes is not None
        assert len(codes) == 3  # (type, subtype, manufacturer)

        type_code, subtype_code, manufacturer = codes
        assert isinstance(type_code, str)
        assert isinstance(subtype_code, str)
        assert isinstance(manufacturer, str)

        # AUDelay is an audio effect ('aufx'), delay type ('dely'), Apple manufacturer ('appl')
        assert type_code == 'aufx'
        assert subtype_code == 'dely'
        assert manufacturer == 'appl'

    def test_find_audio_unit_by_name_case_insensitive(self):
        """Test case-insensitive name matching"""
        # These should all find the same unit
        codes1 = cm.find_audio_unit_by_name('audelay')
        codes2 = cm.find_audio_unit_by_name('AUDELAY')
        codes3 = cm.find_audio_unit_by_name('AuDelay')

        assert codes1 is not None
        assert codes1 == codes2 == codes3

    def test_find_audio_unit_by_name_not_found(self):
        """Test searching for non-existent AudioUnit"""
        codes = cm.find_audio_unit_by_name('NonExistentAudioUnit12345')
        assert codes is None

    def test_find_audio_unit_by_name_partial_match(self):
        """Test partial name matching"""
        # Search for 'Delay' should find an AudioUnit containing 'Delay'
        codes = cm.find_audio_unit_by_name('Delay')
        assert codes is not None

    def test_audio_effects_chain_add_effect_by_name(self):
        """Test adding effect to chain by name"""
        chain = cm.AudioEffectsChain()

        # Add AUDelay by name
        delay_node = chain.add_effect_by_name('AUDelay')
        assert delay_node is not None
        assert isinstance(delay_node, int)
        assert chain.node_count == 1

        chain.dispose()

    def test_audio_effects_chain_add_effect_by_name_not_found(self):
        """Test adding non-existent effect by name"""
        chain = cm.AudioEffectsChain()

        # Try to add non-existent effect
        node = chain.add_effect_by_name('NonExistentEffect12345')
        assert node is None
        assert chain.node_count == 0

        chain.dispose()

    def test_audio_effects_chain_by_name_complete_workflow(self):
        """Test complete workflow using name-based effect addition"""
        chain = cm.AudioEffectsChain()

        # Add effects by name
        delay_node = chain.add_effect_by_name('AUDelay')
        output_node = chain.add_output()

        assert delay_node is not None
        assert output_node is not None
        assert chain.node_count == 2

        # Connect
        chain.connect(delay_node, output_node)

        # Dispose
        chain.dispose()
