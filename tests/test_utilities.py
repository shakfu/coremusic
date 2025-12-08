"""Tests for high-level audio utilities."""

import os
import pytest
import tempfile
from pathlib import Path
import coremusic as cm
import coremusic.capi as capi
from coremusic.audio.analysis import AudioAnalyzer


class TestAudioAnalyzer:
    """Test AudioAnalyzer utilities"""

    @pytest.mark.skipif(not cm.NUMPY_AVAILABLE, reason="NumPy not available")
    def test_detect_silence(self, amen_wav_path):
        """Test silence detection"""
        silence_regions = AudioAnalyzer.detect_silence(
            amen_wav_path, threshold_db=-50, min_duration=0.1
        )
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
            silence_regions = AudioAnalyzer.detect_silence(
                audio, threshold_db=-50, min_duration=0.1
            )
            assert isinstance(silence_regions, list)

    @pytest.mark.skipif(not cm.NUMPY_AVAILABLE, reason="NumPy not available")
    def test_get_peak_amplitude(self, amen_wav_path):
        """Test peak amplitude detection"""
        peak = AudioAnalyzer.get_peak_amplitude(amen_wav_path)
        assert isinstance(peak, float)
        assert 0 <= peak <= 1.5

    @pytest.mark.skipif(not cm.NUMPY_AVAILABLE, reason="NumPy not available")
    def test_get_peak_amplitude_with_audio_file_object(self, amen_wav_path):
        """Test peak amplitude with AudioFile object"""
        with cm.AudioFile(amen_wav_path) as audio:
            peak = AudioAnalyzer.get_peak_amplitude(audio)
            assert isinstance(peak, float)
            assert peak > 0

    @pytest.mark.skipif(not cm.NUMPY_AVAILABLE, reason="NumPy not available")
    def test_calculate_rms(self, amen_wav_path):
        """Test RMS calculation"""
        rms = AudioAnalyzer.calculate_rms(amen_wav_path)
        assert isinstance(rms, float)
        assert 0 < rms < 1.0
        peak = AudioAnalyzer.get_peak_amplitude(amen_wav_path)
        assert rms <= peak

    @pytest.mark.skipif(not cm.NUMPY_AVAILABLE, reason="NumPy not available")
    def test_calculate_rms_with_audio_file_object(self, amen_wav_path):
        """Test RMS with AudioFile object"""
        with cm.AudioFile(amen_wav_path) as audio:
            rms = AudioAnalyzer.calculate_rms(audio)
            assert isinstance(rms, float)
            assert rms > 0

    def test_get_file_info(self, amen_wav_path):
        """Test file info extraction"""
        info = AudioAnalyzer.get_file_info(amen_wav_path)
        assert "path" in info
        assert "duration" in info
        assert "sample_rate" in info
        assert "format_id" in info
        assert "channels" in info
        assert "bits_per_channel" in info
        assert info["duration"] > 0
        assert info["sample_rate"] == 44100.0
        assert info["format_id"] == "lpcm"
        assert info["channels"] == 2
        assert info["is_stereo"] is True
        if cm.NUMPY_AVAILABLE:
            assert "peak_amplitude" in info
            assert "rms" in info


class TestBatchProcessing:
    """Test batch processing utilities"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_convert_audio_file_stereo_to_mono(self, amen_wav_path, temp_dir):
        """Test stereo to mono conversion"""
        output_path = os.path.join(temp_dir, "output.wav")
        output_format = cm.AudioFormatPresets.wav_44100_mono()
        cm.convert_audio_file(amen_wav_path, output_path, output_format)
        assert os.path.exists(output_path)
        with cm.AudioFile(output_path) as audio:
            format = audio.format
            assert format.sample_rate == 44100.0
            assert format.channels_per_frame == 1

    def test_convert_audio_file_sample_rate(self, amen_wav_path, temp_dir):
        """Test sample rate conversion (44.1kHz -> 48kHz)"""
        output_path = os.path.join(temp_dir, "output_48k.wav")
        output_format = cm.AudioFormatPresets.wav_48000_stereo()
        cm.convert_audio_file(amen_wav_path, output_path, output_format)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        with cm.AudioFile(output_path) as audio:
            format = audio.format
            assert format.sample_rate == 48000.0
            assert format.channels_per_frame == 2

    def test_convert_audio_file_bit_depth(self, amen_wav_path, temp_dir):
        """Test bit depth conversion (16-bit -> 24-bit)"""
        output_path = os.path.join(temp_dir, "output_24bit.wav")
        output_format = cm.AudioFormatPresets.wav_96000_stereo()
        cm.convert_audio_file(amen_wav_path, output_path, output_format)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        with cm.AudioFile(output_path) as audio:
            format = audio.format
            assert format.sample_rate == 96000.0
            assert format.bits_per_channel == 24
            assert format.channels_per_frame == 2

    def test_convert_audio_file_combined_conversions(self, amen_wav_path, temp_dir):
        """Test combined sample rate and channel conversion (44.1kHz stereo -> 48kHz mono)"""
        output_path = os.path.join(temp_dir, "output_48k_mono.wav")
        output_format = cm.AudioFormat(
            sample_rate=48000.0,
            format_id="lpcm",
            format_flags=12,
            bytes_per_packet=2,
            frames_per_packet=1,
            bytes_per_frame=2,
            channels_per_frame=1,
            bits_per_channel=16,
        )
        cm.convert_audio_file(amen_wav_path, output_path, output_format)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        with cm.AudioFile(output_path) as audio:
            format = audio.format
            assert format.sample_rate == 48000.0
            assert format.channels_per_frame == 1
            assert format.bits_per_channel == 16

    def test_batch_convert_single_file(self, amen_wav_path, temp_dir):
        """Test batch conversion with single file"""
        import shutil

        input_file = os.path.join(temp_dir, "input.wav")
        shutil.copy(amen_wav_path, input_file)
        output_format = cm.AudioFormatPresets.wav_44100_mono()
        output_dir = os.path.join(temp_dir, "output")
        converted = cm.batch_convert(
            input_pattern=f"{temp_dir}/input.wav",
            output_format=output_format,
            output_dir=output_dir,
            output_extension="wav",
        )
        assert len(converted) == 1
        assert os.path.exists(converted[0])
        with cm.AudioFile(converted[0]) as audio:
            assert audio.format.channels_per_frame == 1

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
            progress_callback=progress_callback,
        )
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
        assert format.format_id == "lpcm"

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

    @pytest.mark.skip(
        reason="trim_audio requires ExtendedAudioFile.write() support - TODO"
    )
    def test_trim_audio_first_segment(self, amen_wav_path, temp_dir):
        """Test trimming first N seconds"""
        output_path = os.path.join(temp_dir, "trimmed.wav")
        capi.trim_audio(amen_wav_path, output_path, start_time=0.0, end_time=1.0)
        assert os.path.exists(output_path)
        with cm.AudioFile(output_path) as audio:
            duration = audio.duration
            assert 0.9 < duration < 1.1

    @pytest.mark.skip(
        reason="trim_audio requires ExtendedAudioFile.write() support - TODO"
    )
    def test_trim_audio_middle_segment(self, amen_wav_path, temp_dir):
        """Test trimming middle segment"""
        output_path = os.path.join(temp_dir, "trimmed.wav")
        capi.trim_audio(amen_wav_path, output_path, start_time=0.5, end_time=1.5)
        assert os.path.exists(output_path)
        with cm.AudioFile(output_path) as audio:
            duration = audio.duration
            assert 0.9 < duration < 1.1

    @pytest.mark.skip(
        reason="trim_audio requires ExtendedAudioFile.write() support - TODO"
    )
    def test_trim_audio_skip_beginning(self, amen_wav_path, temp_dir):
        """Test trimming from start time to end"""
        output_path = os.path.join(temp_dir, "trimmed.wav")
        with cm.AudioFile(amen_wav_path) as audio:
            original_duration = audio.duration
        capi.trim_audio(amen_wav_path, output_path, start_time=1.0)
        assert os.path.exists(output_path)
        with cm.AudioFile(output_path) as audio:
            trimmed_duration = audio.duration
            assert abs(original_duration - 1.0 - trimmed_duration) < 0.1


class TestUtilitiesIntegration:
    """Integration tests for utilities"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.skipif(not cm.NUMPY_AVAILABLE, reason="NumPy not available")
    def test_workflow_analyze_and_convert(self, amen_wav_path, temp_dir):
        """Test complete workflow: analyze then convert"""
        info = AudioAnalyzer.get_file_info(amen_wav_path)
        assert info["sample_rate"] == 44100.0
        output_path = os.path.join(temp_dir, "mono.wav")
        cm.convert_audio_file(
            amen_wav_path, output_path, cm.AudioFormatPresets.wav_44100_mono()
        )
        mono_info = AudioAnalyzer.get_file_info(output_path)
        assert mono_info["channels"] == 1
        assert mono_info["is_mono"] is True
        original_peak = info["peak_amplitude"]
        mono_peak = mono_info["peak_amplitude"]
        assert abs(original_peak - mono_peak) / original_peak < 0.2

    @pytest.mark.skip(reason="Requires trim_audio and sample rate conversion - TODO")
    def test_workflow_trim_and_convert(self, amen_wav_path, temp_dir):
        """Test workflow: trim then convert format"""
        trimmed_path = os.path.join(temp_dir, "trimmed.wav")
        capi.trim_audio(amen_wav_path, trimmed_path, start_time=0.0, end_time=1.0)
        final_path = os.path.join(temp_dir, "final.wav")
        cm.convert_audio_file(
            trimmed_path, final_path, cm.AudioFormatPresets.wav_48000_stereo()
        )
        info = AudioAnalyzer.get_file_info(final_path)
        assert 0.9 < info["duration"] < 1.1
        assert info["sample_rate"] == 48000.0


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
        mixer_node = chain.add_effect("aumi", "3dem", "appl")
        assert isinstance(mixer_node, int)
        assert chain.node_count == 1

    def test_connect_nodes(self):
        """Test connecting nodes in the chain"""
        chain = cm.AudioEffectsChain()
        mixer_node = chain.add_effect("aumi", "3dem", "appl")
        output_node = chain.add_output()
        chain.connect(mixer_node, output_node)

    def test_remove_node(self):
        """Test removing a node from the chain"""
        chain = cm.AudioEffectsChain()
        mixer_node = chain.add_effect("aumi", "3dem", "appl")
        assert chain.node_count == 1
        chain.remove_node(mixer_node)
        assert chain.node_count == 0

    def test_chain_lifecycle(self):
        """Test complete chain lifecycle (open, initialize, start, stop)"""
        chain = cm.AudioEffectsChain()
        output_node = chain.add_output()
        chain.open()
        assert chain.is_open is True
        chain.initialize()
        assert chain.is_initialized is True
        try:
            chain.start()
            assert chain.is_running is True
            chain.stop()
            assert chain.is_running is False
        except Exception:
            pass
        finally:
            chain.dispose()

    def test_context_manager(self):
        """Test using AudioEffectsChain as context manager"""
        with cm.AudioEffectsChain() as chain:
            output_node = chain.add_output()
            assert chain.node_count == 1

    def test_create_simple_effect_chain(self):
        """Test creating a simple linear effects chain"""
        chain = cm.create_simple_effect_chain([("aumi", "3dem", "appl")])
        assert chain.node_count == 2

    def test_create_multi_effect_chain(self):
        """Test creating a multi-effect chain"""
        chain = cm.create_simple_effect_chain(
            [("aumi", "3dem", "appl"), ("aumi", "mxmx", "appl")]
        )
        assert chain.node_count == 3

    @pytest.mark.skip(reason="Requires specific AudioUnit availability")
    def test_reverb_eq_chain(self):
        """Test creating a reverb + EQ effect chain"""
        chain = cm.create_simple_effect_chain(
            [("aumu", "rvb2", "appl"), ("aufx", "eqal", "appl")]
        )
        assert chain.node_count == 3
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
        assert len(units) > 0
        assert isinstance(units, list)
        first_unit = units[0]
        assert "name" in first_unit
        assert "type" in first_unit
        assert "subtype" in first_unit
        assert "manufacturer" in first_unit

    def test_find_audio_unit_by_name_audelay(self):
        """Test finding AUDelay by name"""
        component = cm.find_audio_unit_by_name("AUDelay")
        assert component is not None
        from coremusic.objects import AudioComponent

        assert isinstance(component, AudioComponent)
        desc = component._description
        assert desc.type == "aufx"
        assert desc.subtype == "dely"
        assert desc.manufacturer == "appl"
        unit = component.create_instance()
        assert unit is not None
        unit.dispose()

    def test_find_audio_unit_by_name_case_insensitive(self):
        """Test case-insensitive name matching"""
        component1 = cm.find_audio_unit_by_name("audelay")
        component2 = cm.find_audio_unit_by_name("AUDELAY")
        component3 = cm.find_audio_unit_by_name("AuDelay")
        assert component1 is not None
        assert component2 is not None
        assert component3 is not None
        assert component1._description.type == component2._description.type
        assert component1._description.subtype == component2._description.subtype

    def test_find_audio_unit_by_name_not_found(self):
        """Test searching for non-existent AudioUnit"""
        component = cm.find_audio_unit_by_name("NonExistentAudioUnit12345")
        assert component is None

    def test_find_audio_unit_by_name_partial_match(self):
        """Test partial name matching"""
        component = cm.find_audio_unit_by_name("Delay")
        assert component is not None
        from coremusic.objects import AudioComponent

        assert isinstance(component, AudioComponent)

    def test_audio_effects_chain_add_effect_by_name(self):
        """Test adding effect to chain by name"""
        chain = cm.AudioEffectsChain()
        delay_node = chain.add_effect_by_name("AUDelay")
        assert delay_node is not None
        assert isinstance(delay_node, int)
        assert chain.node_count == 1
        chain.dispose()

    def test_audio_effects_chain_add_effect_by_name_not_found(self):
        """Test adding non-existent effect by name"""
        chain = cm.AudioEffectsChain()
        node = chain.add_effect_by_name("NonExistentEffect12345")
        assert node is None
        assert chain.node_count == 0
        chain.dispose()

    def test_audio_effects_chain_by_name_complete_workflow(self):
        """Test complete workflow using name-based effect addition"""
        chain = cm.AudioEffectsChain()
        delay_node = chain.add_effect_by_name("AUDelay")
        output_node = chain.add_output()
        assert delay_node is not None
        assert output_node is not None
        assert chain.node_count == 2
        chain.connect(delay_node, output_node)
        chain.dispose()

    def test_get_audiounit_names(self):
        """Test getting list of all AudioUnit names"""
        names = cm.get_audiounit_names()
        assert len(names) > 0
        assert isinstance(names, list)
        assert all(isinstance(name, str) for name in names)
        assert any("AUDelay" in name for name in names)

    def test_get_audiounit_names_filter(self):
        """Test filtering AudioUnit names by type"""
        effects = cm.get_audiounit_names(filter_type="aufx")
        assert len(effects) > 0
        assert isinstance(effects, list)
        assert all(isinstance(name, str) for name in effects)

    def test_get_audiounit_names_usage(self):
        """Test practical usage of get_audiounit_names"""
        names = cm.get_audiounit_names()
        delays = [name for name in names if "delay" in name.lower()]
        assert len(delays) > 0
        audelay_available = any("AUDelay" in name for name in names)
        assert audelay_available is True


class TestParseAudioStreamBasicDescription:
    """Tests for parse_audio_stream_basic_description utility"""

    def test_parse_audio_stream_basic_description(self, amen_wav_path):
        """Test parsing AudioStreamBasicDescription from audio file"""
        # Open test file
        file_id = capi.audio_file_open_url(amen_wav_path)

        try:
            # Get format data
            format_data = capi.audio_file_get_property(
                file_id, capi.get_audio_file_property_data_format()
            )

            # Parse it
            asbd = cm.parse_audio_stream_basic_description(format_data)

            # Verify all expected keys are present
            expected_keys = {
                "sample_rate",
                "format_id",
                "format_flags",
                "bytes_per_packet",
                "frames_per_packet",
                "bytes_per_frame",
                "channels_per_frame",
                "bits_per_channel",
                "reserved",
            }
            assert set(asbd.keys()) == expected_keys

            # Verify data types
            assert isinstance(asbd["sample_rate"], float)
            assert isinstance(asbd["format_id"], str)
            assert isinstance(asbd["format_flags"], int)
            assert isinstance(asbd["channels_per_frame"], int)

            # Verify expected values for amen.wav (44.1kHz, 16-bit, stereo)
            assert asbd["sample_rate"] == 44100.0
            assert asbd["format_id"] == "lpcm"
            assert asbd["channels_per_frame"] == 2
            assert asbd["bits_per_channel"] == 16

        finally:
            capi.audio_file_close(file_id)

    def test_parse_audio_stream_basic_description_invalid_length(self):
        """Test that invalid length raises ValueError"""
        with pytest.raises(ValueError, match="must be exactly 40 bytes"):
            cm.parse_audio_stream_basic_description(b"too short")

    def test_parse_audio_stream_basic_description_matches_oo_api(self, amen_wav_path):
        """Test that parsed ASBD matches object-oriented API"""
        # Get format using functional API
        file_id = capi.audio_file_open_url(amen_wav_path)
        format_data = capi.audio_file_get_property(
            file_id, capi.get_audio_file_property_data_format()
        )
        asbd = cm.parse_audio_stream_basic_description(format_data)
        capi.audio_file_close(file_id)

        # Get format using OO API
        with cm.AudioFile(amen_wav_path) as audio:
            fmt = audio.format

            # Compare values
            assert asbd["sample_rate"] == fmt.sample_rate
            assert asbd["format_id"] == fmt.format_id
            assert asbd["format_flags"] == fmt.format_flags
            assert asbd["channels_per_frame"] == fmt.channels_per_frame
            assert asbd["bits_per_channel"] == fmt.bits_per_channel
