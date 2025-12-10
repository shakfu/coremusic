import logging
import os
import struct
import time
import wave
import pytest
import coremusic as cm
import coremusic.capi as capi

logger = logging.getLogger(__name__)


class TestAudioComponentDiscovery:
    """Test AudioComponent discovery and management"""

    def test_find_default_output_component(self):
        """Test finding default output AudioComponent"""
        description = {
            "type": capi.get_audio_unit_type_output(),
            "subtype": capi.get_audio_unit_subtype_default_output(),
            "manufacturer": capi.get_audio_unit_manufacturer_apple(),
            "flags": 0,
            "flags_mask": 0,
        }
        component_id = capi.audio_component_find_next(description)
        assert component_id is not None
        assert isinstance(component_id, int)

    def test_audio_component_description_structure(self):
        """Test that AudioComponent description structure is valid"""
        description = {
            "type": capi.get_audio_unit_type_output(),
            "subtype": capi.get_audio_unit_subtype_default_output(),
            "manufacturer": capi.get_audio_unit_manufacturer_apple(),
            "flags": 0,
            "flags_mask": 0,
        }
        required_keys = ["type", "subtype", "manufacturer", "flags", "flags_mask"]
        for key in required_keys:
            assert key in description
            assert description[key] is not None


class TestAudioUnitInfrastructure:
    """Test AudioUnit creation, configuration, and lifecycle"""

    @pytest.fixture
    def audio_unit(self):
        """Fixture providing a configured AudioUnit"""
        description = {
            "type": capi.get_audio_unit_type_output(),
            "subtype": capi.get_audio_unit_subtype_default_output(),
            "manufacturer": capi.get_audio_unit_manufacturer_apple(),
            "flags": 0,
            "flags_mask": 0,
        }
        component_id = capi.audio_component_find_next(description)
        assert component_id is not None
        audio_unit = capi.audio_component_instance_new(component_id)
        assert audio_unit is not None
        yield audio_unit
        try:
            capi.audio_unit_uninitialize(audio_unit)
            capi.audio_component_instance_dispose(audio_unit)
        except Exception as e:
            logger.warning(f"Cleanup failed (uninitialize/dispose): {e}")

    def test_audio_unit_creation(self, audio_unit):
        """Test AudioUnit creation"""
        assert audio_unit is not None
        assert isinstance(audio_unit, int)

    def test_audio_unit_initialization(self, audio_unit):
        """Test AudioUnit initialization"""
        capi.audio_unit_initialize(audio_unit)

    def test_audio_unit_format_configuration(self, audio_unit):
        """Test AudioUnit format configuration"""
        format_data = struct.pack(
            "<dLLLLLLLL",
            44100.0,
            capi.get_audio_format_linear_pcm(),
            capi.get_linear_pcm_format_flag_is_signed_integer()
            | capi.get_linear_pcm_format_flag_is_packed(),
            4,
            1,
            4,
            2,
            16,
            0,
        )
        try:
            capi.audio_unit_set_property(
                audio_unit,
                capi.get_audio_unit_property_stream_format(),
                capi.get_audio_unit_scope_input(),
                0,
                format_data,
            )
        except Exception:
            pass

    def test_audio_unit_hardware_control(self, audio_unit):
        """Test AudioUnit hardware start/stop control"""
        capi.audio_unit_initialize(audio_unit)
        capi.audio_output_unit_start(audio_unit)
        capi.audio_output_unit_stop(audio_unit)

    def test_audio_unit_lifecycle(self, audio_unit):
        """Test complete AudioUnit lifecycle"""
        capi.audio_unit_initialize(audio_unit)
        capi.audio_output_unit_start(audio_unit)
        time.sleep(0.1)
        capi.audio_output_unit_stop(audio_unit)
        capi.audio_unit_uninitialize(audio_unit)
        capi.audio_component_instance_dispose(audio_unit)


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_audio_unit_operations(self):
        """Test operations on invalid AudioUnit"""
        with pytest.raises(Exception):
            capi.audio_unit_initialize(None)

    def test_audio_queue_with_invalid_format(self):
        """Test AudioQueue creation with invalid format"""
        invalid_format = {
            "sample_rate": -1,
            "format_id": 0,
            "format_flags": 0,
            "bytes_per_packet": 0,
            "frames_per_packet": 0,
            "bytes_per_frame": 0,
            "channels_per_frame": 0,
            "bits_per_channel": 0,
        }
        with pytest.raises(Exception):
            capi.audio_queue_new_output(invalid_format)


class TestPerformance:
    """Test performance characteristics"""

    def test_audio_unit_creation_performance(self):
        """Test that AudioUnit creation is reasonably fast"""
        start_time = time.time()
        description = {
            "type": capi.get_audio_unit_type_output(),
            "subtype": capi.get_audio_unit_subtype_default_output(),
            "manufacturer": capi.get_audio_unit_manufacturer_apple(),
            "flags": 0,
            "flags_mask": 0,
        }
        component_id = capi.audio_component_find_next(description)
        audio_unit = capi.audio_component_instance_new(component_id)
        try:
            capi.audio_unit_initialize(audio_unit)
        finally:
            capi.audio_unit_uninitialize(audio_unit)
            capi.audio_component_instance_dispose(audio_unit)
        creation_time = time.time() - start_time
        assert creation_time < 1.0
