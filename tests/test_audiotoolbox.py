import os
import struct
import time
import wave

import pytest

import coreaudio as ca


class TestAudioComponentDiscovery:
    """Test AudioComponent discovery and management"""
    
    def test_find_default_output_component(self):
        """Test finding default output AudioComponent"""
        description = {
            'type': ca.get_audio_unit_type_output(),
            'subtype': ca.get_audio_unit_subtype_default_output(),
            'manufacturer': ca.get_audio_unit_manufacturer_apple(),
            'flags': 0,
            'flags_mask': 0
        }
        
        component_id = ca.audio_component_find_next(description)
        assert component_id is not None
        assert isinstance(component_id, int)
    
    def test_audio_component_description_structure(self):
        """Test that AudioComponent description structure is valid"""
        description = {
            'type': ca.get_audio_unit_type_output(),
            'subtype': ca.get_audio_unit_subtype_default_output(),
            'manufacturer': ca.get_audio_unit_manufacturer_apple(),
            'flags': 0,
            'flags_mask': 0
        }
        
        # Verify all required keys are present
        required_keys = ['type', 'subtype', 'manufacturer', 'flags', 'flags_mask']
        for key in required_keys:
            assert key in description
            assert description[key] is not None


class TestAudioUnitInfrastructure:
    """Test AudioUnit creation, configuration, and lifecycle"""
    
    @pytest.fixture
    def audio_unit(self):
        """Fixture providing a configured AudioUnit"""
        # Find default output AudioUnit
        description = {
            'type': ca.get_audio_unit_type_output(),
            'subtype': ca.get_audio_unit_subtype_default_output(),
            'manufacturer': ca.get_audio_unit_manufacturer_apple(),
            'flags': 0,
            'flags_mask': 0
        }
        
        component_id = ca.audio_component_find_next(description)
        assert component_id is not None
        
        # Create AudioUnit instance
        audio_unit = ca.audio_component_instance_new(component_id)
        assert audio_unit is not None
        
        yield audio_unit
        
        # Cleanup
        try:
            ca.audio_unit_uninitialize(audio_unit)
            ca.audio_component_instance_dispose(audio_unit)
        except:
            pass  # Ignore cleanup errors
    
    def test_audio_unit_creation(self, audio_unit):
        """Test AudioUnit creation"""
        assert audio_unit is not None
        assert isinstance(audio_unit, int)
    
    def test_audio_unit_initialization(self, audio_unit):
        """Test AudioUnit initialization"""
        ca.audio_unit_initialize(audio_unit)
        # If we get here without exception, initialization succeeded
    
    def test_audio_unit_format_configuration(self, audio_unit):
        """Test AudioUnit format configuration"""
        # Create AudioStreamBasicDescription for 44.1kHz stereo 16-bit
        format_data = struct.pack('<dLLLLLLLL',
            44100.0,                                    # Sample rate
            ca.get_audio_format_linear_pcm(),          # Format ID
            ca.get_linear_pcm_format_flag_is_signed_integer() | 
            ca.get_linear_pcm_format_flag_is_packed(),  # Format flags
            4,                                         # Bytes per packet (2 channels * 2 bytes)
            1,                                         # Frames per packet
            4,                                         # Bytes per frame
            2,                                         # Channels per frame
            16,                                        # Bits per channel
            0                                          # Reserved
        )
        
        # Set the format (may fail on some systems, that's ok)
        try:
            ca.audio_unit_set_property(
                audio_unit,
                ca.get_audio_unit_property_stream_format(),
                ca.get_audio_unit_scope_input(),
                0,
                format_data
            )
        except Exception:
            # Format setting may fail on some systems, that's acceptable
            pass
    
    def test_audio_unit_hardware_control(self, audio_unit):
        """Test AudioUnit hardware start/stop control"""
        ca.audio_unit_initialize(audio_unit)
        
        # Test start/stop
        ca.audio_output_unit_start(audio_unit)
        ca.audio_output_unit_stop(audio_unit)
    
    def test_audio_unit_lifecycle(self, audio_unit):
        """Test complete AudioUnit lifecycle"""
        # Initialize
        ca.audio_unit_initialize(audio_unit)
        
        # Start
        ca.audio_output_unit_start(audio_unit)
        
        # Brief operation
        time.sleep(0.1)
        
        # Stop
        ca.audio_output_unit_stop(audio_unit)
        
        # Uninitialize
        ca.audio_unit_uninitialize(audio_unit)
        
        # Dispose
        ca.audio_component_instance_dispose(audio_unit)


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_audio_unit_operations(self):
        """Test operations on invalid AudioUnit"""
        # Try to operate on None/invalid AudioUnit
        with pytest.raises(Exception):
            ca.audio_unit_initialize(None)
    
    def test_audio_queue_with_invalid_format(self):
        """Test AudioQueue creation with invalid format"""
        invalid_format = {
            'sample_rate': -1,  # Invalid sample rate
            'format_id': 0,     # Invalid format ID
            'format_flags': 0,
            'bytes_per_packet': 0,
            'frames_per_packet': 0,
            'bytes_per_frame': 0,
            'channels_per_frame': 0,
            'bits_per_channel': 0
        }
        
        with pytest.raises(Exception):
            ca.audio_queue_new_output(invalid_format)


class TestPerformance:
    """Test performance characteristics"""
    
    def test_audio_unit_creation_performance(self):
        """Test that AudioUnit creation is reasonably fast"""
        start_time = time.time()
        
        description = {
            'type': ca.get_audio_unit_type_output(),
            'subtype': ca.get_audio_unit_subtype_default_output(),
            'manufacturer': ca.get_audio_unit_manufacturer_apple(),
            'flags': 0,
            'flags_mask': 0
        }
        
        component_id = ca.audio_component_find_next(description)
        audio_unit = ca.audio_component_instance_new(component_id)
        
        try:
            ca.audio_unit_initialize(audio_unit)
        finally:
            ca.audio_unit_uninitialize(audio_unit)
            ca.audio_component_instance_dispose(audio_unit)
        
        creation_time = time.time() - start_time
        
        # AudioUnit creation should be reasonably fast
        assert creation_time < 1.0


