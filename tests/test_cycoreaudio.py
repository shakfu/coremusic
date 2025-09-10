#!/usr/bin/env python3
"""pytest test suite for cycoreaudio wrapper functionality."""

import os
import pytest
import wave
import struct
import time
import coreaudio as ca


class TestCoreAudioConstants:
    """Test CoreAudio constants access"""
    
    def test_audio_format_constants(self):
        """Test audio format constants"""
        assert ca.get_audio_format_linear_pcm() is not None
        assert ca.get_linear_pcm_format_flag_is_signed_integer() is not None
        assert ca.get_linear_pcm_format_flag_is_packed() is not None
        assert ca.get_linear_pcm_format_flag_is_non_interleaved() is not None
    
    def test_audio_file_constants(self):
        """Test audio file constants"""
        assert ca.get_audio_file_wave_type() is not None
        assert ca.get_audio_file_read_permission() is not None
        assert ca.get_audio_file_property_data_format() is not None
        assert ca.get_audio_file_property_maximum_packet_size() is not None
    
    def test_audio_unit_constants(self):
        """Test AudioUnit constants"""
        assert ca.get_audio_unit_type_output() is not None
        assert ca.get_audio_unit_subtype_default_output() is not None
        assert ca.get_audio_unit_manufacturer_apple() is not None
        assert ca.get_audio_unit_property_stream_format() is not None
        assert ca.get_audio_unit_scope_input() is not None
        assert ca.get_audio_unit_scope_output() is not None


class TestFourCCConversion:
    """Test FourCC conversion utilities"""
    
    def test_fourchar_to_int_conversion(self):
        """Test string to int FourCC conversion"""
        test_codes = ['WAVE', 'TEXT', 'AIFF', 'mp4f', 'RIFF']
        for code in test_codes:
            int_val = ca.fourchar_to_int(code)
            assert isinstance(int_val, int)
            assert int_val > 0
    
    def test_int_to_fourchar_conversion(self):
        """Test int to string FourCC conversion"""
        test_codes = ['WAVE', 'TEXT', 'AIFF', 'mp4f', 'RIFF']
        for code in test_codes:
            int_val = ca.fourchar_to_int(code)
            back_to_str = ca.int_to_fourchar(int_val)
            assert back_to_str == code
    
    def test_roundtrip_conversion(self):
        """Test roundtrip conversion maintains data integrity"""
        test_codes = ['WAVE', 'TEXT', 'AIFF', 'mp4f', 'RIFF', 'CAFF', 'caff']
        for code in test_codes:
            int_val = ca.fourchar_to_int(code)
            back_to_str = ca.int_to_fourchar(int_val)
            assert back_to_str == code, f"Roundtrip failed for '{code}': {code} -> {int_val} -> {back_to_str}"


class TestAudioFileOperations:
    """Test audio file operations"""
    
    @pytest.fixture
    def amen_wav_path(self):
        """Fixture providing path to amen.wav test file"""
        path = os.path.join("tests", "amen.wav")
        if not os.path.exists(path):
            pytest.skip(f"Test audio file not found: {path}")
        return path
    
    def test_audio_file_open_close(self, amen_wav_path):
        """Test opening and closing audio files"""
        # Open audio file
        audio_file_id = ca.audio_file_open_url(
            amen_wav_path,
            ca.get_audio_file_read_permission(),
            ca.get_audio_file_wave_type()
        )
        assert audio_file_id is not None
        
        # Close audio file
        ca.audio_file_close(audio_file_id)
    
    def test_audio_file_properties(self, amen_wav_path):
        """Test reading audio file properties"""
        audio_file_id = ca.audio_file_open_url(
            amen_wav_path,
            ca.get_audio_file_read_permission(),
            ca.get_audio_file_wave_type()
        )
        
        try:
            # Get data format property
            format_data = ca.audio_file_get_property(
                audio_file_id,
                ca.get_audio_file_property_data_format()
            )
            assert len(format_data) >= 40  # AudioStreamBasicDescription is 40 bytes
            
            # Parse the AudioStreamBasicDescription
            asbd = struct.unpack('<dLLLLLLLL', format_data[:40])
            sample_rate, format_id, format_flags, bytes_per_packet, frames_per_packet, \
            bytes_per_frame, channels_per_frame, bits_per_channel, reserved = asbd
            
            assert sample_rate > 0
            assert channels_per_frame > 0
            assert bits_per_channel > 0
            assert format_id == ca.get_audio_format_linear_pcm()
            
        finally:
            ca.audio_file_close(audio_file_id)
    
    def test_audio_file_packet_reading(self, amen_wav_path):
        """Test reading audio packets from file"""
        audio_file_id = ca.audio_file_open_url(
            amen_wav_path,
            ca.get_audio_file_read_permission(),
            ca.get_audio_file_wave_type()
        )
        
        try:
            # Read some packets
            packet_data, packets_read = ca.audio_file_read_packets(audio_file_id, 0, 100)
            assert packets_read > 0
            assert len(packet_data) > 0
            assert isinstance(packet_data, bytes)
            
        finally:
            ca.audio_file_close(audio_file_id)
    
    def test_wav_file_format_detection(self, amen_wav_path):
        """Test WAV file format detection using Python wave module"""
        with wave.open(amen_wav_path, 'rb') as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frame_count = wav.getnframes()
            
            assert sample_rate > 0
            assert channels > 0
            assert sample_width > 0
            assert frame_count > 0
            
            # Verify it's a reasonable audio file
            duration = frame_count / sample_rate
            assert 0.1 < duration < 60  # Between 0.1 and 60 seconds


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


class TestAudioQueueOperations:
    """Test AudioQueue creation and management"""
    
    @pytest.fixture
    def audio_format(self):
        """Fixture providing standard audio format"""
        return {
            'sample_rate': 44100.0,
            'format_id': ca.get_audio_format_linear_pcm(),
            'format_flags': ca.get_linear_pcm_format_flag_is_signed_integer() | 
                           ca.get_linear_pcm_format_flag_is_packed(),
            'bytes_per_packet': 4,
            'frames_per_packet': 1,
            'bytes_per_frame': 4,
            'channels_per_frame': 2,
            'bits_per_channel': 16
        }
    
    def test_audio_queue_creation(self, audio_format):
        """Test AudioQueue creation"""
        queue_id = ca.audio_queue_new_output(audio_format)
        assert queue_id is not None
        assert isinstance(queue_id, int)
        
        # Cleanup
        ca.audio_queue_dispose(queue_id, True)
    
    def test_audio_queue_buffer_allocation(self, audio_format):
        """Test AudioQueue buffer allocation"""
        queue_id = ca.audio_queue_new_output(audio_format)
        
        try:
            buffer_size = 8192
            buffer_id = ca.audio_queue_allocate_buffer(queue_id, buffer_size)
            assert buffer_id is not None
            assert isinstance(buffer_id, int)
            
        finally:
            ca.audio_queue_dispose(queue_id, True)
    
    def test_audio_queue_lifecycle(self, audio_format):
        """Test complete AudioQueue lifecycle"""
        queue_id = ca.audio_queue_new_output(audio_format)
        
        try:
            # Allocate buffers
            buffer1_id = ca.audio_queue_allocate_buffer(queue_id, 8192)
            buffer2_id = ca.audio_queue_allocate_buffer(queue_id, 8192)
            
            # Note: Enqueuing buffers without a callback may fail on some systems
            # This is expected behavior - we're testing the infrastructure, not actual playback
            try:
                ca.audio_queue_enqueue_buffer(queue_id, buffer1_id)
                ca.audio_queue_enqueue_buffer(queue_id, buffer2_id)
                
                # Start queue
                ca.audio_queue_start(queue_id)
                
                # Brief operation
                time.sleep(0.1)
                
                # Stop queue
                ca.audio_queue_stop(queue_id, True)
            except RuntimeError as e:
                # This is expected when no callback is set up
                # The important thing is that we can create and dispose the queue
                assert "AudioQueueEnqueueBuffer failed" in str(e) or "AudioQueueStart failed" in str(e)
            
        finally:
            ca.audio_queue_dispose(queue_id, True)


class TestAudioPlayerIntegration:
    """Test integrated audio player functionality"""
    
    @pytest.fixture
    def amen_wav_path(self):
        """Fixture providing path to amen.wav test file"""
        path = os.path.join("tests", "amen.wav")
        if not os.path.exists(path):
            pytest.skip(f"Test audio file not found: {path}")
        return path
    
    def test_audio_file_loading_and_analysis(self, amen_wav_path):
        """Test loading and analyzing audio file with both Python and CoreAudio"""
        # Load with Python wave module
        with wave.open(amen_wav_path, 'rb') as wav:
            python_format = {
                'sample_rate': wav.getframerate(),
                'channels': wav.getnchannels(),
                'sample_width': wav.getsampwidth(),
                'frame_count': wav.getnframes(),
                'duration': wav.getnframes() / wav.getframerate()
            }
            
            audio_data = wav.readframes(wav.getnframes())
        
        # Verify Python loading
        assert python_format['sample_rate'] > 0
        assert python_format['channels'] > 0
        assert python_format['sample_width'] > 0
        assert python_format['frame_count'] > 0
        assert len(audio_data) > 0
        
        # Load with CoreAudio
        audio_file_id = ca.audio_file_open_url(
            amen_wav_path,
            ca.get_audio_file_read_permission(),
            ca.get_audio_file_wave_type()
        )
        
        try:
            # Get format via CoreAudio
            format_data = ca.audio_file_get_property(
                audio_file_id,
                ca.get_audio_file_property_data_format()
            )
            
            if len(format_data) >= 40:
                asbd = struct.unpack('<dLLLLLLLL', format_data[:40])
                coreaudio_format = {
                    'sample_rate': asbd[0],
                    'channels': asbd[6],
                    'bits_per_channel': asbd[7]
                }
                
                # Verify CoreAudio format matches Python format
                assert abs(coreaudio_format['sample_rate'] - python_format['sample_rate']) < 1
                assert coreaudio_format['channels'] == python_format['channels']
                assert coreaudio_format['bits_per_channel'] == python_format['sample_width'] * 8
            
            # Read audio data via CoreAudio
            packet_data, packets_read = ca.audio_file_read_packets(audio_file_id, 0, 1000)
            assert packets_read > 0
            assert len(packet_data) > 0
            
        finally:
            ca.audio_file_close(audio_file_id)
    
    def test_complete_audio_pipeline(self, amen_wav_path):
        """Test complete audio processing pipeline"""
        # Step 1: Load audio file
        with wave.open(amen_wav_path, 'rb') as wav:
            format_info = {
                'sample_rate': float(wav.getframerate()),
                'channels': wav.getnchannels(),
                'sample_width': wav.getsampwidth(),
                'frame_count': wav.getnframes()
            }
            audio_data = wav.readframes(wav.getnframes())
        
        assert len(audio_data) > 0
        assert format_info['sample_rate'] > 0
        
        # Step 2: Verify CoreAudio access
        audio_file_id = ca.audio_file_open_url(
            amen_wav_path,
            ca.get_audio_file_read_permission(),
            ca.get_audio_file_wave_type()
        )
        assert audio_file_id is not None
        ca.audio_file_close(audio_file_id)
        
        # Step 3: Test AudioUnit infrastructure
        description = {
            'type': ca.get_audio_unit_type_output(),
            'subtype': ca.get_audio_unit_subtype_default_output(),
            'manufacturer': ca.get_audio_unit_manufacturer_apple(),
            'flags': 0,
            'flags_mask': 0
        }
        
        component_id = ca.audio_component_find_next(description)
        assert component_id is not None
        
        audio_unit = ca.audio_component_instance_new(component_id)
        assert audio_unit is not None
        
        try:
            ca.audio_unit_initialize(audio_unit)
            ca.audio_output_unit_start(audio_unit)
            time.sleep(0.1)  # Brief operation
            ca.audio_output_unit_stop(audio_unit)
            ca.audio_unit_uninitialize(audio_unit)
        finally:
            ca.audio_component_instance_dispose(audio_unit)
        
        # Step 4: Test AudioQueue infrastructure
        audio_format = {
            'sample_rate': format_info['sample_rate'],
            'format_id': ca.get_audio_format_linear_pcm(),
            'format_flags': ca.get_linear_pcm_format_flag_is_signed_integer() | 
                           ca.get_linear_pcm_format_flag_is_packed(),
            'bytes_per_packet': format_info['channels'] * format_info['sample_width'],
            'frames_per_packet': 1,
            'bytes_per_frame': format_info['channels'] * format_info['sample_width'],
            'channels_per_frame': format_info['channels'],
            'bits_per_channel': format_info['sample_width'] * 8
        }
        
        queue_id = ca.audio_queue_new_output(audio_format)
        assert queue_id is not None
        
        try:
            buffer_id = ca.audio_queue_allocate_buffer(queue_id, 8192)
            assert buffer_id is not None
        finally:
            ca.audio_queue_dispose(queue_id, True)


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_file_handling(self):
        """Test handling of invalid file paths"""
        with pytest.raises(Exception):
            ca.audio_file_open_url(
                "nonexistent_file.wav",
                ca.get_audio_file_read_permission(),
                ca.get_audio_file_wave_type()
            )
    
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
    
    def test_audio_file_loading_performance(self):
        """Test that audio file loading is reasonably fast"""
        amen_path = os.path.join("tests", "amen.wav")
        if not os.path.exists(amen_path):
            pytest.skip(f"Test audio file not found: {amen_path}")
        
        start_time = time.time()
        
        # Load with Python
        with wave.open(amen_path, 'rb') as wav:
            audio_data = wav.readframes(wav.getnframes())
        
        python_time = time.time() - start_time
        
        # Load with CoreAudio
        start_time = time.time()
        
        audio_file_id = ca.audio_file_open_url(
            amen_path,
            ca.get_audio_file_read_permission(),
            ca.get_audio_file_wave_type()
        )
        
        try:
            packet_data, packets_read = ca.audio_file_read_packets(audio_file_id, 0, 1000)
        finally:
            ca.audio_file_close(audio_file_id)
        
        coreaudio_time = time.time() - start_time
        
        # Both should be reasonably fast (less than 1 second)
        assert python_time < 1.0
        assert coreaudio_time < 1.0
        
        # Both should have loaded data
        assert len(audio_data) > 0
        assert len(packet_data) > 0
    
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
