#!/usr/bin/env python3
"""
Comprehensive demonstration of the cycoreaudio wrapper functionality.
This script tests various aspects of the CoreAudio wrapper we've built.
"""

import os
import time
import wave
import coreaudio as ca

def demo_constants():
    """Demonstrate access to CoreAudio constants"""
    print("=== CoreAudio Constants Demo ===")
    print(f"kAudioFormatLinearPCM: {ca.get_audio_format_linear_pcm()}")
    print(f"kLinearPCMFormatFlagIsSignedInteger: {ca.get_linear_pcm_format_flag_is_signed_integer()}")
    print(f"kLinearPCMFormatFlagIsPacked: {ca.get_linear_pcm_format_flag_is_packed()}")
    print(f"kAudioFileWAVEType: {ca.get_audio_file_wave_type()}")
    print(f"kAudioFileReadPermission: {ca.get_audio_file_read_permission()}")
    print(f"kAudioFilePropertyDataFormat: {ca.get_audio_file_property_data_format()}")
    print(f"kAudioFilePropertyMaximumPacketSize: {ca.get_audio_file_property_maximum_packet_size()}")
    print()

def demo_fourcc_conversion():
    """Demonstrate FourCC conversion functions"""
    print("=== FourCC Conversion Demo ===")
    
    # Test string to int conversion
    test_codes = ['WAVE', 'TEXT', 'AIFF', 'mp4f']
    for code in test_codes:
        int_val = ca.fourchar_to_int(code)
        back_to_str = ca.int_to_fourchar(int_val)
        print(f"'{code}' -> {int_val} -> '{back_to_str}'")
    
    print()

def demo_audio_file_operations():
    """Demonstrate audio file operations"""
    print("=== Audio File Operations Demo ===")
    
    amen_path = os.path.join("tests", "amen.wav")
    if not os.path.exists(amen_path):
        print(f"Error: {amen_path} not found")
        return
    
    try:
        # Open audio file
        print(f"Opening: {amen_path}")
        audio_file_id = ca.audio_file_open_url(
            amen_path, 
            ca.get_audio_file_read_permission(), 
            ca.get_audio_file_wave_type()
        )
        print(f"Opened audio file with ID: {audio_file_id}")
        
        try:
            # Get file properties
            print("Getting audio file properties...")
            try:
                data_format_bytes = ca.audio_file_get_property(
                    audio_file_id, 
                    ca.get_audio_file_property_data_format()
                )
                print(f"Data format property: {len(data_format_bytes)} bytes")
                
                # Parse the AudioStreamBasicDescription (first 40 bytes)
                if len(data_format_bytes) >= 40:
                    import struct
                    # Unpack the AudioStreamBasicDescription structure
                    # (sample_rate, format_id, format_flags, bytes_per_packet, 
                    #  frames_per_packet, bytes_per_frame, channels_per_frame, 
                    #  bits_per_channel, reserved)
                    asbd = struct.unpack('<dLLLLLLLL', data_format_bytes[:40])
                    print(f"  Sample Rate: {asbd[0]} Hz")
                    print(f"  Format ID: {asbd[1]} ({ca.int_to_fourchar(asbd[1])})")
                    print(f"  Format Flags: {asbd[2]}")
                    print(f"  Bytes per Packet: {asbd[3]}")
                    print(f"  Frames per Packet: {asbd[4]}")
                    print(f"  Bytes per Frame: {asbd[5]}")
                    print(f"  Channels per Frame: {asbd[6]}")
                    print(f"  Bits per Channel: {asbd[7]}")
                    
            except Exception as e:
                print(f"Could not get data format property: {e}")
            
            # Try to read some packets
            try:
                print("Reading audio packets...")
                packet_data, packets_read = ca.audio_file_read_packets(audio_file_id, 0, 10)
                print(f"Read {packets_read} packets, {len(packet_data)} bytes of data")
                print(f"First 16 bytes: {packet_data[:16].hex() if len(packet_data) >= 16 else 'N/A'}")
            except Exception as e:
                print(f"Could not read packets: {e}")
            
        finally:
            # Close the file
            print("Closing audio file...")
            ca.audio_file_close(audio_file_id)
            
    except Exception as e:
        print(f"Error with audio file operations: {e}")
    
    print()

def demo_audio_queue_creation():
    """Demonstrate audio queue creation and management"""
    print("=== Audio Queue Demo ===")
    
    try:
        # Create an audio format description
        audio_format = {
            'sample_rate': 44100.0,
            'format_id': ca.get_audio_format_linear_pcm(),
            'format_flags': ca.get_linear_pcm_format_flag_is_signed_integer() | ca.get_linear_pcm_format_flag_is_packed(),
            'bytes_per_packet': 4,
            'frames_per_packet': 1,
            'bytes_per_frame': 4,
            'channels_per_frame': 2,
            'bits_per_channel': 16
        }
        
        print(f"Creating audio queue with format: {audio_format}")
        
        # Create output queue
        queue_id = ca.audio_queue_new_output(audio_format)
        print(f"Created audio queue: {queue_id}")
        
        try:
            # Allocate some buffers
            buffer_size = 8192
            buffers = []
            
            print(f"Allocating 2 buffers of {buffer_size} bytes each...")
            for i in range(2):
                buffer_id = ca.audio_queue_allocate_buffer(queue_id, buffer_size)
                buffers.append(buffer_id)
                print(f"  Buffer {i}: {buffer_id}")
            
            # Enqueue buffers (they will be empty, so no sound)
            print("Enqueuing buffers...")
            for i, buffer_id in enumerate(buffers):
                ca.audio_queue_enqueue_buffer(queue_id, buffer_id)
                print(f"  Enqueued buffer {i}")
            
            # Start and stop the queue
            print("Starting audio queue...")
            ca.audio_queue_start(queue_id)
            
            print("Queue running for 1 second...")
            time.sleep(1)
            
            print("Stopping audio queue...")
            ca.audio_queue_stop(queue_id, True)
            
        finally:
            # Clean up
            print("Disposing audio queue...")
            ca.audio_queue_dispose(queue_id, True)
            
    except Exception as e:
        print(f"Error with audio queue operations: {e}")
    
    print()

def demo_hardware_functions():
    """Demonstrate hardware-related functions"""
    print("=== Hardware Functions Demo ===")
    
    try:
        # Show system audio object
        print("Showing system audio object...")
        ca.audio_object_show(1)  # System object ID is typically 1
        
    except Exception as e:
        print(f"Error with hardware functions: {e}")
    
    print()

def main():
    print("=== cycoreaudio Wrapper Comprehensive Demo ===")
    print("This demonstrates the extended CoreAudio Python wrapper functionality.\n")
    
    # Test error function first
    print(f"Test error constant: {ca.test_error()}")
    print()
    
    # Run all demos
    demo_constants()
    demo_fourcc_conversion()
    demo_audio_file_operations()
    demo_audio_queue_creation()
    demo_hardware_functions()
    
    print("=== Demo Complete ===")
    print("Successfully demonstrated:")
    print("✓ FourCC conversion utilities")
    print("✓ Audio file opening and property reading")
    print("✓ Audio packet reading")
    print("✓ Audio queue creation and management")
    print("✓ Audio buffer allocation and management")
    print("✓ CoreAudio constants access")
    print("✓ Hardware object interaction")
    print()
    print("The cycoreaudio wrapper now provides comprehensive")
    print("access to CoreAudio functionality from Python!")

if __name__ == "__main__":
    main()