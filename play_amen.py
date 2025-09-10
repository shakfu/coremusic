#!/usr/bin/env python3
"""
Audio playback script using the extended cycoreaudio wrapper to play amen.wav
"""

import os
import time
import struct
import coreaudio as ca

def get_wav_format(file_path):
    """Get WAV file format information by reading the header"""
    with open(file_path, 'rb') as f:
        # Read WAV header
        header = f.read(44)
        
        # Parse basic WAV format
        if header[:4] != b'RIFF' or header[8:12] != b'WAVE':
            raise ValueError("Not a valid WAV file")
        
        # Extract format information (assuming standard PCM WAV)
        channels = struct.unpack('<H', header[22:24])[0]
        sample_rate = struct.unpack('<L', header[24:28])[0]
        bits_per_sample = struct.unpack('<H', header[34:36])[0]
        
        return {
            'sample_rate': float(sample_rate),
            'format_id': ca.get_audio_format_linear_pcm(),
            'format_flags': ca.get_linear_pcm_format_flag_is_signed_integer() | ca.get_linear_pcm_format_flag_is_packed(),
            'bytes_per_packet': (bits_per_sample // 8) * channels,
            'frames_per_packet': 1,
            'bytes_per_frame': (bits_per_sample // 8) * channels,
            'channels_per_frame': channels,
            'bits_per_channel': bits_per_sample
        }

def play_audio_file(file_path):
    """Play an audio file using CoreAudio AudioQueue"""
    
    # Get file info first
    print(f"Opening audio file: {file_path}")
    
    try:
        # Get WAV format from file header
        audio_format = get_wav_format(file_path)
        print(f"Audio format: {audio_format}")
        
        # Open the audio file
        audio_file_id = ca.audio_file_open_url(file_path, ca.get_audio_file_read_permission(), ca.get_audio_file_wave_type())
        print(f"Opened audio file with ID: {audio_file_id}")
        
        try:
            # Get file properties
            try:
                data_format_bytes = ca.audio_file_get_property(audio_file_id, ca.get_audio_file_property_data_format())
                print(f"Got data format property: {len(data_format_bytes)} bytes")
            except Exception as e:
                print(f"Could not get data format: {e}")
                # Fall back to using WAV header info
            
            # Create audio queue
            print("Creating audio output queue...")
            queue_id = ca.audio_queue_new_output(audio_format)
            print(f"Created audio queue with ID: {queue_id}")
            
            try:
                # Allocate buffers
                buffer_size = 16384  # 16KB buffers
                print(f"Allocating audio buffers (size: {buffer_size})...")
                
                buffer1_id = ca.audio_queue_allocate_buffer(queue_id, buffer_size)
                buffer2_id = ca.audio_queue_allocate_buffer(queue_id, buffer_size)
                print(f"Allocated buffers: {buffer1_id}, {buffer2_id}")
                
                # Start playing
                print("Starting audio playback...")
                ca.audio_queue_start(queue_id)
                
                # Play for a few seconds
                print("Playing for 5 seconds...")
                time.sleep(5)
                
                print("Stopping playback...")
                ca.audio_queue_stop(queue_id, True)
                
            finally:
                print("Disposing of audio queue...")
                ca.audio_queue_dispose(queue_id, True)
                
        finally:
            print("Closing audio file...")
            ca.audio_file_close(audio_file_id)
            
    except Exception as e:
        print(f"Error playing audio: {e}")
        raise

def main():
    # Path to the amen.wav file
    amen_path = os.path.join("tests", "amen.wav")
    
    if not os.path.exists(amen_path):
        print(f"Error: {amen_path} not found")
        return
    
    print("=== CoreAudio Wrapper Audio Playback Test ===")
    print(f"File: {amen_path}")
    print(f"File size: {os.path.getsize(amen_path)} bytes")
    
    try:
        play_audio_file(amen_path)
        print("Playback completed successfully!")
    except Exception as e:
        print(f"Playback failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()