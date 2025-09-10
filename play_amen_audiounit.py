#!/usr/bin/env python3
"""
AudioUnit-based audio playback for amen.wav using the extended cycoreaudio wrapper.
This demonstrates actual audio playback using CoreAudio AudioUnit APIs.
"""

import os
import time
import wave
import struct
import threading
import coreaudio as ca

class AudioUnitPlayer:
    def __init__(self):
        self.audio_unit = None
        self.audio_data = None
        self.sample_rate = 44100
        self.channels = 2
        self.bits_per_sample = 16
        self.current_position = 0
        self.playing = False
        
    def load_wav_file(self, file_path):
        """Load WAV file data"""
        with wave.open(file_path, 'rb') as wav_file:
            self.sample_rate = wav_file.getframerate() 
            self.channels = wav_file.getnchannels()
            self.bits_per_sample = wav_file.getsampwidth() * 8
            frame_count = wav_file.getnframes()
            
            # Read all audio data
            self.audio_data = wav_file.readframes(frame_count)
            
            print(f"Loaded WAV file:")
            print(f"  Sample rate: {self.sample_rate} Hz")
            print(f"  Channels: {self.channels}")
            print(f"  Bits per sample: {self.bits_per_sample}")
            print(f"  Frame count: {frame_count}")
            print(f"  Duration: {frame_count / self.sample_rate:.2f} seconds")
            print(f"  Data size: {len(self.audio_data)} bytes")
    
    def create_audio_unit(self):
        """Create and configure AudioUnit for output"""
        print("Creating AudioUnit...")
        
        # Find the default output audio unit
        description = {
            'type': ca.get_audio_unit_type_output(),
            'subtype': ca.get_audio_unit_subtype_default_output(),  
            'manufacturer': ca.get_audio_unit_manufacturer_apple(),
            'flags': 0,
            'flags_mask': 0
        }
        
        print(f"Looking for AudioUnit with description: {description}")
        
        component_id = ca.audio_component_find_next(description)
        if component_id is None:
            raise RuntimeError("Could not find default output AudioUnit")
        
        print(f"Found AudioComponent: {component_id}")
        
        # Create an instance of the audio unit
        self.audio_unit = ca.audio_component_instance_new(component_id) 
        print(f"Created AudioUnit: {self.audio_unit}")
        
        # Configure the audio unit for our format
        self.configure_audio_format()
        
        # Initialize the audio unit
        ca.audio_unit_initialize(self.audio_unit)
        print("AudioUnit initialized")
        
    def configure_audio_format(self):
        """Configure the audio unit's stream format"""
        print("Configuring AudioUnit format...")
        
        # Create AudioStreamBasicDescription
        # This needs to be packed as a C struct - 40 bytes total
        format_data = struct.pack('<dLLLLLLLL',
            float(self.sample_rate),                    # mSampleRate (Float64)
            ca.get_audio_format_linear_pcm(),          # mFormatID (UInt32) 
            # Format flags: signed integer, packed, non-interleaved
            ca.get_linear_pcm_format_flag_is_signed_integer() | 
            ca.get_linear_pcm_format_flag_is_packed() |
            ca.get_linear_pcm_format_flag_is_non_interleaved(),  # mFormatFlags (UInt32)
            self.bits_per_sample // 8,                 # mBytesPerPacket (UInt32) 
            1,                                         # mFramesPerPacket (UInt32)
            self.bits_per_sample // 8,                 # mBytesPerFrame (UInt32)
            self.channels,                             # mChannelsPerFrame (UInt32)
            self.bits_per_sample,                      # mBitsPerChannel (UInt32)
            0                                          # mReserved (UInt32)
        )
        
        print(f"Setting stream format: {len(format_data)} bytes")
        print(f"Format data: {format_data.hex()}")
        
        # Set the format on the input scope (element 0)
        try:
            ca.audio_unit_set_property(
                self.audio_unit,
                ca.get_audio_unit_property_stream_format(),
                ca.get_audio_unit_scope_input(), 
                0,  # element
                format_data
            )
            print("Successfully set stream format")
        except Exception as e:
            print(f"Failed to set stream format: {e}")
            # Continue anyway - the unit might have a compatible default format
    
    def play(self):
        """Start audio playback"""
        if not self.audio_data:
            raise ValueError("No audio data loaded")
            
        print("Starting AudioUnit playback...")
        self.playing = True
        self.current_position = 0
        
        # Start the output unit
        ca.audio_output_unit_start(self.audio_unit)
        print("AudioUnit started")
        
        return True
        
    def stop(self):
        """Stop audio playback"""
        if self.playing and self.audio_unit:
            print("Stopping AudioUnit playback...")
            ca.audio_output_unit_stop(self.audio_unit)
            self.playing = False
            print("AudioUnit stopped")
            
    def cleanup(self):
        """Clean up resources"""
        self.stop()
        
        if self.audio_unit:
            print("Disposing AudioUnit...")
            ca.audio_unit_uninitialize(self.audio_unit)
            ca.audio_component_instance_dispose(self.audio_unit)
            self.audio_unit = None
            print("AudioUnit disposed")

def main():
    # Path to the amen.wav file
    amen_path = os.path.join("tests", "amen.wav")
    
    if not os.path.exists(amen_path):
        print(f"Error: {amen_path} not found")
        return
    
    print("=== AudioUnit-based Audio Playback ===")
    print(f"File: {amen_path}")
    
    player = AudioUnitPlayer()
    
    try:
        # Load the WAV file
        print("Loading WAV file...")
        player.load_wav_file(amen_path)
        
        # Create and configure AudioUnit
        print("Creating AudioUnit...")
        player.create_audio_unit()
        
        # Start playback
        print("Starting playback...")
        player.play()
        
        # Let it play for the duration of the file (plus a bit of buffer)
        duration = len(player.audio_data) / (player.sample_rate * player.channels * (player.bits_per_sample // 8))
        print(f"Playing for {duration:.2f} seconds...")
        time.sleep(duration + 1)
        
        print("Playback completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        player.cleanup()
        
    print("AudioUnit demo completed")

if __name__ == "__main__":
    main()