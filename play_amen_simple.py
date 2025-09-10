#!/usr/bin/env python3
"""
Simple audio playback using pure Python with wave module and simple audio output.
This demonstrates that the basic wrapper works.
"""

import os
import time
import wave
import array
import threading
import coreaudio as ca

class AudioPlayer:
    def __init__(self):
        self.audio_queue = None
        self.audio_file = None
        self.buffers = []
        self.playing = False
        self.current_frame = 0
        self.audio_data = None
        
    def load_wav_file(self, file_path):
        """Load WAV file data"""
        with wave.open(file_path, 'rb') as wav_file:
            # Get audio properties
            self.sample_rate = wav_file.getframerate()
            self.channels = wav_file.getnchannels()
            self.sample_width = wav_file.getsampwidth()
            self.frame_count = wav_file.getnframes()
            
            # Read all audio data
            self.audio_data = wav_file.readframes(self.frame_count)
            
            print(f"Loaded WAV file:")
            print(f"  Sample rate: {self.sample_rate} Hz")
            print(f"  Channels: {self.channels}")
            print(f"  Sample width: {self.sample_width} bytes")
            print(f"  Frame count: {self.frame_count}")
            print(f"  Duration: {self.frame_count / self.sample_rate:.2f} seconds")
            print(f"  Data size: {len(self.audio_data)} bytes")
            
    def create_audio_queue(self):
        """Create CoreAudio queue for playback"""
        audio_format = {
            'sample_rate': float(self.sample_rate),
            'format_id': ca.get_audio_format_linear_pcm(),
            'format_flags': ca.get_linear_pcm_format_flag_is_signed_integer() | ca.get_linear_pcm_format_flag_is_packed(),
            'bytes_per_packet': self.sample_width * self.channels,
            'frames_per_packet': 1,
            'bytes_per_frame': self.sample_width * self.channels,
            'channels_per_frame': self.channels,
            'bits_per_channel': self.sample_width * 8
        }
        
        print(f"Creating audio queue with format: {audio_format}")
        self.audio_queue = ca.audio_queue_new_output(audio_format)
        print(f"Created audio queue: {self.audio_queue}")
        
    def setup_buffers(self):
        """Allocate audio buffers"""
        buffer_size = 16384  # 16KB per buffer
        num_buffers = 3
        
        print(f"Allocating {num_buffers} buffers of {buffer_size} bytes each")
        for i in range(num_buffers):
            buffer_id = ca.audio_queue_allocate_buffer(self.audio_queue, buffer_size)
            self.buffers.append(buffer_id)
            print(f"Allocated buffer {i}: {buffer_id}")
    
    def play(self):
        """Start audio playback"""
        if not self.audio_data:
            raise ValueError("No audio data loaded")
            
        print("Starting audio playback...")
        self.playing = True
        self.current_frame = 0
        
        # Start the audio queue
        ca.audio_queue_start(self.audio_queue)
        
        # The current implementation doesn't actually fill buffers with audio data
        # This is a limitation of our simple callback - in a real implementation,
        # we would need to fill the buffers with audio data from our loaded file
        
        print("Audio queue started (note: actual audio data playback not yet implemented)")
        return True
        
    def stop(self):
        """Stop audio playback"""
        if self.playing:
            print("Stopping audio playback...")
            ca.audio_queue_stop(self.audio_queue, True)
            self.playing = False
            
    def cleanup(self):
        """Clean up resources"""
        if self.audio_queue:
            print("Disposing audio queue...")
            ca.audio_queue_dispose(self.audio_queue, True)
            self.audio_queue = None

def main():
    # Path to the amen.wav file
    amen_path = os.path.join("tests", "amen.wav")
    
    if not os.path.exists(amen_path):
        print(f"Error: {amen_path} not found")
        return
    
    print("=== Enhanced CoreAudio Wrapper Test ===")
    print(f"File: {amen_path}")
    
    player = AudioPlayer()
    
    try:
        # Load the WAV file
        player.load_wav_file(amen_path)
        
        # Create audio queue
        player.create_audio_queue()
        
        # Set up buffers
        player.setup_buffers()
        
        # Start playback
        player.play()
        
        # Let it "play" for a few seconds
        print("Running for 3 seconds...")
        time.sleep(3)
        
        # Stop playback
        player.stop()
        
        print("Test completed successfully!")
        print("Note: This demonstrates the CoreAudio wrapper infrastructure.")
        print("Actual audio output would require implementing buffer filling in the callback.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        player.cleanup()

if __name__ == "__main__":
    main()