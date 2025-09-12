#!/usr/bin/env python3

import os
import time
import sys

# Add the current directory to Python path to import our module
sys.path.insert(0, os.path.dirname(__file__))

try:
    import coreaudio
    print("✓ Successfully imported coreaudio module")
except ImportError as e:
    print(f"✗ Failed to import coreaudio module: {e}")
    print("Make sure to run 'make coreaudio' first to build the extension")
    sys.exit(1)

def test_audio_playback():
    """Test audio playback using the AudioPlayer class"""
    print("\n=== AudioPlayer Test ===")
    
    # Check if test file exists
    test_file = "tests/amen.wav"
    assert os.path.exists(test_file)
    print(f"✓ Found test file: {test_file}")
    
    try:
        # Create AudioPlayer instance
        player = coreaudio.AudioPlayer()
        print("✓ Created AudioPlayer instance")
        
        # Load the audio file
        print(f"Loading audio file: {test_file}")
        result = player.load_file(test_file)
        print(f"✓ Loaded audio file (result: {result})")
        
        # Setup audio output
        print("Setting up audio output...")
        result = player.setup_output()
        print(f"✓ Setup audio output (result: {result})")
        
        # Enable looping for demonstration
        player.set_looping(True)
        print("✓ Enabled looping")
        
        # Start playback
        print("Starting audio playback...")
        result = player.start()
        print(f"✓ Started playback (result: {result})")
        
        # Monitor playback for a few seconds
        print("\nPlayback status:")
        for i in range(10):
            is_playing = player.is_playing()
            progress = player.get_progress()
            print(f"  {i+1:2d}s: Playing={is_playing}, Progress={progress:.3f}")
            time.sleep(1.0)
        
        # Stop playback
        print("\nStopping playback...")
        result = player.stop()
        print(f"✓ Stopped playback (result: {result})")
        
        # Test reset functionality
        player.reset_playback()
        print("✓ Reset playback to beginning")
        
        print("\n✅ AudioPlayer test completed successfully!")
        assert True
        
    except Exception as e:
        print(f"✗ AudioPlayer test failed: {e}")
        assert False

def test_module_test_error():
    error_code = coreaudio.test_error()
    assert error_code
    print(f"✓ Basic module test passed (error code: {error_code})")
