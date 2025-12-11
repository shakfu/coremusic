#!/usr/bin/env python3

import os
import time
import sys

from conftest import AMEN_WAV_PATH
import coremusic as cm
import coremusic.capi as capi


def test_audio_playback():
    """Test audio playback using the AudioPlayer class"""
    print("=== AudioPlayer Test ===")

    # Check if test file exists
    test_file = AMEN_WAV_PATH
    assert os.path.exists(test_file)
    print(f"Found test file: {test_file}")

    try:
        # Create AudioPlayer instance
        player = cm.AudioPlayer()
        print("Created AudioPlayer instance")

        # Load the audio file
        print(f"Loading audio file: {test_file}")
        result = player.load_file(test_file)
        print(f"Loaded audio file (result: {result})")

        # Setup audio output
        print("Setting up audio output...")
        result = player.setup_output()
        print(f"Setup audio output (result: {result})")

        # Enable looping for demonstration
        player.set_looping(True)
        print("Enabled looping")

        # Start playback
        print("Starting audio playback...")
        result = player.start()
        print(f"Started playback (result: {result})")

        # Monitor playback briefly (reduced from 10s to 2s to avoid MIDI service interference)
        print("Playback status:")
        for i in range(2):
            is_playing = player.is_playing()
            progress = player.get_progress()
            print(f"  {i + 1:2d}s: Playing={is_playing}, Progress={progress:.3f}")
            time.sleep(1.0)

        # Stop playback
        print("Stopping playback...")
        result = player.stop()
        print(f"Stopped playback (result: {result})")

        # Test reset functionality
        player.reset_playback()
        print("Reset playback to beginning")

        print("AudioPlayer test completed successfully!")
        assert True

    except Exception as e:
        print(f" AudioPlayer test failed: {e}")
        assert False


def test_audio_player_play_method():
    """Test that play() method works as an alias for start()"""
    print("=== AudioPlayer play() Method Test ===")

    test_file = AMEN_WAV_PATH
    assert os.path.exists(test_file)

    try:
        # Create AudioPlayer instance
        player = cm.AudioPlayer()
        print("Created AudioPlayer instance")

        # Load and setup
        player.load_file(test_file)
        player.setup_output()
        print("Loaded file and setup output")

        # Test play() method
        print("Testing play() method...")
        result = player.play()
        print(f"Called play() (result: {result})")

        # Verify playback started
        time.sleep(0.5)
        assert player.is_playing(), "Player should be playing after play() call"
        print("✓ play() method works correctly")

        # Stop playback
        player.stop()
        print("Stopped playback")

        # Reset and test start() method for comparison
        player.reset_playback()
        print("Testing start() method...")
        result = player.start()
        print(f"Called start() (result: {result})")

        # Verify playback started
        time.sleep(0.5)
        assert player.is_playing(), "Player should be playing after start() call"
        print("✓ start() method works correctly")

        # Stop playback
        player.stop()

        print("✓ Both play() and start() methods work correctly!")
        assert True

    except Exception as e:
        print(f"✗ play() method test failed: {e}")
        assert False


def test_module_test_error():
    error_code = capi.test_error()
    assert error_code
    print(f"Basic module test passed (error code: {error_code})")
