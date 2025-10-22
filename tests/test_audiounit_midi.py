"""Tests for AudioUnit MIDI functionality

Tests the MIDI support for instrument AudioUnit plugins.
"""

import pytest
import coremusic as cm
import time


class TestAudioUnitMIDI:
    """Test MIDI functionality with AudioUnit instrument plugins"""

    @pytest.fixture
    def instrument_plugin(self):
        """Create and initialize an instrument plugin for testing"""
        # Use DLSMusicDevice (Apple's built-in General MIDI synthesizer)
        # This is guaranteed to exist on all macOS systems
        plugin = cm.AudioUnitPlugin.from_name("DLSMusicDevice", component_type='aumu')
        plugin.instantiate()
        plugin.initialize()
        yield plugin
        plugin.dispose()

    def test_discover_instruments(self):
        """Test discovering instrument plugins"""
        host = cm.AudioUnitHost()
        instruments = host.discover_plugins(type='instrument')

        print(f"\nFound {len(instruments)} instrument plugins")
        assert len(instruments) > 0
        assert isinstance(instruments, list)

        # Should have at least DLSMusicDevice
        dls_found = any('DLS' in inst['name'] for inst in instruments)
        assert dls_found, "DLSMusicDevice not found"

    def test_load_instrument_plugin(self):
        """Test loading an instrument plugin"""
        host = cm.AudioUnitHost()

        with host.load_plugin("DLSMusicDevice", type='instrument') as synth:
            assert synth is not None
            assert synth.type == 'aumu'
            assert synth.is_initialized
            print(f"\nLoaded: {synth.name}")

    def test_note_on_note_off(self, instrument_plugin):
        """Test basic MIDI note on/off"""
        synth = instrument_plugin

        # Should not raise exception
        synth.note_on(channel=0, note=60, velocity=100)
        time.sleep(0.1)
        synth.note_off(channel=0, note=60)

        print("\n✓ Note on/off successful")

    def test_multiple_notes(self, instrument_plugin):
        """Test playing multiple notes simultaneously"""
        synth = instrument_plugin

        # Play a C major chord (C, E, G)
        notes = [60, 64, 67]
        for note in notes:
            synth.note_on(channel=0, note=note, velocity=100)

        time.sleep(0.1)

        # Release all notes
        for note in notes:
            synth.note_off(channel=0, note=note)

        print("\n✓ Multiple notes successful")

    def test_velocity_range(self, instrument_plugin):
        """Test different velocity values"""
        synth = instrument_plugin

        # Test various velocities
        velocities = [1, 32, 64, 96, 127]
        for vel in velocities:
            synth.note_on(channel=0, note=60, velocity=vel)
            time.sleep(0.05)
            synth.note_off(channel=0, note=60)

        print(f"\n✓ Tested {len(velocities)} velocity values")

    def test_note_range(self, instrument_plugin):
        """Test notes across MIDI range"""
        synth = instrument_plugin

        # Test low, middle, and high notes
        test_notes = [21, 60, 108]  # A0, C4, C8
        for note in test_notes:
            synth.note_on(channel=0, note=note, velocity=80)
            time.sleep(0.05)
            synth.note_off(channel=0, note=note)

        print(f"\n✓ Tested {len(test_notes)} note positions")

    def test_control_change(self, instrument_plugin):
        """Test MIDI control change messages"""
        synth = instrument_plugin

        # Test common controllers
        synth.control_change(channel=0, controller=7, value=100)   # Volume
        synth.control_change(channel=0, controller=10, value=64)   # Pan center
        synth.control_change(channel=0, controller=11, value=127)  # Expression

        print("\n✓ Control change messages successful")

    def test_program_change(self, instrument_plugin):
        """Test MIDI program change"""
        synth = instrument_plugin

        # Change to different General MIDI instruments
        programs = [0, 24, 40]  # Acoustic Grand, Nylon Guitar, Violin
        for program in programs:
            synth.program_change(channel=0, program=program)
            time.sleep(0.05)

        print(f"\n✓ Changed to {len(programs)} different programs")

    def test_pitch_bend(self, instrument_plugin):
        """Test MIDI pitch bend"""
        synth = instrument_plugin

        # Play a note
        synth.note_on(channel=0, note=60, velocity=100)

        # Bend pitch
        synth.pitch_bend(channel=0, value=8192)   # Center (no bend)
        time.sleep(0.05)
        synth.pitch_bend(channel=0, value=12288)  # Bend up
        time.sleep(0.05)
        synth.pitch_bend(channel=0, value=8192)   # Back to center

        synth.note_off(channel=0, note=60)

        print("\n✓ Pitch bend successful")

    def test_all_notes_off(self, instrument_plugin):
        """Test all notes off command"""
        synth = instrument_plugin

        # Play multiple notes
        for note in [60, 64, 67]:
            synth.note_on(channel=0, note=note, velocity=100)

        # Turn all off at once
        synth.all_notes_off(channel=0)

        print("\n✓ All notes off successful")

    def test_midi_channels(self, instrument_plugin):
        """Test MIDI on different channels"""
        synth = instrument_plugin

        # Test channels 0-15
        for channel in range(16):
            synth.note_on(channel=channel, note=60, velocity=80)
            time.sleep(0.02)
            synth.note_off(channel=channel, note=60)

        print("\n✓ Tested all 16 MIDI channels")

    def test_send_midi_raw(self, instrument_plugin):
        """Test sending raw MIDI messages"""
        synth = instrument_plugin

        # Send raw MIDI note on: status=0x90, note=60, velocity=100
        synth.send_midi(status=0x90, data1=60, data2=100)
        time.sleep(0.1)
        # Send raw MIDI note off: status=0x80, note=60, velocity=0
        synth.send_midi(status=0x80, data1=60, data2=0)

        print("\n✓ Raw MIDI messages successful")

    def test_midi_on_effect_plugin_fails(self):
        """Test that MIDI methods raise error on effect plugins"""
        host = cm.AudioUnitHost()

        with host.load_plugin("Bandpass", type='effect') as effect:
            assert effect.type == 'aufx'

            # Should raise ValueError
            with pytest.raises(ValueError, match="MIDI only supported for instrument"):
                effect.note_on(channel=0, note=60, velocity=100)

            print("\n✓ Correctly rejected MIDI on effect plugin")

    def test_midi_offset_frames(self, instrument_plugin):
        """Test MIDI with sample offset scheduling"""
        synth = instrument_plugin

        # Schedule notes at different sample offsets
        synth.note_on(channel=0, note=60, velocity=100, offset_frames=0)
        synth.note_on(channel=0, note=64, velocity=100, offset_frames=441)   # ~10ms at 44.1kHz
        synth.note_on(channel=0, note=67, velocity=100, offset_frames=882)   # ~20ms at 44.1kHz

        time.sleep(0.1)

        synth.note_off(channel=0, note=60, offset_frames=0)
        synth.note_off(channel=0, note=64, offset_frames=0)
        synth.note_off(channel=0, note=67, offset_frames=0)

        print("\n✓ Sample-accurate MIDI scheduling successful")

    def test_rapid_note_sequence(self, instrument_plugin):
        """Test rapid note sequence (like arpeggiator)"""
        synth = instrument_plugin

        notes = [60, 64, 67, 72]  # C major arpeggio
        for _ in range(3):  # Repeat 3 times
            for note in notes:
                synth.note_on(channel=0, note=note, velocity=100)
                time.sleep(0.05)
                synth.note_off(channel=0, note=note)

        print("\n✓ Rapid note sequence successful")


class TestAudioUnitMIDIIntegration:
    """Test integrated MIDI workflows"""

    def test_complete_performance(self):
        """Test a complete musical performance"""
        host = cm.AudioUnitHost()

        with host.load_plugin("DLSMusicDevice", type='instrument') as synth:
            print(f"\n♪ Playing on: {synth.name}")

            # Set to Acoustic Grand Piano
            synth.program_change(channel=0, program=0)

            # Set volume
            synth.control_change(channel=0, controller=7, value=100)

            # Play a simple melody: C D E C (in quarter notes)
            melody = [60, 62, 64, 60]
            duration = 0.3

            for note in melody:
                synth.note_on(channel=0, note=note, velocity=90)
                time.sleep(duration)
                synth.note_off(channel=0, note=note)
                time.sleep(0.05)  # Small gap between notes

            print("✓ Complete performance successful")

    def test_multi_channel_performance(self):
        """Test using multiple MIDI channels"""
        host = cm.AudioUnitHost()

        with host.load_plugin("DLSMusicDevice", type='instrument') as synth:
            # Channel 0: Piano
            synth.program_change(channel=0, program=0)

            # Channel 1: Strings
            synth.program_change(channel=1, program=48)

            # Play piano
            synth.note_on(channel=0, note=60, velocity=90)

            # Play strings
            synth.note_on(channel=1, note=64, velocity=70)

            time.sleep(0.5)

            # Release both
            synth.note_off(channel=0, note=60)
            synth.note_off(channel=1, note=64)

            print("\n✓ Multi-channel performance successful")


class TestAudioUnitMIDIErrors:
    """Test error handling for MIDI operations"""

    def test_midi_before_initialize(self):
        """Test MIDI on uninitialized plugin"""
        plugin = cm.AudioUnitPlugin.from_name("DLSMusicDevice", component_type='aumu')
        plugin.instantiate()
        # Don't initialize

        with pytest.raises(RuntimeError, match="Plugin not initialized"):
            plugin.note_on(channel=0, note=60, velocity=100)

        plugin.dispose()

    def test_invalid_channel(self):
        """Test that invalid channels are clamped"""
        plugin = cm.AudioUnitPlugin.from_name("DLSMusicDevice", component_type='aumu')
        plugin.instantiate()
        plugin.initialize()

        # Channel > 15 should work (gets masked to 0-15 in C code)
        # This tests the robustness, not that it raises an error
        plugin.note_on(channel=16, note=60, velocity=100)  # Should clamp to channel 0
        time.sleep(0.05)
        plugin.note_off(channel=16, note=60)

        plugin.dispose()
        print("\n✓ Out-of-range channels handled gracefully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
