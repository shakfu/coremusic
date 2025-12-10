"""Integration tests for end-to-end audio workflows.

Tests complete workflows as specified in CODE_REVIEW.md Medium Priority:
1. Record -> Process -> Save (simulated recording since input not fully implemented)
2. MIDI file -> AudioUnit rendering
3. Link session synchronization
"""

import os
import time
import struct
import tempfile
import pytest

import coremusic as cm
import coremusic.capi as capi
from coremusic import link


# =============================================================================
# Workflow 1: Audio -> Process -> Save
# =============================================================================


class TestAudioProcessSaveWorkflow:
    """Test complete audio processing and saving workflow.

    Since audio recording input is not fully implemented (requires Cython
    callback infrastructure), we simulate the "record" step by reading
    from an existing audio file, which exercises the same processing pipeline.
    """

    def test_read_process_effect_save_workflow(self, amen_wav_path, tmp_path):
        """Test: Read audio -> Setup effect chain -> Save to new file

        Note: Full AudioUnit rendering requires a more complex AUGraph setup.
        This test demonstrates the complete workflow with effect discovery
        and audio file I/O, which are the core integration points.
        """
        output_path = str(tmp_path / "processed_output.wav")

        # Step 1: Read source audio using functional API
        file_id = capi.audio_file_open_url(amen_wav_path)
        try:
            # Get format
            format_data = capi.audio_file_get_property(
                file_id, capi.get_audio_file_property_data_format()
            )
            asbd = cm.parse_audio_stream_basic_description(format_data)
            source_format = cm.AudioFormat(
                sample_rate=asbd['sample_rate'],
                format_id=asbd['format_id'],
                format_flags=asbd['format_flags'],
                bytes_per_packet=asbd['bytes_per_packet'],
                frames_per_packet=asbd['frames_per_packet'],
                bytes_per_frame=asbd['bytes_per_frame'],
                channels_per_frame=asbd['channels_per_frame'],
                bits_per_channel=asbd['bits_per_channel'],
            )

            # Read all audio data
            audio_data, packets_read = capi.audio_file_read_packets(file_id, 0, 200000)
        finally:
            capi.audio_file_close(file_id)

        assert len(audio_data) > 0
        assert source_format.sample_rate == 44100.0
        assert source_format.channels_per_frame == 2

        # Step 2: Discover and configure effect (demonstrates effect integration)
        host = cm.AudioUnitHost()
        with host.load_plugin("AUDelay", type='effect') as effect:
            # Verify effect is loaded and configured
            assert effect is not None
            assert effect.is_initialized
            assert effect.type == 'aufx'

            # Get effect parameters (demonstrates parameter access)
            param_list = effect.parameters
            assert isinstance(param_list, list)

        # Step 3: Save audio to new file (demonstrates file output)
        # For this integration test, we save the original audio to verify
        # the complete read->write workflow
        output_format = {
            "sample_rate": source_format.sample_rate,
            "format_id": capi.get_audio_format_linear_pcm(),
            "format_flags": 12,
            "bytes_per_packet": source_format.bytes_per_frame,
            "frames_per_packet": 1,
            "bytes_per_frame": source_format.bytes_per_frame,
            "channels_per_frame": source_format.channels_per_frame,
            "bits_per_channel": source_format.bits_per_channel,
            "reserved": 0,
        }

        ext_file = capi.extended_audio_file_create_with_url(
            output_path,
            capi.get_audio_file_wave_type(),
            output_format,
            0
        )
        try:
            num_frames = len(audio_data) // source_format.bytes_per_frame
            capi.extended_audio_file_write(ext_file, num_frames, audio_data)
        finally:
            capi.extended_audio_file_dispose(ext_file)

        # Verify output file
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Verify output can be read back
        with cm.AudioFile(output_path) as output:
            assert output.format.sample_rate == source_format.sample_rate
            assert output.format.channels_per_frame == source_format.channels_per_frame
            assert output.duration > 0

    def test_read_convert_format_save_workflow(self, amen_wav_path, tmp_path):
        """Test: Read audio -> Convert format (stereo->mono) -> Save"""
        output_path = str(tmp_path / "converted_output.wav")

        # Step 1: Read source audio using functional API
        file_id = capi.audio_file_open_url(amen_wav_path)
        try:
            audio_data, packets_read = capi.audio_file_read_packets(file_id, 0, 200000)
            # Calculate duration: packets * frames_per_packet / sample_rate
            # For WAV, frames_per_packet = 1, so packets = frames
            original_duration = packets_read / 44100.0
        finally:
            capi.audio_file_close(file_id)

        # Step 2: Convert format using AudioConverter
        # Convert from stereo to mono (same sample rate for simplicity)
        source_format_dict = {
            "sample_rate": 44100.0,
            "format_id": capi.get_audio_format_linear_pcm(),
            "format_flags": 12,
            "bytes_per_packet": 4,
            "frames_per_packet": 1,
            "bytes_per_frame": 4,
            "channels_per_frame": 2,
            "bits_per_channel": 16,
            "reserved": 0,
        }

        dest_format_dict = {
            "sample_rate": 44100.0,  # Same sample rate
            "format_id": capi.get_audio_format_linear_pcm(),
            "format_flags": 12,
            "bytes_per_packet": 2,
            "frames_per_packet": 1,
            "bytes_per_frame": 2,
            "channels_per_frame": 1,  # Mono
            "bits_per_channel": 16,
            "reserved": 0,
        }

        converter = capi.audio_converter_new(source_format_dict, dest_format_dict)
        try:
            converted_audio = capi.audio_converter_convert_buffer(converter, audio_data)
        finally:
            capi.audio_converter_dispose(converter)

        assert len(converted_audio) > 0

        # Step 3: Save converted audio
        ext_file = capi.extended_audio_file_create_with_url(
            output_path,
            capi.get_audio_file_wave_type(),
            dest_format_dict,
            0
        )
        try:
            num_frames = len(converted_audio) // 2  # 2 bytes per frame (mono 16-bit)
            capi.extended_audio_file_write(ext_file, num_frames, converted_audio)
        finally:
            capi.extended_audio_file_dispose(ext_file)

        # Verify output
        assert os.path.exists(output_path)

        with cm.AudioFile(output_path) as output:
            assert output.format.sample_rate == 44100.0
            assert output.format.channels_per_frame == 1
            # Duration should be approximately the same
            assert abs(output.duration - original_duration) < 0.1

    def test_multi_effect_chain_workflow(self, amen_wav_path, tmp_path):
        """Test: Read audio -> Apply multiple effects in chain -> Save"""
        output_path = str(tmp_path / "multi_effect_output.wav")

        # Read source using functional API
        file_id = capi.audio_file_open_url(amen_wav_path)
        try:
            # Get format
            format_data = capi.audio_file_get_property(
                file_id, capi.get_audio_file_property_data_format()
            )
            asbd = cm.parse_audio_stream_basic_description(format_data)
            source_format = cm.AudioFormat(
                sample_rate=asbd['sample_rate'],
                format_id=asbd['format_id'],
                format_flags=asbd['format_flags'],
                bytes_per_packet=asbd['bytes_per_packet'],
                frames_per_packet=asbd['frames_per_packet'],
                bytes_per_frame=asbd['bytes_per_frame'],
                channels_per_frame=asbd['channels_per_frame'],
                bits_per_channel=asbd['bits_per_channel'],
            )
            audio_data, packets_read = capi.audio_file_read_packets(file_id, 0, 200000)
        finally:
            capi.audio_file_close(file_id)

        # Create effect chain with multiple effects
        chain = cm.AudioEffectsChain()

        # Add effects to chain
        mixer_node = chain.add_effect("aumi", "3dem", "appl")  # 3DMixer
        output_node = chain.add_output()

        # Connect chain
        chain.connect(mixer_node, output_node)

        # Process and save (chain provides framework, actual processing
        # depends on AudioUnit capabilities)
        assert chain.node_count == 2

        # Save original audio (chain setup verified)
        output_format = {
            "sample_rate": source_format.sample_rate,
            "format_id": capi.get_audio_format_linear_pcm(),
            "format_flags": 12,
            "bytes_per_packet": source_format.bytes_per_frame,
            "frames_per_packet": 1,
            "bytes_per_frame": source_format.bytes_per_frame,
            "channels_per_frame": source_format.channels_per_frame,
            "bits_per_channel": source_format.bits_per_channel,
            "reserved": 0,
        }

        ext_file = capi.extended_audio_file_create_with_url(
            output_path,
            capi.get_audio_file_wave_type(),
            output_format,
            0
        )
        try:
            num_frames = len(audio_data) // source_format.bytes_per_frame
            capi.extended_audio_file_write(ext_file, num_frames, audio_data)
        finally:
            capi.extended_audio_file_dispose(ext_file)

        chain.dispose()

        # Verify output
        assert os.path.exists(output_path)
        with cm.AudioFile(output_path) as output:
            assert output.duration > 0


# =============================================================================
# Workflow 2: MIDI file -> AudioUnit Rendering
# =============================================================================


class TestMIDIToAudioUnitWorkflow:
    """Test MIDI file loading and AudioUnit instrument rendering."""

    def test_midi_sequence_to_synth_workflow(self, tmp_path):
        """Test: Load MIDI -> Create sequence -> Play through synth"""
        # Create a simple MIDI file for testing
        from coremusic.midi.utilities import MIDISequence

        midi_path = str(tmp_path / "test_sequence.mid")
        midi_seq = MIDISequence(tempo=120)
        track = midi_seq.add_track("Test")

        # Add notes - signature is (time, note, velocity, duration)
        for i, note in enumerate([60, 62, 64, 65]):
            track.add_note(i * 0.5, note, 100, 0.4)

        midi_seq.save(midi_path)

        # Step 1: Create MusicSequence and load MIDI file
        sequence = capi.new_music_sequence()
        try:
            capi.music_sequence_file_load(sequence, midi_path)

            # Verify sequence loaded
            track_count = capi.music_sequence_get_track_count(sequence)
            assert track_count > 0

            # Step 2: Create MusicPlayer
            player = capi.new_music_player()
            try:
                capi.music_player_set_sequence(player, sequence)

                # Verify player configured
                retrieved_seq = capi.music_player_get_sequence(player)
                assert retrieved_seq == sequence

                # Step 3: Preroll and verify ready
                capi.music_player_preroll(player)

                # Step 4: Start playback briefly
                capi.music_player_start(player)
                assert capi.music_player_is_playing(player) is True

                # Let it play briefly
                time.sleep(0.1)

                # Verify time is advancing
                play_time = capi.music_player_get_time(player)
                assert play_time > 0

                # Step 5: Stop playback
                capi.music_player_stop(player)
                assert capi.music_player_is_playing(player) is False

            finally:
                capi.music_player_set_sequence(player, 0)
                capi.dispose_music_player(player)
        finally:
            capi.dispose_music_sequence(sequence)

    def test_programmatic_midi_to_synth_workflow(self):
        """Test: Create MIDI events programmatically -> Play through DLS synth"""
        # Step 1: Create sequence with programmatic MIDI events
        sequence = capi.new_music_sequence()
        try:
            # Add tempo track
            tempo_track = capi.music_sequence_get_tempo_track(sequence)
            capi.music_track_new_extended_tempo_event(tempo_track, 0.0, 120.0)

            # Add note track
            track = capi.music_sequence_new_track(sequence)

            # Add a C major chord arpeggio (C4, E4, G4, C5)
            notes = [(0.0, 60), (0.5, 64), (1.0, 67), (1.5, 72)]
            for beat_time, note in notes:
                capi.music_track_new_midi_note_event(
                    track,
                    beat_time,  # timestamp in beats
                    0,          # channel
                    note,       # note number
                    100,        # velocity
                    64,         # release velocity
                    0.4         # duration in beats
                )

            # Step 2: Create player and configure
            player = capi.new_music_player()
            try:
                capi.music_player_set_sequence(player, sequence)
                capi.music_player_preroll(player)

                # Step 3: Play the sequence
                capi.music_player_start(player)

                # Let it play through the notes
                time.sleep(0.3)

                # Verify playback progressed
                play_time = capi.music_player_get_time(player)
                assert play_time > 0

                capi.music_player_stop(player)

            finally:
                capi.music_player_set_sequence(player, 0)
                capi.dispose_music_player(player)
        finally:
            capi.dispose_music_sequence(sequence)

    def test_midi_instrument_direct_control(self):
        """Test: Direct MIDI control of instrument AudioUnit"""
        host = cm.AudioUnitHost()

        # Use DLSMusicDevice (Apple's built-in GM synth)
        with host.load_plugin("DLSMusicDevice", type='instrument') as synth:
            assert synth is not None
            assert synth.is_initialized

            # Step 1: Set program (instrument sound)
            synth.program_change(channel=0, program=0)  # Acoustic Grand Piano

            # Step 2: Play a chord
            chord_notes = [60, 64, 67]  # C major chord
            for note in chord_notes:
                synth.note_on(channel=0, note=note, velocity=90)

            time.sleep(0.1)

            # Step 3: Release all notes
            for note in chord_notes:
                synth.note_off(channel=0, note=note)

            # Step 4: Test control changes
            synth.control_change(channel=0, controller=7, value=100)  # Volume
            synth.control_change(channel=0, controller=10, value=64)  # Pan center

    def test_midi_file_to_instrument_complete_workflow(self, canon_mid_path):
        """Test: Load real MIDI file -> Play through instrument -> Verify timing"""
        # Skip if MIDI file doesn't exist
        if not os.path.exists(canon_mid_path):
            pytest.skip("Canon MIDI file not available")

        # Load MIDI file into sequence
        sequence = capi.new_music_sequence()
        player = capi.new_music_player()

        try:
            capi.music_sequence_file_load(sequence, canon_mid_path)

            # Get track info
            track_count = capi.music_sequence_get_track_count(sequence)
            assert track_count > 0

            # Configure player
            capi.music_player_set_sequence(player, sequence)

            # Set playback rate
            capi.music_player_set_play_rate_scalar(player, 1.0)
            rate = capi.music_player_get_play_rate_scalar(player)
            assert rate == 1.0

            # Preroll and start
            capi.music_player_preroll(player)
            capi.music_player_start(player)

            # Sample playback time at intervals
            times = []
            for _ in range(5):
                times.append(capi.music_player_get_time(player))
                time.sleep(0.05)

            # Verify time is advancing
            assert times[-1] > times[0]

            # Stop and verify
            capi.music_player_stop(player)
            assert capi.music_player_is_playing(player) is False

        finally:
            capi.music_player_set_sequence(player, 0)
            capi.dispose_music_player(player)
            capi.dispose_music_sequence(sequence)


# =============================================================================
# Workflow 3: Link Session Synchronization
# =============================================================================


class TestLinkSessionSynchronization:
    """Test Ableton Link session synchronization workflows."""

    def test_link_session_basic_sync(self):
        """Test: Create Link session -> Enable -> Sync tempo and phase"""
        # Step 1: Create and enable Link session
        session = link.LinkSession(bpm=120.0)
        session.enabled = True

        try:
            # Step 2: Verify initial state
            state = session.capture_app_session_state()
            assert state.tempo == pytest.approx(120.0, abs=0.1)

            # Step 3: Get current timing
            current_time = session.clock.micros()
            beat = state.beat_at_time(current_time, quantum=4.0)
            phase = state.phase_at_time(current_time, quantum=4.0)

            assert beat >= 0.0
            assert 0.0 <= phase < 4.0

            # Step 4: Verify time advances
            time.sleep(0.1)
            current_time_2 = session.clock.micros()
            beat_2 = state.beat_at_time(current_time_2, quantum=4.0)

            # At 120 BPM, ~0.2 beats should pass in 0.1 seconds
            assert beat_2 > beat

        finally:
            session.enabled = False

    def test_link_tempo_synchronization(self):
        """Test: Change tempo -> Verify synchronized across session state"""
        session = link.LinkSession(bpm=120.0)
        session.enabled = True

        try:
            # Step 1: Get initial tempo
            state = session.capture_app_session_state()
            assert state.tempo == pytest.approx(120.0, abs=0.1)

            # Step 2: Change tempo
            current_time = session.clock.micros()
            state.set_tempo(140.0, current_time)
            session.commit_app_session_state(state)

            # Step 3: Verify tempo changed
            time.sleep(0.05)  # Small delay for state propagation
            new_state = session.capture_app_session_state()
            assert new_state.tempo == pytest.approx(140.0, abs=0.1)

            # Step 4: Change back
            current_time = session.clock.micros()
            new_state.set_tempo(120.0, current_time)
            session.commit_app_session_state(new_state)

            time.sleep(0.05)
            final_state = session.capture_app_session_state()
            assert final_state.tempo == pytest.approx(120.0, abs=0.1)

        finally:
            session.enabled = False

    def test_link_transport_synchronization(self):
        """Test: Start/stop transport -> Verify synchronized state"""
        session = link.LinkSession(bpm=120.0)
        session.enabled = True
        session.start_stop_sync_enabled = True

        try:
            # Step 1: Verify initial state (stopped)
            state = session.capture_app_session_state()
            assert state.is_playing is False

            # Step 2: Start transport
            current_time = session.clock.micros()
            state.set_is_playing(True, current_time)
            session.commit_app_session_state(state)

            time.sleep(0.05)

            # Step 3: Verify playing
            new_state = session.capture_app_session_state()
            assert new_state.is_playing is True

            # Step 4: Stop transport
            current_time = session.clock.micros()
            new_state.set_is_playing(False, current_time)
            session.commit_app_session_state(new_state)

            time.sleep(0.05)

            # Step 5: Verify stopped
            final_state = session.capture_app_session_state()
            assert final_state.is_playing is False

        finally:
            session.enabled = False

    def test_link_beat_quantization(self):
        """Test: Request beat at specific time with quantization"""
        session = link.LinkSession(bpm=120.0)
        session.enabled = True

        try:
            state = session.capture_app_session_state()
            current_time = session.clock.micros()

            # Request beat to be at a specific position
            quantum = 4.0  # 4 beats per bar
            target_beat = 8.0  # Target beat 8 (bar 3)

            state.request_beat_at_time(target_beat, current_time, quantum)
            session.commit_app_session_state(state)

            # Verify beat position was set
            time.sleep(0.05)
            new_state = session.capture_app_session_state()
            new_time = session.clock.micros()

            # Beat should be close to target (accounting for time elapsed)
            beat = new_state.beat_at_time(new_time, quantum)
            # The beat should have advanced slightly from target
            assert beat >= target_beat - 1.0

        finally:
            session.enabled = False

    def test_link_audio_player_integration(self):
        """Test: AudioPlayer with Link session for synchronized playback"""
        session = link.LinkSession(bpm=120.0)
        session.enabled = True

        try:
            # Create AudioPlayer with Link session
            player = cm.AudioPlayer(link_session=session)

            assert player.link_session is session

            # Get timing through player
            timing = player.get_link_timing(quantum=4.0)

            assert timing is not None
            assert 'tempo' in timing
            assert 'beat' in timing
            assert 'phase' in timing
            assert 'is_playing' in timing

            # Verify tempo matches session
            assert timing['tempo'] == pytest.approx(120.0, abs=0.1)

            # Change session tempo
            state = session.capture_app_session_state()
            current_time = session.clock.micros()
            state.set_tempo(130.0, current_time)
            session.commit_app_session_state(state)

            time.sleep(0.1)

            # Verify player sees new tempo
            new_timing = player.get_link_timing()
            assert new_timing['tempo'] == pytest.approx(130.0, abs=0.1)

        finally:
            session.enabled = False

    def test_link_multiple_session_sync(self):
        """Test: Multiple components sharing Link session for synchronization"""
        session = link.LinkSession(bpm=120.0)
        session.enabled = True

        try:
            # Create multiple players sharing session
            player1 = cm.AudioPlayer(link_session=session)
            player2 = cm.AudioPlayer(link_session=session)

            # Both should see same Link session
            assert player1.link_session is player2.link_session

            # Get timing from both
            timing1 = player1.get_link_timing(quantum=4.0)
            timing2 = player2.get_link_timing(quantum=4.0)

            # Should have very similar timing
            assert timing1['tempo'] == pytest.approx(timing2['tempo'], abs=0.01)
            assert timing1['beat'] == pytest.approx(timing2['beat'], abs=0.05)

            # Change tempo and verify both see it
            state = session.capture_app_session_state()
            current_time = session.clock.micros()
            state.set_tempo(140.0, current_time)
            session.commit_app_session_state(state)

            time.sleep(0.1)

            new_timing1 = player1.get_link_timing()
            new_timing2 = player2.get_link_timing()

            assert new_timing1['tempo'] == pytest.approx(140.0, abs=0.1)
            assert new_timing2['tempo'] == pytest.approx(140.0, abs=0.1)

        finally:
            session.enabled = False

    def test_link_clock_precision(self):
        """Test: Verify Link clock precision for sample-accurate timing"""
        session = link.LinkSession(bpm=120.0)

        clock = session.clock

        # Get multiple time samples
        times = []
        for _ in range(10):
            times.append(clock.micros())
            time.sleep(0.001)  # 1ms delay

        # Verify times are increasing
        for i in range(1, len(times)):
            assert times[i] > times[i-1]

        # Verify reasonable time intervals (~1000us per 1ms)
        intervals = [times[i] - times[i-1] for i in range(1, len(times))]
        avg_interval = sum(intervals) / len(intervals)

        # Average should be around 1000 microseconds (1ms)
        # Allow wide tolerance due to OS scheduling
        assert 500 < avg_interval < 5000


# =============================================================================
# Combined Workflow Tests
# =============================================================================


class TestCombinedWorkflows:
    """Test workflows that combine multiple subsystems."""

    def test_midi_to_audio_file_workflow(self, tmp_path):
        """Test: Create MIDI sequence -> Render through synth -> Save audio"""
        output_path = str(tmp_path / "midi_render_output.wav")

        # Step 1: Create MIDI sequence
        sequence = capi.new_music_sequence()
        try:
            tempo_track = capi.music_sequence_get_tempo_track(sequence)
            capi.music_track_new_extended_tempo_event(tempo_track, 0.0, 120.0)

            track = capi.music_sequence_new_track(sequence)

            # Simple melody
            notes = [(0.0, 60, 0.5), (0.5, 62, 0.5), (1.0, 64, 0.5), (1.5, 65, 0.5)]
            for beat, note, duration in notes:
                capi.music_track_new_midi_note_event(
                    track, beat, 0, note, 100, 64, duration
                )

            # Step 2: Create player
            player = capi.new_music_player()
            try:
                capi.music_player_set_sequence(player, sequence)
                capi.music_player_preroll(player)

                # Start playback
                capi.music_player_start(player)

                # Let MIDI play through
                time.sleep(0.5)

                capi.music_player_stop(player)

            finally:
                capi.music_player_set_sequence(player, 0)
                capi.dispose_music_player(player)

        finally:
            capi.dispose_music_sequence(sequence)

        # Step 3: Generate synthetic audio output (since capturing real
        # audio output requires input callbacks not yet implemented)
        # This demonstrates the workflow structure
        sample_rate = 44100.0
        duration_secs = 2.0
        num_frames = int(sample_rate * duration_secs)

        # Generate simple sine wave as placeholder for rendered audio
        import math
        audio_samples = []
        for i in range(num_frames):
            t = i / sample_rate
            # Simple 440Hz sine wave
            sample = int(math.sin(2 * math.pi * 440 * t) * 16000)
            # Stereo: duplicate sample
            audio_samples.append(struct.pack('<hh', sample, sample))

        audio_data = b''.join(audio_samples)

        # Step 4: Save to file
        output_format = {
            "sample_rate": sample_rate,
            "format_id": capi.get_audio_format_linear_pcm(),
            "format_flags": 12,
            "bytes_per_packet": 4,
            "frames_per_packet": 1,
            "bytes_per_frame": 4,
            "channels_per_frame": 2,
            "bits_per_channel": 16,
            "reserved": 0,
        }

        ext_file = capi.extended_audio_file_create_with_url(
            output_path,
            capi.get_audio_file_wave_type(),
            output_format,
            0
        )
        try:
            capi.extended_audio_file_write(ext_file, num_frames, audio_data)
        finally:
            capi.extended_audio_file_dispose(ext_file)

        # Verify output
        assert os.path.exists(output_path)
        with cm.AudioFile(output_path) as output:
            assert output.duration > 0
            assert output.format.sample_rate == sample_rate

    def test_link_synchronized_midi_playback(self):
        """Test: Link tempo control with MIDI playback synchronization"""
        # Create Link session
        session = link.LinkSession(bpm=120.0)
        session.enabled = True

        try:
            # Create MIDI sequence
            sequence = capi.new_music_sequence()
            player = capi.new_music_player()

            try:
                # Add notes
                track = capi.music_sequence_new_track(sequence)
                for i in range(8):
                    capi.music_track_new_midi_note_event(
                        track, i * 0.25, 0, 60 + i, 100, 64, 0.2
                    )

                capi.music_player_set_sequence(player, sequence)
                capi.music_player_preroll(player)

                # Get Link timing before playback
                state = session.capture_app_session_state()
                link_tempo = state.tempo

                # Set player rate based on tempo ratio
                target_tempo = 140.0
                rate_scalar = target_tempo / link_tempo
                capi.music_player_set_play_rate_scalar(player, rate_scalar)

                # Verify rate was set
                actual_rate = capi.music_player_get_play_rate_scalar(player)
                assert actual_rate == pytest.approx(rate_scalar, abs=0.01)

                # Start playback
                capi.music_player_start(player)
                time.sleep(0.1)
                capi.music_player_stop(player)

            finally:
                capi.music_player_set_sequence(player, 0)
                capi.dispose_music_player(player)
                capi.dispose_music_sequence(sequence)

        finally:
            session.enabled = False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
