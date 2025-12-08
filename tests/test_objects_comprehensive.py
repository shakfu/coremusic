"""Comprehensive tests for the object-oriented coremusic API functionality."""

import pytest
import os
import coremusic as cm
import coremusic.capi as capi


class TestObjectOrientedAPIFunctionality:
    """Test that the OO API is fully functional"""

    def test_audio_format_functionality(self):
        """Test AudioFormat class functionality"""
        format = cm.AudioFormat(
            44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16
        )
        assert format.is_pcm
        assert format.is_stereo
        assert not format.is_mono
        assert format.sample_rate == 44100.0
        assert format.format_id == "lpcm"
        assert format.channels_per_frame == 2
        assert format.bits_per_channel == 16
        repr_str = repr(format)
        assert "AudioFormat" in repr_str
        assert "44100.0" in repr_str

    def test_exception_hierarchy_functionality(self):
        """Test exception hierarchy functionality"""
        try:
            raise cm.CoreAudioError("Test error", 42)
        except cm.CoreAudioError as e:
            assert str(e) == "Test error"
            assert e.status_code == 42
        with pytest.raises(cm.AudioFileError):
            raise cm.AudioFileError("File error")
        with pytest.raises(cm.CoreAudioError):
            raise cm.MIDIError("MIDI error")

    def test_core_audio_object_functionality(self):
        """Test CoreAudioObject base functionality"""
        obj = cm.CoreAudioObject()
        assert not obj.is_disposed
        obj.dispose()
        assert obj.is_disposed
        with pytest.raises(RuntimeError, match="has been disposed"):
            obj._ensure_not_disposed()


    def test_audio_file_functionality(self, amen_wav_path):
        """Test AudioFile functionality with real file"""
        with cm.AudioFile(amen_wav_path) as audio_file:
            assert isinstance(audio_file, cm.AudioFile)
            assert not audio_file.is_disposed
            format = audio_file.format
            assert isinstance(format, cm.AudioFormat)
            data, packet_count = audio_file.read_packets(0, 10)
            assert isinstance(data, bytes)
            assert isinstance(packet_count, int)
            assert len(data) > 0
        assert audio_file.is_disposed

    def test_audio_file_stream_functionality(self):
        """Test AudioFileStream functionality"""
        stream = cm.AudioFileStream()
        stream.open()
        assert not stream.is_disposed
        ready = stream.ready_to_produce_packets
        assert isinstance(ready, bool)
        stream.close()
        assert stream.is_disposed

    def test_audio_queue_functionality(self):
        """Test AudioQueue functionality"""
        format = cm.AudioFormat(
            44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16
        )
        try:
            queue = cm.AudioQueue.new_output(format)
            assert isinstance(queue, cm.AudioQueue)
            assert not queue.is_disposed
            try:
                buffer = queue.allocate_buffer(1024)
                assert isinstance(buffer, cm.AudioBuffer)
                assert buffer.buffer_size == 1024
                queue.enqueue_buffer(buffer)
                try:
                    queue.start()
                    queue.stop()
                except cm.AudioQueueError:
                    pass
            finally:
                queue.dispose()
                assert queue.is_disposed
        except cm.AudioQueueError as e:
            # Check for paramErr (-50) which indicates no audio hardware
            if e.status_code == -50 or "paramErr" in str(e):
                pytest.skip("AudioQueue creation failed - no audio hardware available")
            else:
                raise

    def test_audio_component_description_functionality(self):
        """Test AudioComponentDescription functionality"""
        desc = cm.AudioComponentDescription(
            "auou", "def ", "appl", flags=1, flags_mask=2
        )
        assert desc.type == "auou"
        assert desc.subtype == "def "
        assert desc.manufacturer == "appl"
        assert desc.flags == 1
        assert desc.flags_mask == 2
        dict_repr = desc.to_dict()
        expected = {
            "type": capi.fourchar_to_int("auou"),
            "subtype": capi.fourchar_to_int("def "),
            "manufacturer": capi.fourchar_to_int("appl"),
            "flags": 1,
            "flags_mask": 2,
        }
        assert dict_repr == expected

    def test_audio_unit_functionality(self):
        """Test AudioUnit functionality"""
        try:
            unit = cm.AudioUnit.default_output()
            assert isinstance(unit, cm.AudioUnit)
            assert not unit.is_disposed
            assert not unit.is_initialized
            with unit:
                assert unit.is_initialized
            assert not unit.is_initialized
            assert unit.is_disposed
        except cm.AudioUnitError as e:
            if "not found" in str(e):
                pytest.skip(
                    "Default output AudioUnit not available in test environment"
                )
            else:
                raise

    def test_midi_client_functionality(self):
        """Test MIDIClient functionality"""
        try:
            client = cm.MIDIClient("Test Client")
        except cm.MIDIError:
            pytest.skip("MIDI services not available")
        assert isinstance(client, cm.MIDIClient)
        assert client.name == "Test Client"
        assert not client.is_disposed
        try:
            input_port = client.create_input_port("Test Input")
            output_port = client.create_output_port("Test Output")
            assert isinstance(input_port, cm.MIDIInputPort)
            assert isinstance(output_port, cm.MIDIOutputPort)
            assert input_port.name == "Test Input"
            assert output_port.name == "Test Output"
        finally:
            client.dispose()
            assert client.is_disposed

    def test_dual_api_consistency(self, amen_wav_path):
        """Test consistency between functional and OO APIs"""
        fourcc_func = capi.fourchar_to_int("TEST")
        format_oo = cm.AudioFormat(44100.0, "lpcm", channels_per_frame=2)
        assert isinstance(fourcc_func, int)
        assert format_oo.is_pcm
        func_file_id = capi.audio_file_open_url(amen_wav_path)
        try:
            func_data, func_count = capi.audio_file_read_packets(func_file_id, 0, 5)
        finally:
            capi.audio_file_close(func_file_id)
        with cm.AudioFile(amen_wav_path) as oo_file:
            oo_data, oo_count = oo_file.read_packets(0, 5)
        assert func_data == oo_data
        assert func_count == oo_count

    def test_error_handling_functionality(self):
        """Test error handling in OO API"""
        with pytest.raises(cm.AudioFileError):
            with cm.AudioFile("/nonexistent/file.wav"):
                pass
        invalid_format = cm.AudioFormat(0.0, "", channels_per_frame=0)
        with pytest.raises(cm.AudioQueueError):
            cm.AudioQueue.new_output(invalid_format)
        obj = cm.CoreAudioObject()
        obj.dispose()
        with pytest.raises(RuntimeError, match="has been disposed"):
            obj._ensure_not_disposed()

    def test_resource_management_functionality(self, amen_wav_path):
        """Test proper resource management"""
        objects = []
        for _ in range(3):
            audio_file = cm.AudioFile(amen_wav_path)
            audio_file.open()
            objects.append(audio_file)
        format = cm.AudioFormat(
            44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16
        )
        for _ in range(3):
            try:
                queue = cm.AudioQueue.new_output(format)
                objects.append(queue)
            except cm.AudioQueueError:
                pass
        for i in range(3):
            try:
                client = cm.MIDIClient(f"Test Client {i}")
                objects.append(client)
            except cm.MIDIError:
                pass  # MIDI not available
        for obj in objects:
            obj.dispose()
            assert obj.is_disposed

    def test_polymorphism_functionality(self):
        """Test polymorphic behavior"""
        objects = [
            cm.CoreAudioObject(),
            cm.AudioFile("/dummy/path"),
            cm.AudioFileStream(),
        ]
        format = cm.AudioFormat(
            44100.0, "lpcm", channels_per_frame=2, bits_per_channel=16
        )
        try:
            queue = cm.AudioQueue.new_output(format)
            objects.append(queue)
        except cm.AudioQueueError:
            pass
        try:
            client = cm.MIDIClient("Test Client")
            objects.append(client)
        except cm.MIDIError:
            pass
        for obj in objects:
            assert isinstance(obj, cm.CoreAudioObject)
            assert hasattr(obj, "is_disposed")
            assert hasattr(obj, "dispose")
            assert hasattr(obj, "_ensure_not_disposed")
            if not obj.is_disposed:
                obj.dispose()
