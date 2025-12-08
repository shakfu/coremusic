"""Tests for AudioConverter and ExtendedAudioFile object-oriented classes."""

import os
import tempfile
import pytest
from pathlib import Path
import coremusic as cm
import coremusic.capi as capi


class TestAudioConverter:
    """Test AudioConverter object-oriented wrapper"""

    def test_audio_converter_creation(self, source_format_obj, dest_format_mono_obj):
        """Test AudioConverter object creation"""
        converter = cm.AudioConverter(source_format_obj, dest_format_mono_obj)
        assert isinstance(converter, cm.AudioConverter)
        assert isinstance(converter, cm.CoreAudioObject)
        assert not converter.is_disposed
        assert converter.object_id != 0

    def test_audio_converter_properties(self, source_format_obj, dest_format_mono_obj):
        """Test AudioConverter source and dest format properties"""
        converter = cm.AudioConverter(source_format_obj, dest_format_mono_obj)
        src = converter.source_format
        assert isinstance(src, cm.AudioFormat)
        assert src.sample_rate == 44100.0
        assert src.channels_per_frame == 2
        dst = converter.dest_format
        assert isinstance(dst, cm.AudioFormat)
        assert dst.sample_rate == 44100.0
        assert dst.channels_per_frame == 1

    def test_audio_converter_convert(self, source_format_obj, dest_format_mono_obj):
        """Test AudioConverter conversion"""
        converter = cm.AudioConverter(source_format_obj, dest_format_mono_obj)
        num_frames = 100
        input_data = b"\x00\x01" * (num_frames * 2)
        output_data = converter.convert(input_data)
        assert isinstance(output_data, bytes)
        assert len(output_data) > 0
        assert len(output_data) < len(input_data)

    def test_audio_converter_sample_rate_conversion(
        self, source_format_obj, dest_format_48k_obj
    ):
        """Test AudioConverter creation for sample rate conversion"""
        converter = cm.AudioConverter(source_format_obj, dest_format_48k_obj)
        assert isinstance(converter, cm.AudioConverter)
        assert not converter.is_disposed
        converter.dispose()

    def test_audio_converter_reset(self, source_format_obj, dest_format_mono_obj):
        """Test AudioConverter reset"""
        converter = cm.AudioConverter(source_format_obj, dest_format_mono_obj)
        input_data = b"\x00\x01" * 200
        output1 = converter.convert(input_data)
        converter.reset()
        output2 = converter.convert(input_data)
        assert isinstance(output2, bytes)
        assert len(output2) > 0

    def test_audio_converter_context_manager(self, source_format_obj, dest_format_mono_obj):
        """Test AudioConverter as context manager"""
        with cm.AudioConverter(source_format_obj, dest_format_mono_obj) as converter:
            assert isinstance(converter, cm.AudioConverter)
            assert not converter.is_disposed
            input_data = b"\x00\x01\x02\x03" * 100
            output_data = converter.convert(input_data)
            assert len(output_data) > 0
        assert converter.is_disposed

    def test_audio_converter_manual_disposal(self, source_format_obj, dest_format_mono_obj):
        """Test AudioConverter manual disposal"""
        converter = cm.AudioConverter(source_format_obj, dest_format_mono_obj)
        assert not converter.is_disposed
        converter.dispose()
        assert converter.is_disposed
        converter.dispose()
        assert converter.is_disposed

    def test_audio_converter_operations_after_disposal(
        self, source_format_obj, dest_format_mono_obj
    ):
        """Test operations after disposal raise errors"""
        converter = cm.AudioConverter(source_format_obj, dest_format_mono_obj)
        converter.dispose()
        with pytest.raises(RuntimeError, match="has been disposed"):
            converter.convert(b"\x00\x01")
        with pytest.raises(RuntimeError, match="has been disposed"):
            converter.reset()

    def test_audio_converter_repr(self, source_format_obj, dest_format_mono_obj):
        """Test AudioConverter string representation"""
        converter = cm.AudioConverter(source_format_obj, dest_format_mono_obj)
        repr_str = repr(converter)
        assert "AudioConverter" in repr_str
        assert "44100.0" in repr_str

    def test_audio_converter_with_real_file(
        self, amen_wav_path, source_format_obj, dest_format_mono_obj
    ):
        """Test AudioConverter with real audio file data"""
        with cm.AudioFile(amen_wav_path) as audio_file:
            data, packet_count = audio_file.read_packets(0, 100)
            assert len(data) > 0
            with cm.AudioConverter(source_format_obj, dest_format_mono_obj) as converter:
                mono_data = converter.convert(data)
                assert len(mono_data) > 0
                assert len(mono_data) < len(data)

    def test_audio_converter_error_handling(self):
        """Test AudioConverter error handling"""
        invalid_format = cm.AudioFormat(
            sample_rate=0,
            format_id="lpcm",
            format_flags=0,
            bytes_per_packet=0,
            frames_per_packet=0,
            bytes_per_frame=0,
            channels_per_frame=0,
            bits_per_channel=0,
        )
        valid_format = cm.AudioFormat(
            sample_rate=44100.0,
            format_id="lpcm",
            format_flags=12,
            bytes_per_packet=4,
            frames_per_packet=1,
            bytes_per_frame=4,
            channels_per_frame=2,
            bits_per_channel=16,
        )
        with pytest.raises(cm.AudioConverterError):
            cm.AudioConverter(invalid_format, valid_format)

    def test_audio_converter_sample_rate_conversion_44100_to_48000(
        self, source_format_obj, dest_format_48k_obj
    ):
        """Test sample rate conversion from 44.1kHz to 48kHz using callback API"""
        converter = cm.AudioConverter(source_format_obj, dest_format_48k_obj)
        num_input_frames = 100
        input_data = b"\x00\x01\x02\x03" * num_input_frames
        output_data = converter.convert_with_callback(input_data, num_input_frames)
        assert isinstance(output_data, bytes)
        assert len(output_data) > 0
        expected_output_size = int(num_input_frames * (48000.0 / 44100.0) * 4)
        assert abs(len(output_data) - expected_output_size) < expected_output_size * 0.2
        converter.dispose()

    def test_audio_converter_sample_rate_conversion_48000_to_44100(self):
        """Test sample rate conversion from 48kHz to 44.1kHz using callback API"""
        source_format = cm.AudioFormat(
            sample_rate=48000.0,
            format_id="lpcm",
            format_flags=12,
            bytes_per_packet=4,
            frames_per_packet=1,
            bytes_per_frame=4,
            channels_per_frame=2,
            bits_per_channel=16,
        )
        dest_format = cm.AudioFormat(
            sample_rate=44100.0,
            format_id="lpcm",
            format_flags=12,
            bytes_per_packet=4,
            frames_per_packet=1,
            bytes_per_frame=4,
            channels_per_frame=2,
            bits_per_channel=16,
        )
        converter = cm.AudioConverter(source_format, dest_format)
        num_input_frames = 100
        input_data = b"\x00\x01\x02\x03" * num_input_frames
        output_data = converter.convert_with_callback(input_data, num_input_frames)
        assert isinstance(output_data, bytes)
        assert len(output_data) > 0
        expected_output_size = int(num_input_frames * (44100.0 / 48000.0) * 4)
        assert abs(len(output_data) - expected_output_size) < expected_output_size * 0.2
        converter.dispose()

    def test_audio_converter_sample_rate_conversion_with_real_file(self, amen_wav_path):
        """Test sample rate conversion with real audio file"""
        source_format = cm.AudioFormat(
            sample_rate=44100.0,
            format_id="lpcm",
            format_flags=12,
            bytes_per_packet=4,
            frames_per_packet=1,
            bytes_per_frame=4,
            channels_per_frame=2,
            bits_per_channel=16,
        )
        dest_format = cm.AudioFormat(
            sample_rate=48000.0,
            format_id="lpcm",
            format_flags=12,
            bytes_per_packet=4,
            frames_per_packet=1,
            bytes_per_frame=4,
            channels_per_frame=2,
            bits_per_channel=16,
        )
        with cm.AudioFile(amen_wav_path) as audio_file:
            data, packet_count = audio_file.read_packets(0, 1000)
            assert len(data) > 0
            with cm.AudioConverter(source_format, dest_format) as converter:
                converted_data = converter.convert_with_callback(data, packet_count)
                assert len(converted_data) > 0
                expected_size = int(len(data) * (48000.0 / 44100.0))
                assert abs(len(converted_data) - expected_size) < expected_size * 0.2

    def test_audio_converter_combined_sample_rate_and_channel_conversion(self):
        """Test combined sample rate and channel conversion (44.1kHz stereo to 48kHz mono)"""
        source_format = cm.AudioFormat(
            sample_rate=44100.0,
            format_id="lpcm",
            format_flags=12,
            bytes_per_packet=4,
            frames_per_packet=1,
            bytes_per_frame=4,
            channels_per_frame=2,
            bits_per_channel=16,
        )
        dest_format = cm.AudioFormat(
            sample_rate=48000.0,
            format_id="lpcm",
            format_flags=12,
            bytes_per_packet=2,
            frames_per_packet=1,
            bytes_per_frame=2,
            channels_per_frame=1,
            bits_per_channel=16,
        )
        converter = cm.AudioConverter(source_format, dest_format)
        num_input_frames = 100
        input_data = b"\x00\x01\x02\x03" * num_input_frames
        output_data = converter.convert_with_callback(input_data, num_input_frames)
        assert isinstance(output_data, bytes)
        assert len(output_data) > 0
        expected_frames = int(num_input_frames * (48000.0 / 44100.0))
        expected_size = expected_frames * 2
        assert abs(len(output_data) - expected_size) < expected_size * 0.3
        converter.dispose()

    def test_audio_converter_with_callback_auto_output_packet_count(
        self, source_format_obj, dest_format_48k_obj
    ):
        """Test convert_with_callback with automatic output packet count calculation"""
        converter = cm.AudioConverter(source_format_obj, dest_format_48k_obj)
        num_input_frames = 100
        input_data = b"\x00\x01\x02\x03" * num_input_frames
        output_data = converter.convert_with_callback(
            input_data, num_input_frames, output_packet_count=None
        )
        assert isinstance(output_data, bytes)
        assert len(output_data) > 0
        converter.dispose()


class TestExtendedAudioFile:
    """Test ExtendedAudioFile object-oriented wrapper"""

    def test_extended_audio_file_creation(self, amen_wav_path):
        """Test ExtendedAudioFile object creation"""
        ext_file = cm.ExtendedAudioFile(amen_wav_path)
        assert isinstance(ext_file, cm.ExtendedAudioFile)
        assert isinstance(ext_file, cm.CoreAudioObject)
        assert not ext_file.is_disposed
        assert ext_file.object_id == 0
        ext_file_path = cm.ExtendedAudioFile(Path(amen_wav_path))
        assert isinstance(ext_file_path, cm.ExtendedAudioFile)
        assert str(ext_file_path._path) == str(Path(amen_wav_path))

    def test_extended_audio_file_open_close(self, amen_wav_path):
        """Test ExtendedAudioFile opening and closing"""
        ext_file = cm.ExtendedAudioFile(amen_wav_path)
        result = ext_file.open()
        assert result is ext_file
        assert ext_file.object_id != 0
        assert not ext_file.is_disposed
        ext_file.close()
        assert ext_file.is_disposed

    def test_extended_audio_file_context_manager(self, amen_wav_path):
        """Test ExtendedAudioFile as context manager"""
        with cm.ExtendedAudioFile(amen_wav_path) as ext_file:
            assert isinstance(ext_file, cm.ExtendedAudioFile)
            assert ext_file.object_id != 0
            assert not ext_file.is_disposed
        assert ext_file.is_disposed

    def test_extended_audio_file_file_format_property(self, amen_wav_path):
        """Test ExtendedAudioFile file_format property"""
        with cm.ExtendedAudioFile(amen_wav_path) as ext_file:
            format = ext_file.file_format
            assert isinstance(format, cm.AudioFormat)
            assert format.sample_rate == 44100.0
            assert format.channels_per_frame == 2
            assert format.bits_per_channel == 16

    def test_extended_audio_file_client_format_property(
        self, amen_wav_path, pcm_format
    ):
        """Test ExtendedAudioFile client_format property"""
        with cm.ExtendedAudioFile(amen_wav_path) as ext_file:
            assert ext_file.client_format is None
            ext_file.client_format = pcm_format
            client_fmt = ext_file.client_format
            assert isinstance(client_fmt, cm.AudioFormat)
            assert client_fmt.sample_rate == 44100.0

    def test_extended_audio_file_read(self, amen_wav_path):
        """Test ExtendedAudioFile frame reading"""
        with cm.ExtendedAudioFile(amen_wav_path) as ext_file:
            data, frames_read = ext_file.read(1000)
            assert isinstance(data, bytes)
            assert isinstance(frames_read, int)
            assert len(data) > 0
            assert frames_read > 0
            assert frames_read <= 1000

    def test_extended_audio_file_create(self, temp_audio_file, pcm_format):
        """Test ExtendedAudioFile.create class method"""
        ext_file = cm.ExtendedAudioFile.create(
            temp_audio_file, capi.get_audio_file_wave_type(), pcm_format
        )
        try:
            assert isinstance(ext_file, cm.ExtendedAudioFile)
            assert ext_file.object_id != 0
            assert not ext_file.is_disposed
            num_frames = 1000
            audio_data = bytes([(i % 256) for i in range(num_frames * 4)])
            ext_file.write(num_frames, audio_data)
            assert os.path.exists(temp_audio_file)
            assert os.path.getsize(temp_audio_file) > 0
        finally:
            ext_file.close()

    def test_extended_audio_file_write(self, temp_audio_file, pcm_format):
        """Test ExtendedAudioFile write method"""
        ext_file = cm.ExtendedAudioFile.create(
            temp_audio_file, capi.get_audio_file_wave_type(), pcm_format
        )
        try:
            num_frames = 500
            test_pattern = bytes([(i * 13 % 256) for i in range(num_frames * 4)])
            ext_file.write(num_frames, test_pattern)
            assert os.path.exists(temp_audio_file)
        finally:
            ext_file.close()

    def test_extended_audio_file_write_then_read(self, temp_audio_file, pcm_format):
        """Test write-then-read round-trip"""
        num_frames = 500
        test_pattern = bytes([(i * 13 % 256) for i in range(num_frames * 4)])
        with cm.ExtendedAudioFile.create(
            temp_audio_file, capi.get_audio_file_wave_type(), pcm_format
        ) as ext_file:
            ext_file.write(num_frames, test_pattern)
        with cm.ExtendedAudioFile(temp_audio_file) as ext_file:
            audio_data, frames_read = ext_file.read(num_frames)
            assert frames_read == num_frames
            assert len(audio_data) > 0
            assert len(audio_data) == len(test_pattern)

    def test_extended_audio_file_read_multiple_chunks(self, amen_wav_path):
        """Test reading in multiple chunks"""
        with cm.ExtendedAudioFile(amen_wav_path) as ext_file:
            total_frames = 0
            chunk_size = 1000
            data1, frames1 = ext_file.read(chunk_size)
            assert frames1 > 0
            total_frames += frames1
            data2, frames2 = ext_file.read(chunk_size)
            assert frames2 > 0
            total_frames += frames2
            data3, frames3 = ext_file.read(chunk_size)
            assert frames3 > 0
            total_frames += frames3
            assert total_frames > 0

    def test_extended_audio_file_repr(self, amen_wav_path):
        """Test ExtendedAudioFile string representation"""
        ext_file = cm.ExtendedAudioFile(amen_wav_path)
        repr_str = repr(ext_file)
        assert "ExtendedAudioFile" in repr_str
        assert "closed" in repr_str
        with ext_file:
            repr_str = repr(ext_file)
            assert "ExtendedAudioFile" in repr_str
            assert "open" in repr_str

    def test_extended_audio_file_error_handling(self):
        """Test ExtendedAudioFile error handling"""
        with pytest.raises(cm.AudioFileError):
            with cm.ExtendedAudioFile("/nonexistent/path.wav"):
                pass

    def test_extended_audio_file_operations_on_disposed_object(self, amen_wav_path):
        """Test operations on disposed ExtendedAudioFile object"""
        ext_file = cm.ExtendedAudioFile(amen_wav_path)
        ext_file.open()
        ext_file.close()
        with pytest.raises(RuntimeError, match="has been disposed"):
            ext_file.read(100)
        with pytest.raises(RuntimeError, match="has been disposed"):
            _ = ext_file.file_format

    def test_extended_audio_file_format_conversion(self, amen_wav_path):
        """Test ExtendedAudioFile automatic format conversion"""
        with cm.ExtendedAudioFile(amen_wav_path) as ext_file:
            file_format = ext_file.file_format
            assert file_format.channels_per_frame == 2
            mono_format = cm.AudioFormat(
                sample_rate=44100.0,
                format_id="lpcm",
                format_flags=12,
                bytes_per_packet=2,
                frames_per_packet=1,
                bytes_per_frame=2,
                channels_per_frame=1,
                bits_per_channel=16,
            )
            ext_file.client_format = mono_format
            data, frames_read = ext_file.read(100)
            assert len(data) > 0
            assert frames_read > 0
            expected_bytes = frames_read * 2
            assert len(data) == expected_bytes

    def test_extended_audio_file_manual_disposal(self, amen_wav_path):
        """Test ExtendedAudioFile manual disposal"""
        ext_file = cm.ExtendedAudioFile(amen_wav_path)
        ext_file.open()
        assert not ext_file.is_disposed
        ext_file.dispose()
        assert ext_file.is_disposed
        ext_file.dispose()
        assert ext_file.is_disposed
