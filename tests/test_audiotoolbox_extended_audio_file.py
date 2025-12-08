"""Tests for ExtendedAudioFile functional API."""

import os
import tempfile
import struct
import pytest
import coremusic as cm
import coremusic.capi as capi


class TestExtendedAudioFileAPI:
    """Test ExtendedAudioFile functional API"""

    def test_extended_audio_file_open_url(self, amen_wav_path):
        """Test opening an existing audio file"""
        ext_file_id = capi.extended_audio_file_open_url(amen_wav_path)
        assert ext_file_id is not None
        assert ext_file_id > 0
        capi.extended_audio_file_dispose(ext_file_id)

    def test_extended_audio_file_read(self, amen_wav_path):
        """Test reading frames from audio file"""
        ext_file_id = capi.extended_audio_file_open_url(amen_wav_path)
        try:
            audio_data, frames_read = capi.extended_audio_file_read(ext_file_id, 1000)
            assert isinstance(audio_data, bytes)
            assert isinstance(frames_read, int)
            assert len(audio_data) > 0
            assert frames_read > 0
            assert frames_read <= 1000
        finally:
            capi.extended_audio_file_dispose(ext_file_id)

    def test_extended_audio_file_get_file_data_format(self, amen_wav_path):
        """Test getting file data format property"""
        ext_file_id = capi.extended_audio_file_open_url(amen_wav_path)
        try:
            property_id = capi.get_extended_audio_file_property_file_data_format()
            format_data = capi.extended_audio_file_get_property(
                ext_file_id, property_id
            )
            assert isinstance(format_data, bytes)
            assert len(format_data) >= 40
            asbd = struct.unpack("<dLLLLLLLL", format_data[:40])
            (
                sample_rate,
                format_id,
                format_flags,
                bytes_per_packet,
                frames_per_packet,
                bytes_per_frame,
                channels_per_frame,
                bits_per_channel,
                reserved,
            ) = asbd
            assert sample_rate == 44100.0
            assert channels_per_frame == 2
            assert bits_per_channel == 16
        finally:
            capi.extended_audio_file_dispose(ext_file_id)

    def test_extended_audio_file_create_and_write(
        self, temp_audio_file, pcm_format_dict
    ):
        """Test creating and writing to a new audio file"""
        ext_file_id = capi.extended_audio_file_create_with_url(
            temp_audio_file, capi.get_audio_file_wave_type(), pcm_format_dict, 0
        )
        try:
            num_frames = 1000
            audio_data = bytes([(i % 256) for i in range(num_frames * 4)])
            capi.extended_audio_file_write(ext_file_id, num_frames, audio_data)
            assert os.path.exists(temp_audio_file)
            assert os.path.getsize(temp_audio_file) > 0
        finally:
            capi.extended_audio_file_dispose(ext_file_id)

    def test_extended_audio_file_set_client_data_format(
        self, amen_wav_path, pcm_format_dict
    ):
        """Test setting client data format for format conversion"""
        ext_file_id = capi.extended_audio_file_open_url(amen_wav_path)
        try:
            property_id = capi.get_extended_audio_file_property_client_data_format()
            asbd_bytes = struct.pack(
                "<dLLLLLLLL",
                pcm_format_dict["sample_rate"],
                pcm_format_dict["format_id"],
                pcm_format_dict["format_flags"],
                pcm_format_dict["bytes_per_packet"],
                pcm_format_dict["frames_per_packet"],
                pcm_format_dict["bytes_per_frame"],
                pcm_format_dict["channels_per_frame"],
                pcm_format_dict["bits_per_channel"],
                pcm_format_dict["reserved"],
            )
            capi.extended_audio_file_set_property(ext_file_id, property_id, asbd_bytes)
            audio_data, frames_read = capi.extended_audio_file_read(ext_file_id, 100)
            assert len(audio_data) > 0
            assert frames_read > 0
        finally:
            capi.extended_audio_file_dispose(ext_file_id)

    def test_extended_audio_file_get_file_length_frames(self, amen_wav_path):
        """Test getting file length in frames"""
        ext_file_id = capi.extended_audio_file_open_url(amen_wav_path)
        try:
            property_id = capi.get_extended_audio_file_property_file_length_frames()
            length_data = capi.extended_audio_file_get_property(
                ext_file_id, property_id
            )
            assert isinstance(length_data, bytes)
            assert len(length_data) >= 8
            frame_count = struct.unpack("<q", length_data[:8])[0]
            assert frame_count > 0
            assert frame_count > 100000
        finally:
            capi.extended_audio_file_dispose(ext_file_id)

    def test_extended_audio_file_property_getters(self):
        """Test ExtendedAudioFile property ID getter functions"""
        assert isinstance(capi.get_extended_audio_file_property_file_data_format(), int)
        assert isinstance(
            capi.get_extended_audio_file_property_file_channel_layout(), int
        )
        assert isinstance(
            capi.get_extended_audio_file_property_client_data_format(), int
        )
        assert isinstance(
            capi.get_extended_audio_file_property_client_channel_layout(), int
        )
        assert isinstance(
            capi.get_extended_audio_file_property_codec_manufacturer(), int
        )
        assert isinstance(capi.get_extended_audio_file_property_audio_file(), int)
        assert isinstance(
            capi.get_extended_audio_file_property_file_length_frames(), int
        )

    def test_extended_audio_file_read_multiple_chunks(self, amen_wav_path):
        """Test reading audio file in multiple chunks"""
        ext_file_id = capi.extended_audio_file_open_url(amen_wav_path)
        try:
            total_frames_read = 0
            chunk_size = 1000
            audio_data1, frames_read1 = capi.extended_audio_file_read(
                ext_file_id, chunk_size
            )
            assert frames_read1 > 0
            total_frames_read += frames_read1
            audio_data2, frames_read2 = capi.extended_audio_file_read(
                ext_file_id, chunk_size
            )
            assert frames_read2 > 0
            total_frames_read += frames_read2
            audio_data3, frames_read3 = capi.extended_audio_file_read(
                ext_file_id, chunk_size
            )
            assert frames_read3 > 0
            total_frames_read += frames_read3
            assert total_frames_read > 0
        finally:
            capi.extended_audio_file_dispose(ext_file_id)

    def test_extended_audio_file_create_overwrite(
        self, temp_audio_file, pcm_format_dict
    ):
        """Test creating file with overwrite flag"""
        with open(temp_audio_file, "wb") as f:
            f.write(b"dummy data")
        assert os.path.exists(temp_audio_file)
        initial_size = os.path.getsize(temp_audio_file)
        ext_file_id = capi.extended_audio_file_create_with_url(
            temp_audio_file, capi.get_audio_file_wave_type(), pcm_format_dict, 0
        )
        try:
            num_frames = 100
            audio_data = b"\x00\x01\x02\x03" * num_frames
            capi.extended_audio_file_write(ext_file_id, num_frames, audio_data)
        finally:
            capi.extended_audio_file_dispose(ext_file_id)
        assert os.path.exists(temp_audio_file)
        final_size = os.path.getsize(temp_audio_file)
        assert final_size != initial_size

    def test_extended_audio_file_error_handling(self):
        """Test ExtendedAudioFile error handling"""
        with pytest.raises((RuntimeError, cm.CoreAudioError)):
            capi.extended_audio_file_open_url("/nonexistent/path/to/file.wav")
        with pytest.raises((RuntimeError, cm.CoreAudioError)):
            capi.extended_audio_file_read(999999, 100)

    def test_extended_audio_file_write_then_read(
        self, temp_audio_file, pcm_format_dict
    ):
        """Test write-then-read round-trip"""
        ext_file_id = capi.extended_audio_file_create_with_url(
            temp_audio_file, capi.get_audio_file_wave_type(), pcm_format_dict, 0
        )
        num_frames = 500
        test_pattern = bytes([(i * 13 % 256) for i in range(num_frames * 4)])
        try:
            capi.extended_audio_file_write(ext_file_id, num_frames, test_pattern)
        finally:
            capi.extended_audio_file_dispose(ext_file_id)
        ext_file_id_read = capi.extended_audio_file_open_url(temp_audio_file)
        try:
            audio_data, frames_read = capi.extended_audio_file_read(
                ext_file_id_read, num_frames
            )
            assert frames_read == num_frames
            assert len(audio_data) > 0
            assert len(audio_data) == len(test_pattern)
        finally:
            capi.extended_audio_file_dispose(ext_file_id_read)

    def test_extended_audio_file_read_entire_file(self, amen_wav_path):
        """Test reading entire audio file"""
        ext_file_id = capi.extended_audio_file_open_url(amen_wav_path)
        try:
            property_id = capi.get_extended_audio_file_property_file_length_frames()
            length_data = capi.extended_audio_file_get_property(
                ext_file_id, property_id
            )
            total_frames = struct.unpack("<q", length_data[:8])[0]
            audio_data, frames_read = capi.extended_audio_file_read(
                ext_file_id, total_frames
            )
            assert frames_read > 0
            assert frames_read <= total_frames
            assert len(audio_data) > 0
        finally:
            capi.extended_audio_file_dispose(ext_file_id)
