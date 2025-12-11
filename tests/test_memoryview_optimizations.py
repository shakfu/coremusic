"""Tests for memoryview-optimized zero-copy audio functions.

These tests verify the high-performance memoryview variants of audio functions
that avoid memory allocation and copying by working directly with caller-provided
buffers.
"""

import pytest

import coremusic as cm
from coremusic import capi


class TestAudioFileReadPacketsInto:
    """Tests for audio_file_read_packets_into() zero-copy function"""

    def test_read_packets_into_basic(self, amen_wav_path):
        """Test basic zero-copy packet reading"""
        file_id = capi.audio_file_open_url(amen_wav_path)
        try:
            # Get max packet size to determine buffer size
            max_packet_size_bytes = capi.audio_file_get_property(
                file_id, capi.get_audio_file_property_maximum_packet_size()
            )
            import struct
            max_packet_size = struct.unpack('<I', max_packet_size_bytes)[0]

            # Allocate buffer for 100 packets
            num_packets = 100
            buffer = bytearray(max_packet_size * num_packets)

            # Read directly into buffer
            bytes_read, packets_read = capi.audio_file_read_packets_into(
                file_id, 0, num_packets, buffer
            )

            assert bytes_read > 0
            assert packets_read > 0
            assert packets_read <= num_packets
            # Verify data is in buffer (non-zero bytes)
            assert any(b != 0 for b in buffer[:bytes_read])
        finally:
            capi.audio_file_close(file_id)

    def test_read_packets_into_matches_original(self, amen_wav_path):
        """Verify zero-copy function produces identical data to original"""
        file_id = capi.audio_file_open_url(amen_wav_path)
        try:
            # Get max packet size
            max_packet_size_bytes = capi.audio_file_get_property(
                file_id, capi.get_audio_file_property_maximum_packet_size()
            )
            import struct
            max_packet_size = struct.unpack('<I', max_packet_size_bytes)[0]

            num_packets = 50

            # Read using original function
            original_data, original_count = capi.audio_file_read_packets(
                file_id, 0, num_packets
            )

            # Read using zero-copy function
            buffer = bytearray(max_packet_size * num_packets)
            bytes_read, packets_read = capi.audio_file_read_packets_into(
                file_id, 0, num_packets, buffer
            )

            # Results should be identical
            assert packets_read == original_count
            assert bytes_read == len(original_data)
            assert bytes(buffer[:bytes_read]) == original_data
        finally:
            capi.audio_file_close(file_id)

    def test_read_packets_into_buffer_too_small(self, amen_wav_path):
        """Test error handling when buffer is too small"""
        file_id = capi.audio_file_open_url(amen_wav_path)
        try:
            # Create a buffer that's too small
            buffer = bytearray(10)  # Way too small for any audio packets

            with pytest.raises(ValueError, match="Buffer too small"):
                capi.audio_file_read_packets_into(file_id, 0, 100, buffer)
        finally:
            capi.audio_file_close(file_id)

    def test_read_packets_into_numpy_array(self, amen_wav_path):
        """Test zero-copy reading into numpy array"""
        pytest.importorskip("numpy")
        import numpy as np

        file_id = capi.audio_file_open_url(amen_wav_path)
        try:
            # Get max packet size
            max_packet_size_bytes = capi.audio_file_get_property(
                file_id, capi.get_audio_file_property_maximum_packet_size()
            )
            import struct
            max_packet_size = struct.unpack('<I', max_packet_size_bytes)[0]

            num_packets = 100

            # Allocate numpy array
            buffer = np.zeros(max_packet_size * num_packets, dtype=np.int8)

            # Read directly into numpy array
            bytes_read, packets_read = capi.audio_file_read_packets_into(
                file_id, 0, num_packets, buffer
            )

            assert bytes_read > 0
            assert packets_read > 0
            # Verify data was written (non-zero values exist)
            assert np.any(buffer[:bytes_read] != 0)
        finally:
            capi.audio_file_close(file_id)

    def test_read_packets_into_sequential(self, amen_wav_path):
        """Test sequential reading with buffer reuse"""
        file_id = capi.audio_file_open_url(amen_wav_path)
        try:
            # Get max packet size
            max_packet_size_bytes = capi.audio_file_get_property(
                file_id, capi.get_audio_file_property_maximum_packet_size()
            )
            import struct
            max_packet_size = struct.unpack('<I', max_packet_size_bytes)[0]

            num_packets = 50
            buffer = bytearray(max_packet_size * num_packets)
            total_bytes = 0
            total_packets = 0
            start_packet = 0

            # Read multiple times, reusing the same buffer
            for _ in range(3):
                bytes_read, packets_read = capi.audio_file_read_packets_into(
                    file_id, start_packet, num_packets, buffer
                )
                if packets_read == 0:
                    break
                total_bytes += bytes_read
                total_packets += packets_read
                start_packet += packets_read

            assert total_packets > 0
            assert total_bytes > 0
        finally:
            capi.audio_file_close(file_id)


class TestAudioConverterConvertBufferInto:
    """Tests for audio_converter_convert_buffer_into() zero-copy function"""

    @pytest.fixture
    def source_format(self):
        """44.1kHz stereo 16-bit PCM format"""
        return {
            'sample_rate': 44100.0,
            'format_id': capi.fourchar_to_int('lpcm'),
            'format_flags': 0x0C,  # kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked
            'bytes_per_packet': 4,
            'frames_per_packet': 1,
            'bytes_per_frame': 4,
            'channels_per_frame': 2,
            'bits_per_channel': 16,
        }

    @pytest.fixture
    def dest_format_mono(self):
        """44.1kHz mono 16-bit PCM format"""
        return {
            'sample_rate': 44100.0,
            'format_id': capi.fourchar_to_int('lpcm'),
            'format_flags': 0x0C,
            'bytes_per_packet': 2,
            'frames_per_packet': 1,
            'bytes_per_frame': 2,
            'channels_per_frame': 1,
            'bits_per_channel': 16,
        }

    def test_convert_buffer_into_basic(self, source_format, dest_format_mono):
        """Test basic zero-copy buffer conversion"""
        converter_id = capi.audio_converter_new(source_format, dest_format_mono)
        try:
            # Create input data (stereo)
            num_frames = 100
            input_data = bytearray(b"\x00\x10\x00\x20" * num_frames)  # stereo samples

            # Allocate output buffer (mono = half the size)
            output_buffer = bytearray(len(input_data) * 2)  # extra space for safety

            # Convert directly into output buffer
            bytes_written = capi.audio_converter_convert_buffer_into(
                converter_id, input_data, output_buffer
            )

            assert bytes_written > 0
            assert bytes_written < len(input_data)  # mono is smaller than stereo
        finally:
            capi.audio_converter_dispose(converter_id)

    def test_convert_buffer_into_matches_original(self, source_format, dest_format_mono):
        """Verify zero-copy function produces identical results to original"""
        converter_id = capi.audio_converter_new(source_format, dest_format_mono)
        try:
            # Create input data
            num_frames = 100
            input_bytes = b"\x00\x01" * (num_frames * 2)

            # Convert using original function
            original_result = capi.audio_converter_convert_buffer(converter_id, input_bytes)

            # Reset converter for fair comparison
            capi.audio_converter_reset(converter_id)

            # Convert using zero-copy function
            input_data = bytearray(input_bytes)
            output_buffer = bytearray(len(input_data) * 4)
            bytes_written = capi.audio_converter_convert_buffer_into(
                converter_id, input_data, output_buffer
            )

            # Results should be identical
            assert bytes_written == len(original_result)
            assert bytes(output_buffer[:bytes_written]) == original_result
        finally:
            capi.audio_converter_dispose(converter_id)

    def test_convert_buffer_into_accepts_bytes(self, source_format, dest_format_mono):
        """Test that zero-copy function accepts bytes input"""
        converter_id = capi.audio_converter_new(source_format, dest_format_mono)
        try:
            # Use bytes (immutable) as input
            input_data = b"\x00\x10\x00\x20" * 50

            output_buffer = bytearray(len(input_data) * 4)
            bytes_written = capi.audio_converter_convert_buffer_into(
                converter_id, input_data, output_buffer
            )

            assert bytes_written > 0
        finally:
            capi.audio_converter_dispose(converter_id)

    def test_convert_buffer_into_numpy_arrays(self, source_format, dest_format_mono):
        """Test zero-copy conversion with numpy arrays"""
        pytest.importorskip("numpy")
        import numpy as np

        converter_id = capi.audio_converter_new(source_format, dest_format_mono)
        try:
            # Create input as numpy array (must be uint8 to match unsigned char)
            num_frames = 100
            input_data = np.zeros(num_frames * 4, dtype=np.uint8)  # stereo 16-bit
            input_data[::2] = 16  # Some non-zero data

            # Output as numpy array (must be uint8)
            output_buffer = np.zeros(num_frames * 4, dtype=np.uint8)

            bytes_written = capi.audio_converter_convert_buffer_into(
                converter_id, input_data, output_buffer
            )

            assert bytes_written > 0
            # Verify output has data
            assert np.any(output_buffer[:bytes_written] != 0)
        finally:
            capi.audio_converter_dispose(converter_id)

    def test_convert_buffer_into_chained(self, source_format, dest_format_mono):
        """Test chained conversions with buffer reuse"""
        converter_id = capi.audio_converter_new(source_format, dest_format_mono)
        try:
            # Pre-allocate reusable buffers
            input_buffer = bytearray(400)
            output_buffer = bytearray(400)

            total_converted = 0

            # Perform multiple conversions reusing buffers
            for i in range(5):
                # Fill input buffer with different data
                for j in range(len(input_buffer)):
                    input_buffer[j] = (i + j) % 256

                capi.audio_converter_reset(converter_id)
                bytes_written = capi.audio_converter_convert_buffer_into(
                    converter_id, input_buffer, output_buffer
                )
                total_converted += bytes_written

            assert total_converted > 0
        finally:
            capi.audio_converter_dispose(converter_id)

    def test_convert_buffer_into_invalid_converter(self):
        """Test error handling for invalid converter ID"""
        input_data = bytearray(b"\x00\x01" * 100)
        output_buffer = bytearray(400)

        with pytest.raises((RuntimeError, cm.CoreAudioError)):
            capi.audio_converter_convert_buffer_into(999999, input_data, output_buffer)


class TestExtendedAudioFileReadInto:
    """Tests for extended_audio_file_read_into() zero-copy function"""

    def test_read_into_basic(self, amen_wav_path):
        """Test basic zero-copy extended audio file reading"""
        ext_file_id = capi.extended_audio_file_open_url(amen_wav_path)
        try:
            num_frames = 1024
            num_channels = 2
            bytes_per_sample = 4  # float32

            # Allocate buffer
            buffer = bytearray(num_frames * num_channels * bytes_per_sample)

            # Read directly into buffer
            bytes_read, frames_read = capi.extended_audio_file_read_into(
                ext_file_id, num_frames, buffer, num_channels
            )

            assert bytes_read > 0
            assert frames_read > 0
            assert frames_read <= num_frames
        finally:
            capi.extended_audio_file_dispose(ext_file_id)

    def test_read_into_matches_original(self, amen_wav_path):
        """Verify zero-copy function produces similar data to original"""
        ext_file_id = capi.extended_audio_file_open_url(amen_wav_path)
        try:
            num_frames = 512
            num_channels = 2
            bytes_per_sample = 4

            # Read using original function
            original_data, original_frames = capi.extended_audio_file_read(
                ext_file_id, num_frames
            )

            # Reopen to reset position
            capi.extended_audio_file_dispose(ext_file_id)
            ext_file_id = capi.extended_audio_file_open_url(amen_wav_path)

            # Read using zero-copy function
            buffer = bytearray(num_frames * num_channels * bytes_per_sample)
            bytes_read, frames_read = capi.extended_audio_file_read_into(
                ext_file_id, num_frames, buffer, num_channels
            )

            # Frame counts should match
            assert frames_read == original_frames
        finally:
            capi.extended_audio_file_dispose(ext_file_id)

    def test_read_into_buffer_too_small(self, amen_wav_path):
        """Test error handling when buffer is too small"""
        ext_file_id = capi.extended_audio_file_open_url(amen_wav_path)
        try:
            # Create a buffer that's too small
            buffer = bytearray(10)

            with pytest.raises(ValueError, match="Buffer too small"):
                capi.extended_audio_file_read_into(ext_file_id, 1024, buffer)
        finally:
            capi.extended_audio_file_dispose(ext_file_id)

    def test_read_into_sequential(self, amen_wav_path):
        """Test sequential reading with buffer reuse"""
        ext_file_id = capi.extended_audio_file_open_url(amen_wav_path)
        try:
            num_frames = 512
            num_channels = 2
            buffer = bytearray(num_frames * num_channels * 4)
            total_frames = 0

            # Read multiple times, reusing the same buffer
            for _ in range(5):
                bytes_read, frames_read = capi.extended_audio_file_read_into(
                    ext_file_id, num_frames, buffer, num_channels
                )
                if frames_read == 0:
                    break
                total_frames += frames_read

            assert total_frames > 0
        finally:
            capi.extended_audio_file_dispose(ext_file_id)


class TestAudioUnitRenderInto:
    """Tests for audio_unit_render_into() zero-copy function"""

    @pytest.fixture
    def default_output_unit(self):
        """Create a default output AudioUnit"""
        desc = {
            'component_type': capi.get_audio_unit_type_output(),
            'component_sub_type': capi.get_audio_unit_subtype_default_output(),
            'component_manufacturer': capi.get_audio_unit_manufacturer_apple(),
            'component_flags': 0,
            'component_flags_mask': 0
        }
        component = capi.audio_component_find_next(desc)
        if component is None:
            pytest.skip("No default output AudioUnit available")
        unit_id = capi.audio_component_instance_new(component)
        try:
            capi.audio_unit_initialize(unit_id)
        except RuntimeError:
            capi.audio_component_instance_dispose(unit_id)
            pytest.skip("AudioUnit initialization failed (may require audio device)")
        yield unit_id
        capi.audio_unit_uninitialize(unit_id)
        capi.audio_component_instance_dispose(unit_id)

    def test_render_into_basic(self, default_output_unit):
        """Test basic zero-copy AudioUnit rendering"""
        num_frames = 512
        num_channels = 2
        buffer_size = num_frames * num_channels * 4  # float32

        # Create input and output buffers
        input_data = bytearray(buffer_size)
        output_buffer = bytearray(buffer_size)

        # Fill input with some data
        for i in range(0, len(input_data), 4):
            input_data[i:i+4] = b'\x00\x00\x80\x3f'  # 1.0 as float32

        bytes_written = capi.audio_unit_render_into(
            default_output_unit, input_data, output_buffer, num_frames, num_channels
        )

        assert bytes_written == buffer_size

    def test_render_into_buffer_validation(self, default_output_unit):
        """Test buffer size validation"""
        num_frames = 512
        num_channels = 2

        # Input buffer too small
        input_data = bytearray(10)
        output_buffer = bytearray(num_frames * num_channels * 4)

        with pytest.raises(ValueError, match="Input buffer too small"):
            capi.audio_unit_render_into(
                default_output_unit, input_data, output_buffer, num_frames, num_channels
            )

        # Output buffer too small
        input_data = bytearray(num_frames * num_channels * 4)
        output_buffer = bytearray(10)

        with pytest.raises(ValueError, match="Output buffer too small"):
            capi.audio_unit_render_into(
                default_output_unit, input_data, output_buffer, num_frames, num_channels
            )


class TestAudioConverterFillComplexBufferInto:
    """Tests for audio_converter_fill_complex_buffer_into() zero-copy function"""

    @pytest.fixture
    def source_format_44100(self):
        """44.1kHz stereo 16-bit PCM format"""
        return {
            'sample_rate': 44100.0,
            'format_id': capi.fourchar_to_int('lpcm'),
            'format_flags': 0x0C,
            'bytes_per_packet': 4,
            'frames_per_packet': 1,
            'bytes_per_frame': 4,
            'channels_per_frame': 2,
            'bits_per_channel': 16,
        }

    @pytest.fixture
    def dest_format_48000(self):
        """48kHz stereo 16-bit PCM format"""
        return {
            'sample_rate': 48000.0,
            'format_id': capi.fourchar_to_int('lpcm'),
            'format_flags': 0x0C,
            'bytes_per_packet': 4,
            'frames_per_packet': 1,
            'bytes_per_frame': 4,
            'channels_per_frame': 2,
            'bits_per_channel': 16,
        }

    def test_fill_complex_buffer_into_basic(self, source_format_44100, dest_format_48000):
        """Test basic zero-copy complex buffer conversion"""
        converter_id = capi.audio_converter_new(source_format_44100, dest_format_48000)
        try:
            # Create input data (100 packets)
            num_packets = 100
            input_data = bytearray(b"\x00\x10\x00\x20" * num_packets)

            # Output buffer with room for upsampling
            output_buffer = bytearray(len(input_data) * 4)

            # Expected output packets (approximately 100 * 48000/44100)
            expected_output_packets = int(num_packets * 48000 / 44100) + 10

            bytes_written, packets_out = capi.audio_converter_fill_complex_buffer_into(
                converter_id, input_data, output_buffer,
                num_packets, expected_output_packets, source_format_44100
            )

            assert bytes_written > 0
            assert packets_out > 0
        finally:
            capi.audio_converter_dispose(converter_id)

    def test_fill_complex_buffer_into_matches_original(self, source_format_44100, dest_format_48000):
        """Verify zero-copy function produces identical results to original"""
        converter_id = capi.audio_converter_new(source_format_44100, dest_format_48000)
        try:
            num_packets = 50
            input_bytes = b"\x00\x01\x02\x03" * num_packets
            expected_output_packets = int(num_packets * 48000 / 44100) + 10

            # Convert using original function
            original_result, original_packets = capi.audio_converter_fill_complex_buffer(
                converter_id, input_bytes,
                num_packets, expected_output_packets, source_format_44100
            )

            # Reset converter
            capi.audio_converter_reset(converter_id)

            # Convert using zero-copy function
            input_data = bytearray(input_bytes)
            output_buffer = bytearray(len(input_data) * 4)
            bytes_written, packets_out = capi.audio_converter_fill_complex_buffer_into(
                converter_id, input_data, output_buffer,
                num_packets, expected_output_packets, source_format_44100
            )

            # Results should be identical
            assert packets_out == original_packets
            assert bytes_written == len(original_result)
            assert bytes(output_buffer[:bytes_written]) == original_result
        finally:
            capi.audio_converter_dispose(converter_id)

    def test_fill_complex_buffer_into_numpy(self, source_format_44100, dest_format_48000):
        """Test zero-copy complex buffer conversion with numpy arrays"""
        pytest.importorskip("numpy")
        import numpy as np

        converter_id = capi.audio_converter_new(source_format_44100, dest_format_48000)
        try:
            num_packets = 100
            input_data = np.zeros(num_packets * 4, dtype=np.uint8)
            input_data[::2] = 16

            output_buffer = np.zeros(num_packets * 16, dtype=np.uint8)
            expected_output_packets = int(num_packets * 48000 / 44100) + 10

            bytes_written, packets_out = capi.audio_converter_fill_complex_buffer_into(
                converter_id, input_data, output_buffer,
                num_packets, expected_output_packets, source_format_44100
            )

            assert bytes_written > 0
            assert packets_out > 0
        finally:
            capi.audio_converter_dispose(converter_id)


class TestAudioFileStreamParseBuffer:
    """Tests for audio_file_stream_parse_buffer() zero-copy function"""

    @pytest.fixture
    def wav_file_data(self, amen_wav_path):
        """Read raw WAV file bytes (not decoded PCM)"""
        with open(amen_wav_path, "rb") as f:
            return f.read()

    def test_parse_buffer_basic(self, wav_file_data):
        """Test basic zero-copy stream parsing"""
        # Open a stream parser with WAV file type hint
        stream_id = capi.audio_file_stream_open(capi.get_audio_file_wave_type())
        try:
            # Parse header using bytearray (zero-copy)
            buffer = bytearray(wav_file_data[:1024])
            status = capi.audio_file_stream_parse_buffer(stream_id, buffer)
            assert status == 0
        finally:
            capi.audio_file_stream_close(stream_id)

    def test_parse_buffer_matches_original(self, wav_file_data):
        """Verify zero-copy function produces same result as original"""
        header_chunk = wav_file_data[:1024]

        # Parse with original function
        stream1 = capi.audio_file_stream_open(capi.get_audio_file_wave_type())
        try:
            status1 = capi.audio_file_stream_parse_bytes(stream1, header_chunk)
        finally:
            capi.audio_file_stream_close(stream1)

        # Parse with zero-copy function
        stream2 = capi.audio_file_stream_open(capi.get_audio_file_wave_type())
        try:
            buffer = bytearray(header_chunk)
            status2 = capi.audio_file_stream_parse_buffer(stream2, buffer)
        finally:
            capi.audio_file_stream_close(stream2)

        assert status1 == status2

    def test_parse_buffer_accepts_bytes(self, wav_file_data):
        """Test that zero-copy function accepts bytes input"""
        stream_id = capi.audio_file_stream_open(capi.get_audio_file_wave_type())
        try:
            # Pass bytes directly (immutable)
            status = capi.audio_file_stream_parse_buffer(stream_id, wav_file_data[:1024])
            assert status == 0
        finally:
            capi.audio_file_stream_close(stream_id)

    def test_parse_buffer_chunked(self, wav_file_data):
        """Test chunked parsing with buffer reuse"""
        stream_id = capi.audio_file_stream_open(capi.get_audio_file_wave_type())
        try:
            # Parse in chunks, reusing a buffer
            chunk_size = 512
            buffer = bytearray(chunk_size)

            # Parse first 4KB in chunks
            for i in range(0, min(len(wav_file_data), 4096), chunk_size):
                chunk = wav_file_data[i:i+chunk_size]
                if not chunk:
                    break
                # Copy chunk into reusable buffer
                buffer[:len(chunk)] = chunk
                status = capi.audio_file_stream_parse_buffer(
                    stream_id, memoryview(buffer)[:len(chunk)]
                )
                assert status == 0
        finally:
            capi.audio_file_stream_close(stream_id)

    def test_parse_buffer_numpy(self, wav_file_data):
        """Test zero-copy parsing with numpy array"""
        pytest.importorskip("numpy")
        import numpy as np

        stream_id = capi.audio_file_stream_open(capi.get_audio_file_wave_type())
        try:
            # Convert to numpy array
            np_buffer = np.frombuffer(wav_file_data[:1024], dtype=np.uint8).copy()
            status = capi.audio_file_stream_parse_buffer(stream_id, np_buffer)
            assert status == 0
        finally:
            capi.audio_file_stream_close(stream_id)


class TestMemoryviewPerformance:
    """Performance comparison tests (informational, not strict assertions)"""

    def test_read_packets_performance_comparison(self, amen_wav_path):
        """Compare performance of original vs zero-copy read"""
        import time

        file_id = capi.audio_file_open_url(amen_wav_path)
        try:
            # Get max packet size
            max_packet_size_bytes = capi.audio_file_get_property(
                file_id, capi.get_audio_file_property_maximum_packet_size()
            )
            import struct
            max_packet_size = struct.unpack('<I', max_packet_size_bytes)[0]

            num_packets = 1000
            iterations = 10

            # Time original function
            start = time.perf_counter()
            for _ in range(iterations):
                data, count = capi.audio_file_read_packets(file_id, 0, num_packets)
            original_time = time.perf_counter() - start

            # Time zero-copy function (pre-allocate buffer once)
            buffer = bytearray(max_packet_size * num_packets)
            start = time.perf_counter()
            for _ in range(iterations):
                bytes_read, count = capi.audio_file_read_packets_into(
                    file_id, 0, num_packets, buffer
                )
            zerocopy_time = time.perf_counter() - start

            # Just ensure both work - performance may vary
            assert original_time > 0
            assert zerocopy_time > 0
            # Log for informational purposes (not a strict assertion)
            print(f"\nRead packets ({iterations} iterations, {num_packets} packets each):")
            print(f"  Original: {original_time*1000:.2f}ms")
            print(f"  Zero-copy: {zerocopy_time*1000:.2f}ms")
            if zerocopy_time < original_time:
                print(f"  Speedup: {original_time/zerocopy_time:.2f}x")
        finally:
            capi.audio_file_close(file_id)
