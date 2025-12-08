from conftest import AMEN_WAV_PATH
import coremusic.capi as capi
import coremusic as cm


def test_format_data_parsing():

    # Open an audio file
    audio_file = capi.audio_file_open_url(AMEN_WAV_PATH)

    # Get file format information
    format_data = capi.audio_file_get_property(
        audio_file, capi.get_audio_file_property_data_format()
    )

    # use parse_audio_stream_basic_description() to parse the binary format data
    assert cm.parse_audio_stream_basic_description(format_data) == {
        'sample_rate': 44100.0, 
        'format_id': 'lpcm', 
        'format_flags': 12, 
        'bytes_per_packet': 4, 
        'frames_per_packet': 1, 
        'bytes_per_frame': 4, 
        'channels_per_frame': 2, 
        'bits_per_channel': 16, 
        'reserved': 0
    }

    # Close the file (manual cleanup required with the functional capi)
    capi.audio_file_close(audio_file)
