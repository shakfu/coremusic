"""CoreMusic: Python bindings for Apple CoreAudio, CoreMIDI, and Ableton Link.

Import from domain subpackages::

    from coremusic.audio import AudioFile, AudioFormat, AudioUnit, AUGraph
    from coremusic.midi import MIDIClient, MusicPlayer, MusicSequence
    from coremusic.exceptions import CoreAudioError, AudioFileError
    from coremusic.base import CoreAudioObject, AudioPlayer
    from coremusic.constants import AudioFileProperty, AudioFormatID
    from coremusic.shortcuts import play, convert
"""

__version__ = "0.2.1"
