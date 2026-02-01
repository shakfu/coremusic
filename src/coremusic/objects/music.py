"""Music player classes for coremusic.

This module provides classes for MIDI sequencing and playback:
- MusicTrack: Track for sequencing MIDI events
- MusicSequence: Sequence for MIDI composition
- MusicPlayer: Player for playing back sequences
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

from .. import capi
from .exceptions import MusicPlayerError

__all__ = [
    "MusicTrack",
    "MusicSequence",
    "MusicPlayer",
]


class MusicTrack(capi.CoreAudioObject):
    """Music track for sequencing MIDI events

    Represents a single track in a music sequence. Tracks contain
    MIDI events (notes, control changes, etc.) that can be played back.

    Note:
        MusicTrack objects are created by MusicSequence and should not
        be instantiated directly.
    """

    def __init__(self, track_id: int, parent_sequence: "MusicSequence"):
        """Initialize a music track

        Args:
            track_id: CoreAudio track identifier
            parent_sequence: Parent MusicSequence object
        """
        super().__init__()
        self._set_object_id(track_id)
        self._parent_sequence = parent_sequence

    def add_midi_note(
        self,
        time: float,
        channel: int,
        note: int,
        velocity: int,
        release_velocity: int = 64,
        duration: float = 1.0
    ) -> None:
        """Add a MIDI note event to the track

        Args:
            time: Time position in beats (must be non-negative)
            channel: MIDI channel (0-15)
            note: MIDI note number (0-127)
            velocity: Note-on velocity (1-127)
            release_velocity: Note-off velocity (0-127)
            duration: Note duration in beats (must be positive)

        Raises:
            ValueError: If parameters are out of valid MIDI range
            MusicPlayerError: If adding note fails

        Example::

            # Add middle C with velocity 100 for 1 beat
            track.add_midi_note(0.0, 0, 60, 100, 64, 1.0)
        """
        if time < 0:
            raise ValueError(f"time must be non-negative, got {time}")
        if not 0 <= channel <= 15:
            raise ValueError(f"channel must be 0-15, got {channel}")
        if not 0 <= note <= 127:
            raise ValueError(f"note must be 0-127, got {note}")
        if not 1 <= velocity <= 127:
            raise ValueError(f"velocity must be 1-127, got {velocity}")
        if not 0 <= release_velocity <= 127:
            raise ValueError(f"release_velocity must be 0-127, got {release_velocity}")
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")

        self._ensure_not_disposed()
        try:
            capi.music_track_new_midi_note_event(  # type: ignore[call-arg]
                self.object_id, time, channel, note, velocity, release_velocity, duration  # type: ignore[arg-type]
            )
        except Exception as e:
            raise MusicPlayerError(f"Failed to add MIDI note: {e}")

    def add_midi_channel_event(
        self,
        time: float,
        status: int,
        data1: int,
        data2: int = 0
    ) -> None:
        """Add a MIDI channel event to the track

        Args:
            time: Time position in beats (must be non-negative)
            status: MIDI status byte (e.g., 0xB0 for CC on channel 0, 0x80-0xEF)
            data1: First data byte (0-127)
            data2: Second data byte (0-127)

        Raises:
            ValueError: If parameters are out of valid MIDI range
            MusicPlayerError: If adding event fails

        Example::

            # Add program change to piano on channel 0
            track.add_midi_channel_event(0.0, 0xC0, 0)

            # Add volume control change
            track.add_midi_channel_event(0.0, 0xB0, 7, 100)
        """
        if time < 0:
            raise ValueError(f"time must be non-negative, got {time}")
        if not 0x80 <= status <= 0xEF:
            raise ValueError(f"status must be 0x80-0xEF (channel message), got {hex(status)}")
        if not 0 <= data1 <= 127:
            raise ValueError(f"data1 must be 0-127, got {data1}")
        if not 0 <= data2 <= 127:
            raise ValueError(f"data2 must be 0-127, got {data2}")

        self._ensure_not_disposed()
        try:
            capi.music_track_new_midi_channel_event(  # type: ignore[call-arg]
                self.object_id, time, status, data1, data2  # type: ignore[arg-type]
            )
        except Exception as e:
            raise MusicPlayerError(f"Failed to add MIDI channel event: {e}")

    def add_tempo_event(self, time: float, bpm: float) -> None:
        """Add a tempo change event to the track

        Args:
            time: Time position in beats (must be non-negative)
            bpm: Tempo in beats per minute (must be positive, typically 20-999)

        Raises:
            ValueError: If time is negative or bpm is not positive
            MusicPlayerError: If adding tempo event fails

        Note:
            Tempo events should typically be added to the tempo track,
            obtained via MusicSequence.tempo_track property.

        Example::

            # Set tempo to 120 BPM at the start
            tempo_track.add_tempo_event(0.0, 120.0)

            # Speed up to 140 BPM at beat 32
            tempo_track.add_tempo_event(32.0, 140.0)
        """
        if time < 0:
            raise ValueError(f"time must be non-negative, got {time}")
        if bpm <= 0:
            raise ValueError(f"bpm must be positive, got {bpm}")

        self._ensure_not_disposed()
        try:
            capi.music_track_new_extended_tempo_event(self.object_id, time, bpm)
        except Exception as e:
            raise MusicPlayerError(f"Failed to add tempo event: {e}")

    def __repr__(self) -> str:
        return f"MusicTrack(id={self.object_id})"


class MusicSequence(capi.CoreAudioObject):
    """Music sequence for MIDI composition and playback

    A MusicSequence contains multiple tracks of MIDI events and can be
    played back through a MusicPlayer. Supports saving/loading MIDI files.
    """

    def __init__(self) -> None:
        """Create a new music sequence

        Raises:
            MusicPlayerError: If sequence creation fails
        """
        super().__init__()
        try:
            sequence_id = capi.new_music_sequence()
            self._set_object_id(sequence_id)
            self._tracks: List[MusicTrack] = []
            self._tempo_track: Optional[MusicTrack] = None
        except Exception as e:
            raise MusicPlayerError(f"Failed to create music sequence: {e}")

    def new_track(self) -> MusicTrack:
        """Create a new track in the sequence

        Returns:
            New MusicTrack object

        Raises:
            MusicPlayerError: If track creation fails

        Example::

            sequence = cm.MusicSequence()
            track = sequence.new_track()
            track.add_midi_note(0.0, 0, 60, 100)
        """
        self._ensure_not_disposed()
        try:
            track_id = capi.music_sequence_new_track(self.object_id)
            track = MusicTrack(track_id, self)
            self._tracks.append(track)
            return track
        except Exception as e:
            raise MusicPlayerError(f"Failed to create track: {e}")

    def dispose_track(self, track: MusicTrack) -> None:
        """Remove a track from the sequence

        Args:
            track: Track to remove

        Raises:
            MusicPlayerError: If track removal fails
        """
        self._ensure_not_disposed()
        try:
            capi.music_sequence_dispose_track(self.object_id, track.object_id)
            if track in self._tracks:
                self._tracks.remove(track)
            track.dispose()
        except Exception as e:
            raise MusicPlayerError(f"Failed to dispose track: {e}")

    def get_track(self, index: int) -> MusicTrack:
        """Get track at specified index

        Args:
            index: Track index (0-based, must be non-negative)

        Returns:
            MusicTrack at the specified index

        Raises:
            ValueError: If index is negative
            IndexError: If index >= track_count
            MusicPlayerError: If getting track fails

        Example::

            # Iterate through all tracks in a sequence
            sequence = cm.MusicSequence()
            sequence.load_from_file("song.mid")

            for i in range(sequence.track_count):
                track = sequence.get_track(i)
                print(f"Track {i}: {track}")
        """
        if index < 0:
            raise ValueError(f"index must be non-negative, got {index}")

        self._ensure_not_disposed()

        # Check bounds
        count = self.track_count
        if index >= count:
            if count == 0:
                raise IndexError(f"track index {index} out of range (sequence has no tracks)")
            raise IndexError(f"track index {index} out of range (0-{count-1})")

        try:
            track_id = capi.music_sequence_get_ind_track(self.object_id, index)
            # Check if we already have this track in our cache
            for track in self._tracks:
                if track.object_id == track_id:
                    return track
            # Create new wrapper for existing track
            track = MusicTrack(track_id, self)
            self._tracks.append(track)
            return track
        except Exception as e:
            raise MusicPlayerError(f"Failed to get track at index {index}: {e}")

    @property
    def track_count(self) -> int:
        """Get number of tracks in sequence

        Returns:
            Number of tracks
        """
        self._ensure_not_disposed()
        try:
            return capi.music_sequence_get_track_count(self.object_id)
        except Exception as e:
            raise MusicPlayerError(f"Failed to get track count: {e}")

    @property
    def tempo_track(self) -> MusicTrack:
        """Get the tempo track for this sequence

        The tempo track contains tempo change events that affect playback.

        Returns:
            Tempo MusicTrack

        Raises:
            MusicPlayerError: If getting tempo track fails
        """
        self._ensure_not_disposed()
        if self._tempo_track is None:
            try:
                tempo_track_id = capi.music_sequence_get_tempo_track(self.object_id)
                self._tempo_track = MusicTrack(tempo_track_id, self)
            except Exception as e:
                raise MusicPlayerError(f"Failed to get tempo track: {e}")
        return self._tempo_track

    @property
    def sequence_type(self) -> int:
        """Get the sequence type (beats, seconds, or samples)

        Returns:
            Sequence type constant
        """
        self._ensure_not_disposed()
        try:
            return capi.music_sequence_get_sequence_type(self.object_id)
        except Exception as e:
            raise MusicPlayerError(f"Failed to get sequence type: {e}")

    @sequence_type.setter
    def sequence_type(self, seq_type: int) -> None:
        """Set the sequence type

        Args:
            seq_type: Sequence type (use get_music_sequence_type_* constants)

        Raises:
            MusicPlayerError: If setting sequence type fails
        """
        self._ensure_not_disposed()
        try:
            capi.music_sequence_set_sequence_type(self.object_id, seq_type)
        except Exception as e:
            raise MusicPlayerError(f"Failed to set sequence type: {e}")

    def load_from_file(self, file_path: Union[str, Path]) -> None:
        """Load sequence from a MIDI file

        Args:
            file_path: Path to MIDI file

        Raises:
            MusicPlayerError: If loading fails

        Example::

            sequence = cm.MusicSequence()
            sequence.load_from_file("song.mid")
            print(f"Loaded {sequence.track_count} tracks")
        """
        self._ensure_not_disposed()
        try:
            capi.music_sequence_file_load(self.object_id, str(file_path))  # type: ignore[call-arg]
            # Clear track cache since file load may change tracks
            self._tracks = []
            self._tempo_track = None
        except Exception as e:
            raise MusicPlayerError(f"Failed to load from file {file_path}: {e}")

    def dispose(self) -> None:
        """Dispose the sequence and all its tracks"""
        if not self.is_disposed:
            try:
                # Dispose all cached tracks first
                for track in self._tracks:
                    try:
                        track.dispose()
                    except Exception:
                        pass
                if self._tempo_track is not None:
                    try:
                        self._tempo_track.dispose()
                    except Exception:
                        pass

                capi.dispose_music_sequence(self.object_id)
            except Exception:
                pass  # Best effort cleanup
            finally:
                self._tracks = []
                self._tempo_track = None
                super().dispose()

    def __repr__(self) -> str:
        if self.is_disposed:
            return "MusicSequence(disposed)"
        try:
            count = self.track_count
            return f"MusicSequence(tracks={count})"
        except Exception:
            return "MusicSequence()"


class MusicPlayer(capi.CoreAudioObject):
    """Music player for playing back MIDI sequences

    MusicPlayer provides playback control for MusicSequence objects,
    including start/stop, tempo control, and time position.
    """

    def __init__(self) -> None:
        """Create a new music player

        Raises:
            MusicPlayerError: If player creation fails

        Example::

            player = cm.MusicPlayer()
            sequence = cm.MusicSequence()
            # ... add tracks and events to sequence ...
            player.sequence = sequence
            player.preroll()
            player.start()
        """
        super().__init__()
        try:
            player_id = capi.new_music_player()
            self._set_object_id(player_id)
            self._sequence: Optional[MusicSequence] = None
        except Exception as e:
            raise MusicPlayerError(f"Failed to create music player: {e}")

    @property
    def sequence(self) -> Optional[MusicSequence]:
        """Get the currently assigned sequence

        Returns:
            Current MusicSequence or None
        """
        return self._sequence

    @sequence.setter
    def sequence(self, sequence: Optional[MusicSequence]) -> None:
        """Set the sequence to play

        Args:
            sequence: MusicSequence to assign, or None to clear

        Raises:
            MusicPlayerError: If setting sequence fails
        """
        self._ensure_not_disposed()
        try:
            if sequence is None:
                capi.music_player_set_sequence(self.object_id, 0)
                self._sequence = None
            else:
                capi.music_player_set_sequence(self.object_id, sequence.object_id)
                self._sequence = sequence
        except Exception as e:
            raise MusicPlayerError(f"Failed to set sequence: {e}")

    @property
    def time(self) -> float:
        """Get current playback time in beats

        Returns:
            Current time position
        """
        self._ensure_not_disposed()
        try:
            return capi.music_player_get_time(self.object_id)
        except Exception as e:
            raise MusicPlayerError(f"Failed to get time: {e}")

    @time.setter
    def time(self, time: float) -> None:
        """Set playback time position

        Args:
            time: Time position in beats (must be non-negative)

        Raises:
            ValueError: If time is negative
            MusicPlayerError: If setting time fails
        """
        if time < 0:
            raise ValueError(f"time must be non-negative, got {time}")

        self._ensure_not_disposed()
        try:
            capi.music_player_set_time(self.object_id, time)
        except Exception as e:
            raise MusicPlayerError(f"Failed to set time: {e}")

    @property
    def play_rate(self) -> float:
        """Get playback rate scalar

        Returns:
            Play rate (1.0 = normal speed, 2.0 = double speed, etc.)
        """
        self._ensure_not_disposed()
        try:
            return capi.music_player_get_play_rate_scalar(self.object_id)
        except Exception as e:
            raise MusicPlayerError(f"Failed to get play rate: {e}")

    @play_rate.setter
    def play_rate(self, rate: float) -> None:
        """Set playback rate scalar

        Args:
            rate: Play rate scalar (must be > 0, typically 0.1 to 10.0)

        Raises:
            MusicPlayerError: If setting play rate fails
            ValueError: If rate is invalid
        """
        self._ensure_not_disposed()
        if rate <= 0:
            raise ValueError(f"Play rate must be positive, got {rate}")
        try:
            capi.music_player_set_play_rate_scalar(self.object_id, rate)
        except Exception as e:
            raise MusicPlayerError(f"Failed to set play rate: {e}")

    @property
    def is_playing(self) -> bool:
        """Check if player is currently playing

        Returns:
            True if playing, False otherwise
        """
        self._ensure_not_disposed()
        try:
            return capi.music_player_is_playing(self.object_id)
        except Exception as e:
            raise MusicPlayerError(f"Failed to check playing state: {e}")

    def preroll(self) -> None:
        """Prepare player for playback

        Should be called before start() to ensure smooth playback start.

        Raises:
            MusicPlayerError: If preroll fails
        """
        self._ensure_not_disposed()
        try:
            capi.music_player_preroll(self.object_id)
        except Exception as e:
            raise MusicPlayerError(f"Failed to preroll: {e}")

    def start(self) -> None:
        """Start playback

        Raises:
            MusicPlayerError: If starting playback fails

        Example::

            player.preroll()
            player.start()
            # ... wait for playback ...
            player.stop()
        """
        self._ensure_not_disposed()
        try:
            capi.music_player_start(self.object_id)
        except Exception as e:
            raise MusicPlayerError(f"Failed to start playback: {e}")

    def stop(self) -> None:
        """Stop playback

        Raises:
            MusicPlayerError: If stopping playback fails
        """
        self._ensure_not_disposed()
        try:
            capi.music_player_stop(self.object_id)
        except Exception as e:
            raise MusicPlayerError(f"Failed to stop playback: {e}")

    def dispose(self) -> None:
        """Dispose the player and free resources"""
        if not self.is_disposed:
            try:
                # Stop playback first
                if self.is_playing:
                    self.stop()
                # Clear sequence reference
                if self._sequence is not None:
                    try:
                        capi.music_player_set_sequence(self.object_id, 0)
                    except Exception:
                        pass

                capi.dispose_music_player(self.object_id)
            except Exception:
                pass  # Best effort cleanup
            finally:
                self._sequence = None
                super().dispose()

    def __enter__(self) -> "MusicPlayer":
        """Enter context manager"""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and dispose"""
        self.dispose()

    def __repr__(self) -> str:
        if self.is_disposed:
            return "MusicPlayer(disposed)"
        try:
            status = "playing" if self.is_playing else "stopped"
            time = self.time
            rate = self.play_rate
            return f"MusicPlayer({status}, time={time:.2f}, rate={rate:.2f})"
        except Exception:
            return "MusicPlayer()"
