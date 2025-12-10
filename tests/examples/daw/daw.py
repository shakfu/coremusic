#!/usr/bin/env python3
"""CoreMusic DAW (Digital Audio Workstation) Essentials Module.

This module provides higher-level abstractions for building DAW-like applications:
- Multi-track audio/MIDI timeline
- Transport control (play/pause/stop/record)
- Session management
- Automation
- Clip management
- Plugin integration

Example:
    >>> import coremusic as cm
    >>> timeline = cm.Timeline(sample_rate=48000, tempo=128.0)
    >>> drums = timeline.add_track("Drums", "audio")
    >>> drums.add_clip(cm.Clip("drums.wav"), start_time=0.0)
    >>> timeline.play()
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ============================================================================
# AudioUnit Plugin Wrapper
# ============================================================================


class AudioUnitPlugin:
    """Wrapper for an AudioUnit plugin (instrument or effect).

    Example:
        >>> plugin = AudioUnitPlugin("DLSMusicDevice", plugin_type="instrument")
        >>> plugin.send_midi(60, 100)  # Play middle C
        >>> plugin.process_audio(audio_buffer, sample_rate)
    """

    def __init__(self, name: str, plugin_type: str = "effect",
                 manufacturer: str = "appl"):
        """Initialize AudioUnit plugin.

        Args:
            name: Plugin name or subtype (e.g., "DLSMusicDevice", "AUDelay")
            plugin_type: 'instrument' or 'effect'
            manufacturer: Manufacturer code (default 'appl' for Apple)
        """
        self.name = name
        self.plugin_type = plugin_type
        self.manufacturer = manufacturer
        self.unit_id: Optional[int] = None
        self._initialized = False
        self._sample_rate = 44100.0

    def initialize(self, sample_rate: float = 44100.0) -> None:
        """Initialize the AudioUnit."""
        if self._initialized:
            return

        try:
            import struct

            from coremusic.capi import (
                audio_component_find_next, audio_component_instance_new,
                audio_unit_initialize, audio_unit_set_property,
                fourchar_to_int, get_audio_unit_property_stream_format,
                get_audio_unit_scope_input, get_audio_unit_scope_output,
                get_linear_pcm_format_flag_is_packed,
                get_linear_pcm_format_flag_is_signed_integer)

            # Determine component type
            if self.plugin_type == "instrument":
                comp_type = fourchar_to_int('aumu')  # kAudioUnitType_MusicDevice
            else:
                comp_type = fourchar_to_int('aufx')  # kAudioUnitType_Effect

            # Find component
            desc = {
                'type': comp_type,
                'subtype': fourchar_to_int(self.name) if len(self.name) == 4 else 0,
                'manufacturer': fourchar_to_int(self.manufacturer),
                'flags': 0,
                'flags_mask': 0
            }

            component_id = audio_component_find_next(desc)
            if component_id is None:
                raise RuntimeError(f"AudioUnit '{self.name}' not found")

            # Create instance
            self.unit_id = audio_component_instance_new(component_id)

            # Set up stream format
            self._sample_rate = sample_rate
            asbd_data = struct.pack(
                '<dIIIIIII',
                sample_rate,  # mSampleRate
                fourchar_to_int('lpcm'),  # mFormatID
                get_linear_pcm_format_flag_is_signed_integer() | get_linear_pcm_format_flag_is_packed(),
                4,  # mBytesPerPacket (stereo 16-bit)
                1,  # mFramesPerPacket
                4,  # mBytesPerFrame
                2,  # mChannelsPerFrame
                16  # mBitsPerChannel
            )

            # Set format on input and output
            audio_unit_set_property(
                self.unit_id,
                get_audio_unit_property_stream_format(),
                get_audio_unit_scope_output(),
                0,
                asbd_data
            )

            audio_unit_set_property(
                self.unit_id,
                get_audio_unit_property_stream_format(),
                get_audio_unit_scope_input(),
                0,
                asbd_data
            )

            # Initialize unit
            audio_unit_initialize(self.unit_id)
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize AudioUnit '{self.name}': {e}")
            raise

    def send_midi(self, note: int, velocity: int, note_on: bool = True,
                  channel: int = 0) -> None:
        """Send MIDI note to instrument plugin.

        Args:
            note: MIDI note number (0-127)
            velocity: Velocity (0-127)
            note_on: True for note on, False for note off
            channel: MIDI channel (0-15)
        """
        if not self._initialized or self.unit_id is None:
            return

        try:
            import struct

            from coremusic.capi import audio_unit_set_property

            # Create MIDI packet (status, note, velocity)
            status = (0x90 if note_on else 0x80) | (channel & 0x0F)
            midi_data = struct.pack('BBB', status, note & 0x7F, velocity & 0x7F)

            # Send via kAudioUnitProperty_MIDIEvent (property ID 0x1012)
            audio_unit_set_property(
                self.unit_id,
                0x1012,  # kAudioUnitProperty_MIDIEvent
                0,  # global scope
                0,
                midi_data
            )
        except Exception as e:
            logger.warning(f"Failed to send MIDI: {e}")

    def process_audio(self, audio_data: bytes, num_frames: int) -> bytes:
        """Process audio through the plugin.

        Args:
            audio_data: Input audio data
            num_frames: Number of frames to process

        Returns:
            Processed audio data
        """
        if not self._initialized or self.unit_id is None:
            return audio_data

        try:
            from coremusic.capi import audio_unit_render

            output = audio_unit_render(
                self.unit_id,
                audio_data,
                num_frames,
                self._sample_rate,
                2  # stereo
            )
            return output
        except Exception as e:
            logger.warning(f"Audio processing failed: {e}")
            return audio_data

    def dispose(self) -> None:
        """Clean up the AudioUnit."""
        if self._initialized and self.unit_id is not None:
            try:
                from coremusic.capi import (audio_component_instance_dispose,
                                            audio_unit_uninitialize)
                audio_unit_uninitialize(self.unit_id)
                audio_component_instance_dispose(self.unit_id)
            except Exception as e:
                logger.warning(f"Failed to dispose AudioUnit: {e}")
            finally:
                self._initialized = False
                self.unit_id = None

    def __del__(self):
        """Cleanup on deletion."""
        self.dispose()


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class TimelineMarker:
    """Represents a marker/cue point in timeline.

    Attributes:
        position: Position in seconds
        name: Marker name
        color: Optional color (e.g., "#FF0000")
    """

    position: float
    name: str
    color: Optional[str] = None

    def __repr__(self) -> str:
        color_str = f", color={self.color}" if self.color else ""
        return f"TimelineMarker(position={self.position:.2f}s, name='{self.name}'{color_str})"


@dataclass
class TimeRange:
    """Represents a time range (e.g., loop region).

    Attributes:
        start: Start time in seconds
        end: End time in seconds
    """

    start: float
    end: float

    @property
    def duration(self) -> float:
        """Duration of the time range in seconds."""
        return self.end - self.start

    def __repr__(self) -> str:
        return f"TimeRange({self.start:.2f}s - {self.end:.2f}s, duration={self.duration:.2f}s)"

    def contains(self, time: float) -> bool:
        """Check if a time point is within this range."""
        return self.start <= time <= self.end


# ============================================================================
# Clip Class
# ============================================================================


@dataclass
class MIDINote:
    """Represents a single MIDI note.

    Attributes:
        note: MIDI note number (0-127)
        velocity: Note velocity (0-127)
        start_time: Note start time in seconds
        duration: Note duration in seconds
        channel: MIDI channel (0-15)
    """
    note: int
    velocity: int
    start_time: float
    duration: float
    channel: int = 0


class MIDIClip:
    """Container for MIDI notes.

    Example:
        >>> midi_clip = MIDIClip()
        >>> midi_clip.add_note(60, 100, 0.0, 0.5)  # Middle C
        >>> midi_clip.add_note(64, 90, 0.5, 0.5)   # E
    """

    def __init__(self):
        """Initialize MIDI clip."""
        self.notes: List[MIDINote] = []

    def add_note(self, note: int, velocity: int, start_time: float,
                 duration: float, channel: int = 0) -> None:
        """Add a MIDI note to the clip."""
        self.notes.append(MIDINote(note, velocity, start_time, duration, channel))
        self.notes.sort(key=lambda n: n.start_time)

    def get_notes_in_range(self, start: float, end: float) -> List[MIDINote]:
        """Get all notes that occur within a time range."""
        return [n for n in self.notes
                if n.start_time < end and (n.start_time + n.duration) > start]


class Clip:
    """Represents an audio or MIDI clip on timeline.

    Example:
        >>> clip = Clip("drums.wav")  # Audio clip
        >>> clip.trim(1.0, 5.0)
        >>>
        >>> midi_clip = Clip(MIDIClip(), clip_type="midi")  # MIDI clip
    """

    def __init__(self, source: Union[str, Path, MIDIClip, Any], clip_type: str = "audio"):
        """Initialize clip.

        Args:
            source: Audio file path or MIDIClip for MIDI clips
            clip_type: 'audio' or 'midi'
        """
        self.source = source
        self.clip_type = clip_type
        self.start_time = 0.0
        self.offset = 0.0  # Trim from start of source
        self.duration: Optional[float] = None  # None = full file
        self.fade_in = 0.0
        self.fade_out = 0.0
        self.gain = 1.0  # Linear gain multiplier
        self._cached_duration: Optional[float] = None

    @property
    def is_midi(self) -> bool:
        """Check if this is a MIDI clip."""
        return self.clip_type == "midi" or isinstance(self.source, MIDIClip)

    def trim(self, start: float, end: float) -> "Clip":
        """Trim clip to specific range.

        Args:
            start: Start time within source file (seconds)
            end: End time within source file (seconds)

        Returns:
            Self for method chaining
        """
        self.offset = start
        self.duration = end - start
        return self

    def set_fades(self, fade_in: float = 0.0, fade_out: float = 0.0) -> "Clip":
        """Set fade in/out durations.

        Args:
            fade_in: Fade in duration in seconds
            fade_out: Fade out duration in seconds

        Returns:
            Self for method chaining
        """
        self.fade_in = fade_in
        self.fade_out = fade_out
        return self

    def get_duration(self) -> float:
        """Get clip duration (accounting for trim).

        Returns:
            Duration in seconds
        """
        if self.duration is not None:
            return self.duration

        # Try to determine duration from source
        if isinstance(self.source, (str, Path)):
            if self._cached_duration is None:
                try:
                    import coremusic as cm

                    with cm.AudioFile(str(self.source)) as af:
                        self._cached_duration = af.duration
                except Exception as e:
                    logger.warning(f"Could not determine clip duration: {e}")
                    return 0.0
            return self._cached_duration - self.offset
        return 0.0

    @property
    def end_time(self) -> float:
        """End time of clip on timeline."""
        return self.start_time + self.get_duration()

    def __repr__(self) -> str:
        source_str = str(self.source)
        if len(source_str) > 30:
            source_str = "..." + source_str[-27:]
        return f"Clip(source={source_str}, start={self.start_time:.2f}s, duration={self.get_duration():.2f}s)"


# ============================================================================
# Automation Lane Class
# ============================================================================


class AutomationLane:
    """Automation data for a parameter.

    Example:
        >>> lane = AutomationLane("volume")
        >>> lane.add_point(0.0, 0.0)   # Start at 0
        >>> lane.add_point(2.0, 1.0)   # Fade to 1.0 over 2 seconds
        >>> print(lane.get_value(1.0))  # Get value at 1 second
        0.5
    """

    def __init__(self, parameter: str):
        """Initialize automation lane.

        Args:
            parameter: Parameter name (e.g., "volume", "pan")
        """
        self.parameter = parameter
        self.points: List[Tuple[float, float]] = []  # (time, value)
        self.interpolation = "linear"  # or "step", "cubic"

    def add_point(self, time: float, value: float) -> None:
        """Add automation point.

        Args:
            time: Time in seconds
            value: Parameter value
        """
        self.points.append((time, value))
        self.points.sort(key=lambda p: p[0])

    def remove_point(self, index: int) -> None:
        """Remove automation point by index."""
        if 0 <= index < len(self.points):
            self.points.pop(index)

    def clear(self) -> None:
        """Remove all automation points."""
        self.points.clear()

    def get_value(self, time: float) -> float:
        """Get interpolated value at given time.

        Args:
            time: Time in seconds

        Returns:
            Interpolated parameter value
        """
        if not self.points:
            return 0.0

        # Before first point
        if time <= self.points[0][0]:
            return self.points[0][1]

        # After last point
        if time >= self.points[-1][0]:
            return self.points[-1][1]

        # Find surrounding points
        for i in range(len(self.points) - 1):
            t1, v1 = self.points[i]
            t2, v2 = self.points[i + 1]

            if t1 <= time <= t2:
                if self.interpolation == "step":
                    return v1
                elif self.interpolation == "linear":
                    # Linear interpolation
                    alpha = (time - t1) / (t2 - t1)
                    return v1 + alpha * (v2 - v1)
                elif self.interpolation == "cubic":
                    # Simple cubic interpolation (could be enhanced with Catmull-Rom)
                    alpha = (time - t1) / (t2 - t1)
                    # Smooth step function
                    alpha = alpha * alpha * (3 - 2 * alpha)
                    return v1 + alpha * (v2 - v1)

        return 0.0

    def __repr__(self) -> str:
        return f"AutomationLane(parameter='{self.parameter}', points={len(self.points)}, interpolation='{self.interpolation}')"


# ============================================================================
# Track Class
# ============================================================================


class Track:
    """Represents a single audio or MIDI track.

    Example:
        >>> track = Track("Drums", "audio")
        >>> track.add_clip(Clip("drums.wav"), start_time=0.0)
        >>> track.volume = 0.8
        >>> track.mute = False
    """

    def __init__(self, name: str, track_type: str = "audio"):
        """Initialize track.

        Args:
            name: Track name
            track_type: 'audio' or 'midi'
        """
        self.name = name
        self.track_type = track_type
        self.clips: List[Clip] = []
        self.volume = 1.0
        self.pan = 0.0  # -1.0 (left) to 1.0 (right)
        self.mute = False
        self.solo = False
        self.armed = False  # Recording armed
        self.plugins: List[Any] = []  # AudioUnitPlugin instances
        self.automation: Dict[str, AutomationLane] = {}

    def add_clip(self, clip: Clip, start_time: float) -> Clip:
        """Add audio/MIDI clip at specified time.

        Args:
            clip: Clip to add
            start_time: Start time on timeline in seconds

        Returns:
            The added clip
        """
        clip.start_time = start_time
        self.clips.append(clip)
        return clip

    def remove_clip(self, clip: Clip) -> None:
        """Remove clip from track."""
        if clip in self.clips:
            self.clips.remove(clip)

    def record_enable(self, enabled: bool = True) -> None:
        """Enable/disable recording on this track."""
        self.armed = enabled

    def add_plugin(self, plugin_name: str, plugin_type: str = "effect",
                   manufacturer: str = "appl", **config: Any) -> AudioUnitPlugin:
        """Add AudioUnit plugin to track's effect chain.

        Args:
            plugin_name: Name of AudioUnit plugin (4-char code or name)
            plugin_type: 'effect' or 'instrument'
            manufacturer: Manufacturer code (default 'appl' for Apple)
            **config: Plugin configuration parameters

        Returns:
            AudioUnitPlugin instance

        Example:
            >>> track.add_plugin("AUDelay", plugin_type="effect")
            >>> track.add_plugin("DLSMusicDevice", plugin_type="instrument")
        """
        try:
            plugin = AudioUnitPlugin(plugin_name, plugin_type, manufacturer)
            self.plugins.append(plugin)
            return plugin
        except Exception as e:
            logger.error(f"Could not add plugin {plugin_name}: {e}")
            raise

    def set_instrument(self, instrument_name: str, manufacturer: str = "appl") -> AudioUnitPlugin:
        """Set the instrument plugin for this MIDI track.

        Args:
            instrument_name: Instrument AudioUnit name (e.g., "DLSMusicDevice")
            manufacturer: Manufacturer code (default 'appl')

        Returns:
            AudioUnitPlugin instance
        """
        # Clear existing instrument plugins
        self.plugins = [p for p in self.plugins if p.plugin_type != "instrument"]

        # Add new instrument
        plugin = AudioUnitPlugin(instrument_name, "instrument", manufacturer)
        self.plugins.insert(0, plugin)  # Instrument goes first
        return plugin

    def automate(self, parameter: str) -> AutomationLane:
        """Get or create automation lane for parameter.

        Args:
            parameter: Parameter name to automate

        Returns:
            AutomationLane instance
        """
        if parameter not in self.automation:
            self.automation[parameter] = AutomationLane(parameter)
        return self.automation[parameter]

    def get_clips_at_time(self, time: float) -> List[Clip]:
        """Get all clips active at given time.

        Args:
            time: Time in seconds

        Returns:
            List of active clips
        """
        active_clips = []
        for clip in self.clips:
            if clip.start_time <= time <= clip.end_time:
                active_clips.append(clip)
        return active_clips

    def __repr__(self) -> str:
        return f"Track(name='{self.name}', type='{self.track_type}', clips={len(self.clips)}, plugins={len(self.plugins)})"


# ============================================================================
# Timeline Class
# ============================================================================


class Timeline:
    """Multi-track timeline with transport control.

    Example:
        >>> timeline = Timeline(sample_rate=48000, tempo=128.0)
        >>> drums = timeline.add_track("Drums", "audio")
        >>> drums.add_clip(Clip("drums.wav"), start_time=0.0)
        >>> timeline.add_marker(16.0, "Chorus")
        >>> timeline.set_loop_region(16.0, 32.0)
        >>> timeline.play()
    """

    def __init__(self, sample_rate: float = 44100.0, tempo: float = 120.0):
        """Initialize timeline.

        Args:
            sample_rate: Audio sample rate
            tempo: Initial tempo in BPM
        """
        self.sample_rate = sample_rate
        self.tempo = tempo
        self.tracks: List[Track] = []
        self.markers: List[TimelineMarker] = []
        self.loop_region: Optional[TimeRange] = None
        self._playhead = 0.0
        self._is_playing = False
        self._is_recording = False
        self._link_session: Optional[Any] = None

    def add_track(self, name: str, track_type: str = "audio") -> Track:
        """Add new track to timeline.

        Args:
            name: Track name
            track_type: 'audio' or 'midi'

        Returns:
            Created Track instance
        """
        track = Track(name, track_type)
        self.tracks.append(track)
        return track

    def remove_track(self, track: Track) -> None:
        """Remove track from timeline."""
        if track in self.tracks:
            self.tracks.remove(track)

    def get_track(self, name: str) -> Optional[Track]:
        """Get track by name."""
        for track in self.tracks:
            if track.name == name:
                return track
        return None

    def enable_link(self, enabled: bool = True) -> None:
        """Enable Ableton Link synchronization.

        Args:
            enabled: Whether to enable Link
        """
        if enabled and not self._link_session:
            try:
                import coremusic as cm

                self._link_session = cm.link.LinkSession(bpm=self.tempo)  # type: ignore[attr-defined]
                logger.info("Ableton Link enabled")
            except Exception as e:
                logger.error(f"Could not enable Link: {e}")
        elif not enabled and self._link_session:
            try:
                self._link_session.close()
                self._link_session = None
                logger.info("Ableton Link disabled")
            except Exception:
                pass

    def play(self, from_time: Optional[float] = None) -> None:
        """Start playback.

        Args:
            from_time: Start position in seconds (None = current playhead)
        """
        if from_time is not None:
            self._playhead = from_time

        self._is_playing = True
        logger.info(f"Playback started at {self._playhead:.2f}s")

        # Setup audio output and render loop would go here
        # self._start_playback_engine()

    def pause(self) -> None:
        """Pause playback (keeps playhead position)."""
        self._is_playing = False
        logger.info(f"Playback paused at {self._playhead:.2f}s")

    def stop(self) -> None:
        """Stop playback and reset playhead to start."""
        self._is_playing = False
        self._playhead = 0.0
        logger.info("Playback stopped")

        # Stop audio engine
        # self._stop_playback_engine()

    def record(self, armed_tracks: Optional[List[Track]] = None) -> None:
        """Start recording on armed tracks.

        Args:
            armed_tracks: List of tracks to record (None = all armed tracks)
        """
        if armed_tracks is None:
            armed_tracks = [t for t in self.tracks if t.armed]

        if not armed_tracks:
            logger.warning("No armed tracks to record")
            return

        self._is_recording = True
        logger.info(f"Recording started on {len(armed_tracks)} track(s)")
        self.play()

    def add_marker(self, position: float, name: str, color: Optional[str] = None) -> TimelineMarker:
        """Add marker/cue point at position.

        Args:
            position: Position in seconds
            name: Marker name
            color: Optional color string

        Returns:
            Created TimelineMarker
        """
        marker = TimelineMarker(position, name, color)
        self.markers.append(marker)
        self.markers.sort(key=lambda m: m.position)
        return marker

    def remove_marker(self, marker: TimelineMarker) -> None:
        """Remove marker from timeline."""
        if marker in self.markers:
            self.markers.remove(marker)

    def get_markers_in_range(self, start: float, end: float) -> List[TimelineMarker]:
        """Get all markers within time range."""
        return [m for m in self.markers if start <= m.position <= end]

    def set_loop_region(self, start: float, end: float) -> None:
        """Set loop region.

        Args:
            start: Loop start in seconds
            end: Loop end in seconds
        """
        self.loop_region = TimeRange(start, end)
        logger.info(f"Loop region set: {self.loop_region}")

    def clear_loop_region(self) -> None:
        """Clear loop region."""
        self.loop_region = None

    def export(
        self,
        output_path: str,
        time_range: Optional[TimeRange] = None,
        format: str = "wav",
    ) -> None:
        """Export timeline to audio file (mixdown).

        Args:
            output_path: Output file path
            time_range: Time range to export (None = entire timeline)
            format: Audio format ('wav', 'aiff', etc.)

        Note:
            This is a placeholder - actual rendering would require implementing
            a full audio rendering engine.
        """
        logger.info(f"Export requested: {output_path}")

        if time_range is None:
            # Calculate timeline duration
            duration = self.get_duration()
            time_range = TimeRange(0.0, duration)

        logger.info(f"Export range: {time_range}")

        # Actual rendering would go here:
        # 1. Iterate through time range in buffer-sized chunks
        # 2. For each track, render clips and apply automation
        # 3. Mix all tracks together
        # 4. Write to output file

        raise NotImplementedError("Export functionality requires audio rendering engine")

    def get_duration(self) -> float:
        """Get total timeline duration (end of last clip).

        Returns:
            Duration in seconds
        """
        max_end = 0.0
        for track in self.tracks:
            for clip in track.clips:
                clip_end = clip.end_time
                if clip_end > max_end:
                    max_end = clip_end
        return max_end

    @property
    def playhead(self) -> float:
        """Current playhead position in seconds."""
        return self._playhead

    @playhead.setter
    def playhead(self, position: float) -> None:
        """Set playhead position."""
        self._playhead = max(0.0, position)

    @property
    def is_playing(self) -> bool:
        """Whether timeline is currently playing."""
        return self._is_playing

    @property
    def is_recording(self) -> bool:
        """Whether timeline is currently recording."""
        return self._is_recording

    def __repr__(self) -> str:
        return f"Timeline(tempo={self.tempo}bpm, tracks={len(self.tracks)}, duration={self.get_duration():.2f}s)"


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "AudioUnitPlugin",
    "MIDINote",
    "MIDIClip",
    "TimelineMarker",
    "TimeRange",
    "Clip",
    "AutomationLane",
    "Track",
    "Timeline",
]
