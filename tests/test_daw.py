"""Tests for DAW (Digital Audio Workstation) essentials module."""

import os
import pytest
from pathlib import Path
import coremusic as cm
from coremusic.daw import (
    TimelineMarker,
    TimeRange,
    Clip,
    AutomationLane,
    Track,
    Timeline,
)


class TestTimelineMarker:
    """Test TimelineMarker data class"""

    def test_create_marker(self):
        """Test marker creation"""
        marker = TimelineMarker(16.0, "Chorus")
        assert marker.position == 16.0
        assert marker.name == "Chorus"
        assert marker.color is None

    def test_marker_with_color(self):
        """Test marker with color"""
        marker = TimelineMarker(32.0, "Bridge", color="#FF0000")
        assert marker.position == 32.0
        assert marker.name == "Bridge"
        assert marker.color == "#FF0000"

    def test_marker_repr(self):
        """Test marker string representation"""
        marker = TimelineMarker(8.0, "Verse")
        repr_str = repr(marker)
        assert "8.00s" in repr_str
        assert "Verse" in repr_str


class TestTimeRange:
    """Test TimeRange data class"""

    def test_create_time_range(self):
        """Test time range creation"""
        time_range = TimeRange(10.0, 20.0)
        assert time_range.start == 10.0
        assert time_range.end == 20.0
        assert time_range.duration == 10.0

    def test_contains(self):
        """Test time range contains"""
        time_range = TimeRange(5.0, 15.0)
        assert time_range.contains(5.0)
        assert time_range.contains(10.0)
        assert time_range.contains(15.0)
        assert not time_range.contains(4.9)
        assert not time_range.contains(15.1)

    def test_zero_duration(self):
        """Test zero duration range"""
        time_range = TimeRange(5.0, 5.0)
        assert time_range.duration == 0.0


class TestClip:
    """Test Clip class"""

    def test_create_clip(self):
        """Test clip creation"""
        clip = Clip("test.wav")
        assert clip.source == "test.wav"
        assert clip.start_time == 0.0
        assert clip.offset == 0.0
        assert clip.duration is None
        assert clip.fade_in == 0.0
        assert clip.fade_out == 0.0
        assert clip.gain == 1.0

    def test_trim_clip(self):
        """Test clip trimming"""
        clip = Clip("test.wav")
        result = clip.trim(1.0, 5.0)
        assert result is clip  # Returns self
        assert clip.offset == 1.0
        assert clip.duration == 4.0

    def test_set_fades(self):
        """Test setting fades"""
        clip = Clip("test.wav")
        result = clip.set_fades(fade_in=0.5, fade_out=0.3)
        assert result is clip  # Returns self
        assert clip.fade_in == 0.5
        assert clip.fade_out == 0.3

    def test_method_chaining(self):
        """Test method chaining"""
        clip = Clip("test.wav").trim(1.0, 5.0).set_fades(0.5, 0.3)
        assert clip.offset == 1.0
        assert clip.duration == 4.0
        assert clip.fade_in == 0.5
        assert clip.fade_out == 0.3

    def test_clip_with_path(self):
        """Test clip with Path object"""
        clip = Clip(Path("test.wav"))
        assert isinstance(clip.source, Path)

    def test_get_duration_with_explicit_duration(self):
        """Test get_duration with explicit duration"""
        clip = Clip("test.wav")
        clip.duration = 10.0
        assert clip.get_duration() == 10.0

    def test_end_time_property(self):
        """Test end_time property"""
        clip = Clip("test.wav")
        clip.start_time = 5.0
        clip.duration = 10.0
        assert clip.end_time == 15.0


class TestAutomationLane:
    """Test AutomationLane class"""

    def test_create_automation_lane(self):
        """Test automation lane creation"""
        lane = AutomationLane("volume")
        assert lane.parameter == "volume"
        assert len(lane.points) == 0
        assert lane.interpolation == "linear"

    def test_add_points(self):
        """Test adding automation points"""
        lane = AutomationLane("volume")
        lane.add_point(0.0, 0.0)
        lane.add_point(2.0, 1.0)
        lane.add_point(4.0, 0.5)

        assert len(lane.points) == 3
        assert lane.points[0] == (0.0, 0.0)
        assert lane.points[1] == (2.0, 1.0)
        assert lane.points[2] == (4.0, 0.5)

    def test_points_sorted(self):
        """Test that points are automatically sorted"""
        lane = AutomationLane("volume")
        lane.add_point(4.0, 0.5)
        lane.add_point(0.0, 0.0)
        lane.add_point(2.0, 1.0)

        assert lane.points[0] == (0.0, 0.0)
        assert lane.points[1] == (2.0, 1.0)
        assert lane.points[2] == (4.0, 0.5)

    def test_linear_interpolation(self):
        """Test linear interpolation"""
        lane = AutomationLane("volume")
        lane.interpolation = "linear"
        lane.add_point(0.0, 0.0)
        lane.add_point(2.0, 1.0)

        assert lane.get_value(0.0) == 0.0
        assert lane.get_value(1.0) == pytest.approx(0.5)
        assert lane.get_value(2.0) == 1.0

    def test_step_interpolation(self):
        """Test step interpolation"""
        lane = AutomationLane("volume")
        lane.interpolation = "step"
        lane.add_point(0.0, 0.0)
        lane.add_point(2.0, 1.0)

        assert lane.get_value(0.0) == 0.0
        assert lane.get_value(1.0) == 0.0  # Step holds first value
        assert lane.get_value(2.0) == 1.0

    def test_cubic_interpolation(self):
        """Test cubic interpolation"""
        lane = AutomationLane("volume")
        lane.interpolation = "cubic"
        lane.add_point(0.0, 0.0)
        lane.add_point(2.0, 1.0)

        value_at_1 = lane.get_value(1.0)
        assert 0.4 < value_at_1 < 0.6  # Cubic should be smoother

    def test_value_before_first_point(self):
        """Test value before first point"""
        lane = AutomationLane("volume")
        lane.add_point(5.0, 1.0)
        assert lane.get_value(0.0) == 1.0  # Returns first value

    def test_value_after_last_point(self):
        """Test value after last point"""
        lane = AutomationLane("volume")
        lane.add_point(0.0, 0.5)
        assert lane.get_value(10.0) == 0.5  # Returns last value

    def test_empty_lane(self):
        """Test empty automation lane"""
        lane = AutomationLane("volume")
        assert lane.get_value(1.0) == 0.0

    def test_remove_point(self):
        """Test removing automation point"""
        lane = AutomationLane("volume")
        lane.add_point(0.0, 0.0)
        lane.add_point(1.0, 1.0)
        lane.add_point(2.0, 0.5)

        lane.remove_point(1)
        assert len(lane.points) == 2
        assert lane.points[1] == (2.0, 0.5)

    def test_clear(self):
        """Test clearing all points"""
        lane = AutomationLane("volume")
        lane.add_point(0.0, 0.0)
        lane.add_point(1.0, 1.0)

        lane.clear()
        assert len(lane.points) == 0


class TestTrack:
    """Test Track class"""

    def test_create_audio_track(self):
        """Test audio track creation"""
        track = Track("Drums", "audio")
        assert track.name == "Drums"
        assert track.track_type == "audio"
        assert len(track.clips) == 0
        assert track.volume == 1.0
        assert track.pan == 0.0
        assert not track.mute
        assert not track.solo
        assert not track.armed

    def test_create_midi_track(self):
        """Test MIDI track creation"""
        track = Track("Bass", "midi")
        assert track.track_type == "midi"

    def test_add_clip(self):
        """Test adding clip to track"""
        track = Track("Drums")
        clip = Clip("test.wav")
        result = track.add_clip(clip, start_time=5.0)

        assert result is clip
        assert len(track.clips) == 1
        assert clip.start_time == 5.0

    def test_remove_clip(self):
        """Test removing clip from track"""
        track = Track("Drums")
        clip1 = Clip("test1.wav")
        clip2 = Clip("test2.wav")

        track.add_clip(clip1, 0.0)
        track.add_clip(clip2, 5.0)

        track.remove_clip(clip1)
        assert len(track.clips) == 1
        assert clip2 in track.clips

    def test_record_enable(self):
        """Test record enabling"""
        track = Track("Vocals")
        assert not track.armed

        track.record_enable(True)
        assert track.armed

        track.record_enable(False)
        assert not track.armed

    def test_automate(self):
        """Test automation lane creation"""
        track = Track("Vocals")
        volume_lane = track.automate("volume")

        assert isinstance(volume_lane, AutomationLane)
        assert volume_lane.parameter == "volume"
        assert "volume" in track.automation

        # Getting same automation again returns same instance
        volume_lane2 = track.automate("volume")
        assert volume_lane is volume_lane2

    def test_get_clips_at_time(self):
        """Test getting clips at specific time"""
        track = Track("Drums")
        clip1 = Clip("test1.wav")
        clip1.duration = 5.0
        clip2 = Clip("test2.wav")
        clip2.duration = 5.0

        track.add_clip(clip1, start_time=0.0)   # 0-5
        track.add_clip(clip2, start_time=10.0)  # 10-15

        clips_at_2 = track.get_clips_at_time(2.0)
        assert len(clips_at_2) == 1
        assert clip1 in clips_at_2

        clips_at_7 = track.get_clips_at_time(7.0)
        assert len(clips_at_7) == 0

        clips_at_12 = track.get_clips_at_time(12.0)
        assert len(clips_at_12) == 1
        assert clip2 in clips_at_12


class TestTimeline:
    """Test Timeline class"""

    def test_create_timeline(self):
        """Test timeline creation"""
        timeline = Timeline(sample_rate=48000, tempo=128.0)
        assert timeline.sample_rate == 48000
        assert timeline.tempo == 128.0
        assert len(timeline.tracks) == 0
        assert len(timeline.markers) == 0
        assert timeline.loop_region is None
        assert timeline.playhead == 0.0
        assert not timeline.is_playing
        assert not timeline.is_recording

    def test_add_track(self):
        """Test adding track to timeline"""
        timeline = Timeline()
        drums = timeline.add_track("Drums", "audio")

        assert isinstance(drums, Track)
        assert drums.name == "Drums"
        assert len(timeline.tracks) == 1

    def test_remove_track(self):
        """Test removing track"""
        timeline = Timeline()
        drums = timeline.add_track("Drums")
        bass = timeline.add_track("Bass")

        timeline.remove_track(drums)
        assert len(timeline.tracks) == 1
        assert bass in timeline.tracks

    def test_get_track(self):
        """Test getting track by name"""
        timeline = Timeline()
        timeline.add_track("Drums")
        timeline.add_track("Bass")

        drums = timeline.get_track("Drums")
        assert drums is not None
        assert drums.name == "Drums"

        nonexistent = timeline.get_track("Vocals")
        assert nonexistent is None

    def test_playhead_property(self):
        """Test playhead property"""
        timeline = Timeline()
        assert timeline.playhead == 0.0

        timeline.playhead = 10.0
        assert timeline.playhead == 10.0

        # Can't set negative playhead
        timeline.playhead = -5.0
        assert timeline.playhead == 0.0

    def test_play(self):
        """Test play"""
        timeline = Timeline()
        timeline.play()

        assert timeline.is_playing
        assert timeline.playhead == 0.0

    def test_play_from_time(self):
        """Test play from specific time"""
        timeline = Timeline()
        timeline.play(from_time=10.0)

        assert timeline.is_playing
        assert timeline.playhead == 10.0

    def test_pause(self):
        """Test pause"""
        timeline = Timeline()
        timeline.play()
        timeline.playhead = 5.0
        timeline.pause()

        assert not timeline.is_playing
        assert timeline.playhead == 5.0  # Playhead preserved

    def test_stop(self):
        """Test stop"""
        timeline = Timeline()
        timeline.play()
        timeline.playhead = 5.0
        timeline.stop()

        assert not timeline.is_playing
        assert timeline.playhead == 0.0  # Playhead reset

    def test_record(self):
        """Test record"""
        timeline = Timeline()
        track1 = timeline.add_track("Track1")
        track2 = timeline.add_track("Track2")

        track1.record_enable(True)

        timeline.record()
        assert timeline.is_recording
        assert timeline.is_playing

    def test_record_no_armed_tracks(self):
        """Test record with no armed tracks"""
        timeline = Timeline()
        timeline.add_track("Track1")

        timeline.record()
        # Should not crash, just log warning

    def test_add_marker(self):
        """Test adding marker"""
        timeline = Timeline()
        marker = timeline.add_marker(16.0, "Chorus")

        assert isinstance(marker, TimelineMarker)
        assert marker.position == 16.0
        assert marker.name == "Chorus"
        assert len(timeline.markers) == 1

    def test_markers_sorted(self):
        """Test that markers are sorted by position"""
        timeline = Timeline()
        timeline.add_marker(32.0, "Bridge")
        timeline.add_marker(0.0, "Intro")
        timeline.add_marker(16.0, "Verse")

        assert timeline.markers[0].name == "Intro"
        assert timeline.markers[1].name == "Verse"
        assert timeline.markers[2].name == "Bridge"

    def test_remove_marker(self):
        """Test removing marker"""
        timeline = Timeline()
        marker1 = timeline.add_marker(0.0, "Intro")
        marker2 = timeline.add_marker(16.0, "Verse")

        timeline.remove_marker(marker1)
        assert len(timeline.markers) == 1
        assert marker2 in timeline.markers

    def test_get_markers_in_range(self):
        """Test getting markers in time range"""
        timeline = Timeline()
        timeline.add_marker(5.0, "A")
        timeline.add_marker(10.0, "B")
        timeline.add_marker(15.0, "C")
        timeline.add_marker(20.0, "D")

        markers = timeline.get_markers_in_range(8.0, 17.0)
        assert len(markers) == 2
        assert markers[0].name == "B"
        assert markers[1].name == "C"

    def test_set_loop_region(self):
        """Test setting loop region"""
        timeline = Timeline()
        timeline.set_loop_region(16.0, 32.0)

        assert timeline.loop_region is not None
        assert timeline.loop_region.start == 16.0
        assert timeline.loop_region.end == 32.0

    def test_clear_loop_region(self):
        """Test clearing loop region"""
        timeline = Timeline()
        timeline.set_loop_region(16.0, 32.0)
        timeline.clear_loop_region()

        assert timeline.loop_region is None

    def test_get_duration(self):
        """Test getting timeline duration"""
        timeline = Timeline()
        track1 = timeline.add_track("Track1")
        track2 = timeline.add_track("Track2")

        clip1 = Clip("test1.wav")
        clip1.duration = 10.0
        clip2 = Clip("test2.wav")
        clip2.duration = 5.0

        track1.add_clip(clip1, start_time=0.0)   # Ends at 10
        track2.add_clip(clip2, start_time=20.0)  # Ends at 25

        assert timeline.get_duration() == 25.0

    def test_empty_timeline_duration(self):
        """Test duration of empty timeline"""
        timeline = Timeline()
        assert timeline.get_duration() == 0.0


class TestIntegration:
    """Integration tests for DAW workflow"""

    def test_complete_workflow(self):
        """Test complete DAW workflow"""
        # Create timeline
        timeline = Timeline(sample_rate=48000, tempo=128.0)

        # Add tracks
        drums = timeline.add_track("Drums", "audio")
        bass = timeline.add_track("Bass", "audio")
        vocals = timeline.add_track("Vocals", "audio")

        # Add clips
        drums_clip = Clip("drums.wav")
        drums_clip.duration = 32.0
        drums.add_clip(drums_clip, start_time=0.0)

        vocals_clip = Clip("vocals.wav")
        vocals_clip.duration = 24.0
        vocals.add_clip(vocals_clip, start_time=8.0)

        # Add markers
        timeline.add_marker(0.0, "Intro")
        timeline.add_marker(16.0, "Verse")
        timeline.add_marker(32.0, "Chorus")

        # Add automation
        volume_automation = vocals.automate("volume")
        volume_automation.add_point(8.0, 0.0)    # Fade in
        volume_automation.add_point(10.0, 1.0)
        volume_automation.add_point(30.0, 1.0)   # Fade out
        volume_automation.add_point(32.0, 0.0)

        # Set loop region
        timeline.set_loop_region(16.0, 32.0)

        # Verify state
        assert len(timeline.tracks) == 3
        assert len(timeline.markers) == 3
        assert timeline.loop_region is not None
        assert timeline.loop_region.duration == 16.0
        assert timeline.get_duration() == 32.0

        # Test automation values
        assert volume_automation.get_value(8.0) == 0.0
        assert volume_automation.get_value(9.0) == pytest.approx(0.5)
        assert volume_automation.get_value(10.0) == 1.0

    def test_multitrack_clip_management(self):
        """Test managing clips across multiple tracks"""
        timeline = Timeline()

        # Create 3 tracks with multiple clips each
        for i in range(3):
            track = timeline.add_track(f"Track{i+1}")

            # Add 3 clips per track
            for j in range(3):
                clip = Clip(f"clip_{i}_{j}.wav")
                clip.duration = 5.0
                track.add_clip(clip, start_time=j * 10.0)

        # Verify structure
        assert len(timeline.tracks) == 3
        for track in timeline.tracks:
            assert len(track.clips) == 3

        # Test clip retrieval at various times
        track1 = timeline.tracks[0]
        assert len(track1.get_clips_at_time(2.0)) == 1
        assert len(track1.get_clips_at_time(7.0)) == 0
        assert len(track1.get_clips_at_time(12.0)) == 1
