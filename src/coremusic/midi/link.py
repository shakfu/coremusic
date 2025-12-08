#!/usr/bin/env python3
"""Ableton Link + CoreMIDI Integration

This module provides integration between Ableton Link tempo synchronization
and CoreMIDI, enabling:

- MIDI Clock messages synchronized to Link tempo
- Beat-accurate MIDI event scheduling
- Tempo-synced MIDI sequencing
- Multi-device MIDI synchronization via Link

Classes:
    LinkMIDIClock: Sends MIDI clock messages synchronized to Link
    LinkMIDISequencer: Schedules MIDI events at Link beat positions

Example:
    # Send MIDI clock synchronized to Link
    with link.LinkSession(bpm=120.0) as session:
        clock = LinkMIDIClock(session, midi_port)
        clock.start()
        time.sleep(10)
        clock.stop()
"""

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from .. import capi

if TYPE_CHECKING:
    from .. import link as link_module  # type: ignore[attr-defined]
else:
    # Import at runtime (may fail during type checking)
    try:
        from .. import link as link_module
    except ImportError:
        link_module = None  # type: ignore[assignment]


# MIDI Clock Messages (System Real-Time Messages)
MIDI_CLOCK = 0xF8  # Timing Clock (sent 24 times per quarter note)
MIDI_START = 0xFA  # Start
MIDI_CONTINUE = 0xFB  # Continue
MIDI_STOP = 0xFC  # Stop

# MIDI timing constants
MIDI_CLOCKS_PER_QUARTER_NOTE = 24


@dataclass
class MIDIEvent:
    """Scheduled MIDI event with Link beat position

    Attributes:
        beat: Link beat position when event should occur
        message: MIDI message bytes
        sent: Whether event has been sent
    """
    beat: float
    message: bytes
    sent: bool = False


class LinkMIDIClock:
    """MIDI Clock generator synchronized to Ableton Link

    Sends MIDI timing clock messages (0xF8) at the correct rate based on
    Link's current tempo. Sends 24 clock messages per quarter note as per
    MIDI specification.

    The clock runs in a separate thread and queries Link's beat position
    to determine when to send clock messages. This ensures accurate sync
    even when tempo changes.

    Attributes:
        session: LinkSession instance for tempo synchronization
        midi_port: MIDI output port ID
        midi_destination: MIDI destination endpoint ID
        running: Whether clock is currently running
        quantum: Beat quantum for Link (default 4.0)

    Example:
        with link.LinkSession(bpm=120.0) as session:
            port = capi.midi_output_port_create(client, "Clock Out")
            dest = capi.midi_get_destination(0)

            clock = LinkMIDIClock(session, port, dest)
            clock.start()
            time.sleep(10)
            clock.stop()
    """

    def __init__(
        self,
        session: 'link_module.LinkSession',
        midi_port: int,
        midi_destination: int,
        quantum: float = 4.0
    ):
        """Initialize MIDI clock

        Args:
            session: LinkSession for tempo synchronization
            midi_port: MIDI output port ID from midi_output_port_create()
            midi_destination: MIDI destination endpoint ID
            quantum: Beat quantum for Link (default 4.0 for 4/4 time)
        """
        self.session = session
        self.midi_port = midi_port
        self.midi_destination = midi_destination
        self.quantum = quantum
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_beat = 0.0

    def start(self):
        """Start sending MIDI clock messages

        Sends MIDI Start message (0xFA) and begins clock thread.
        """
        if self.running:
            return

        # Send MIDI Start message
        self._send_realtime_message(MIDI_START)

        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._clock_thread, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop sending MIDI clock messages

        Sends MIDI Stop message (0xFC) and stops clock thread.
        """
        if not self.running:
            return

        self.running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        # Send MIDI Stop message
        self._send_realtime_message(MIDI_STOP)

    def _send_realtime_message(self, status_byte: int):
        """Send MIDI System Real-Time message

        Args:
            status_byte: MIDI status byte (0xF8-0xFF)
        """
        try:
            message = bytes([status_byte])
            capi.midi_send_data(
                self.midi_port,
                self.midi_destination,
                message,
                timestamp=0  # Send immediately
            )
        except Exception as e:
            print(f"Error sending MIDI message: {e}")

    def _clock_thread(self):
        """Clock thread that sends MIDI clock messages

        Queries Link beat position and sends clock messages at the correct
        intervals. Runs at high priority with minimal latency.
        """
        # Calculate sleep time for high resolution timing
        # We'll check at 4x the clock rate for accuracy
        sleep_interval = 1.0 / (MIDI_CLOCKS_PER_QUARTER_NOTE * 4 * 2)  # ~2ms at 120 BPM

        while not self._stop_event.is_set():
            try:
                # Get current Link state
                state = self.session.capture_app_session_state()
                current_time = self.session.clock.micros()

                # Get current beat position
                current_beat = state.beat_at_time(current_time, self.quantum)

                # Calculate how many clocks should have been sent
                # 24 clocks per beat (quarter note)
                current_clock_count = int(current_beat * MIDI_CLOCKS_PER_QUARTER_NOTE)
                last_clock_count = int(self._last_beat * MIDI_CLOCKS_PER_QUARTER_NOTE)

                # Send any missed clocks
                clocks_to_send = current_clock_count - last_clock_count
                if clocks_to_send > 0:
                    for _ in range(min(clocks_to_send, 10)):  # Limit burst to 10
                        self._send_realtime_message(MIDI_CLOCK)

                self._last_beat = current_beat

            except Exception as e:
                print(f"Error in clock thread: {e}")

            time.sleep(sleep_interval)


class LinkMIDISequencer:
    """Beat-accurate MIDI event sequencer synchronized to Link

    Schedules MIDI events at specific Link beat positions. Events are sent
    when Link's beat counter reaches the scheduled beat.

    This enables:
    - Quantized MIDI note triggering
    - Beat-synchronized MIDI CC automation
    - Multi-device synchronized sequencing

    Attributes:
        session: LinkSession instance for timing
        midi_port: MIDI output port ID
        midi_destination: MIDI destination endpoint ID
        quantum: Beat quantum for Link
        events: List of scheduled MIDI events
        running: Whether sequencer is active

    Example:
        seq = LinkMIDISequencer(session, port, dest)

        # Schedule notes at beat positions
        seq.schedule_note(beat=0.0, channel=0, note=60, velocity=100, duration=0.5)
        seq.schedule_note(beat=1.0, channel=0, note=64, velocity=100, duration=0.5)

        seq.start()
        time.sleep(5)
        seq.stop()
    """

    def __init__(
        self,
        session: 'link_module.LinkSession',
        midi_port: int,
        midi_destination: int,
        quantum: float = 4.0
    ):
        """Initialize MIDI sequencer

        Args:
            session: LinkSession for timing
            midi_port: MIDI output port ID
            midi_destination: MIDI destination endpoint ID
            quantum: Beat quantum (default 4.0)
        """
        self.session = session
        self.midi_port = midi_port
        self.midi_destination = midi_destination
        self.quantum = quantum
        self.events: List[MIDIEvent] = []
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def schedule_event(self, beat: float, message: bytes):
        """Schedule a MIDI event at a specific beat position

        Args:
            beat: Link beat position (e.g., 0.0, 1.0, 2.5)
            message: MIDI message bytes
        """
        with self._lock:
            self.events.append(MIDIEvent(beat=beat, message=message))
            # Keep events sorted by beat
            self.events.sort(key=lambda e: e.beat)

    def schedule_note(
        self,
        beat: float,
        channel: int,
        note: int,
        velocity: int,
        duration: float
    ):
        """Schedule a MIDI note with automatic note-off

        Args:
            beat: Beat position to start note
            channel: MIDI channel (0-15)
            note: Note number (0-127)
            velocity: Note velocity (0-127)
            duration: Note duration in beats
        """
        # Note On
        status, data1, data2 = capi.midi_note_on(channel, note, velocity)
        self.schedule_event(beat, bytes([status, data1, data2]))

        # Note Off
        status, data1, data2 = capi.midi_note_off(channel, note, velocity=0)
        self.schedule_event(beat + duration, bytes([status, data1, data2]))

    def schedule_cc(self, beat: float, channel: int, controller: int, value: int):
        """Schedule a MIDI CC message

        Args:
            beat: Beat position
            channel: MIDI channel (0-15)
            controller: Controller number (0-127)
            value: Controller value (0-127)
        """
        status, data1, data2 = capi.midi_control_change(channel, controller, value)
        self.schedule_event(beat, bytes([status, data1, data2]))

    def clear_events(self):
        """Clear all scheduled events"""
        with self._lock:
            self.events.clear()

    def start(self):
        """Start the sequencer"""
        if self.running:
            return

        self.running = True
        self._stop_event.clear()

        # Reset all events to unsent
        with self._lock:
            for event in self.events:
                event.sent = False

        self._thread = threading.Thread(target=self._sequencer_thread, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the sequencer"""
        if not self.running:
            return

        self.running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _sequencer_thread(self):
        """Sequencer thread that sends scheduled events"""
        # Check at high frequency for accurate timing
        sleep_interval = 0.001  # 1ms

        while not self._stop_event.is_set():
            try:
                # Get current beat
                state = self.session.capture_app_session_state()
                current_time = self.session.clock.micros()
                current_beat = state.beat_at_time(current_time, self.quantum)

                # Check for events to send
                with self._lock:
                    for event in self.events:
                        if not event.sent and current_beat >= event.beat:
                            # Send the event
                            try:
                                capi.midi_send_data(
                                    self.midi_port,
                                    self.midi_destination,
                                    event.message,
                                    timestamp=0
                                )
                                event.sent = True
                            except Exception as e:
                                print(f"Error sending MIDI event: {e}")

            except Exception as e:
                print(f"Error in sequencer thread: {e}")

            time.sleep(sleep_interval)


def link_beat_to_host_time(
    session: 'link_module.LinkSession',
    beat: float,
    quantum: float = 4.0
) -> int:
    """Convert Link beat position to host time (mach_absolute_time)

    Useful for scheduling MIDI events with CoreMIDI timestamps.

    Args:
        session: LinkSession instance
        beat: Beat position
        quantum: Beat quantum

    Returns:
        Host time in ticks (mach_absolute_time format)
    """
    state = session.capture_app_session_state()
    time_micros = state.time_at_beat(beat, quantum)
    result: int = session.clock.micros_to_ticks(time_micros)
    return result


def host_time_to_link_beat(
    session: 'link_module.LinkSession',
    host_time_ticks: int,
    quantum: float = 4.0
) -> float:
    """Convert host time to Link beat position

    Args:
        session: LinkSession instance
        host_time_ticks: Host time in ticks (mach_absolute_time)
        quantum: Beat quantum

    Returns:
        Beat position
    """
    time_micros = session.clock.ticks_to_micros(host_time_ticks)
    state = session.capture_app_session_state()
    result: float = state.beat_at_time(time_micros, quantum)
    return result


__all__ = [
    'LinkMIDIClock',
    'LinkMIDISequencer',
    'MIDIEvent',
    'link_beat_to_host_time',
    'host_time_to_link_beat',
    'MIDI_CLOCK',
    'MIDI_START',
    'MIDI_CONTINUE',
    'MIDI_STOP',
    'MIDI_CLOCKS_PER_QUARTER_NOTE',
]
