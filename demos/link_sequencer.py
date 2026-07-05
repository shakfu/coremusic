#!/usr/bin/env python3
"""A step sequencer whose timing is driven by Ableton Link.

Creates an Ableton Link session (so it tempo- and phase-syncs with any other
Link-enabled app on the network -- Ableton Live, another instance of this
script, etc.) and plays a repeating note pattern locked to the shared Link
beat grid. Audio is synthesized directly in the output stream's render
callback, which reads Link's beat position via the audio-thread session-state
API.

If no audio output device is available, it falls back to printing the live
Link timeline (tempo, beat, phase, peer count) so the sync is still visible.

Usage::

    python demos/link_sequencer.py                    # 120 BPM, 12 s
    python demos/link_sequencer.py --bpm 100 --duration 30
    python demos/link_sequencer.py --print-only       # no audio, show timeline

Start Ableton Live (or a second copy of this script) with Link enabled to see
the tempo and phase lock together.
"""

from __future__ import annotations

import argparse
import math
import struct
import time
from typing import Callable

from coremusic import link
from coremusic.audio.streaming import AudioOutputStream

# 8-step pattern of MIDI note numbers; None is a rest. A minor-ish riff.
PATTERN: list[int | None] = [57, None, 60, 64, None, 67, 64, 60]


def midi_to_hz(note: int) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def make_link_generator(
    session: "link.LinkSession",
    *,
    sample_rate: float,
    channels: int,
    quantum: float,
    gain: float,
) -> Callable[[int], bytes]:
    """Build the render callback that sequences notes off the Link timeline."""
    clock = session.clock
    num_steps = len(PATTERN)
    step_len = quantum / num_steps  # length of one step in beats
    phase = 0.0
    current_step = -1

    def generate(frame_count: int) -> bytes:
        nonlocal phase, current_step
        # Capture Link's shared timeline once per buffer (audio-thread API),
        # then extrapolate per sample using the current tempo.
        state = session.capture_audio_session_state()
        tempo = state.tempo
        beat0 = state.beat_at_time(clock.micros(), quantum)
        beats_per_sample = (tempo / 60.0) / sample_rate

        out = []
        for i in range(frame_count):
            beat = beat0 + i * beats_per_sample
            bar_beat = beat % quantum
            step = int(bar_beat / step_len) % num_steps
            frac = (bar_beat - step * step_len) / step_len  # 0..1 within step

            note = PATTERN[step]
            if note is None:
                sample = 0.0
            else:
                if step != current_step:
                    # New note: restart oscillator + envelope for a clean pluck.
                    phase = 0.0
                    current_step = step
                envelope = math.exp(-frac * 5.0)
                sample = math.sin(phase) * envelope * gain
                phase += 2.0 * math.pi * midi_to_hz(note) / sample_rate
                if phase >= 2.0 * math.pi:
                    phase -= 2.0 * math.pi
            out.extend([sample] * channels)

        return struct.pack(f"<{len(out)}f", *out)

    return generate


def print_timeline(
    session: "link.LinkSession", quantum: float, duration: float
) -> None:
    """Fallback: show the live Link timeline without producing audio."""
    clock = session.clock
    print("No audio output; showing Link timeline (Ctrl-C to stop):")
    end = time.monotonic() + duration
    try:
        while time.monotonic() < end:
            state = session.capture_app_session_state()
            now = clock.micros()
            beat = state.beat_at_time(now, quantum)
            phase = state.phase_at_time(now, quantum)
            print(
                f"  tempo={state.tempo:6.1f} BPM  beat={beat:8.2f}  "
                f"phase={phase:4.2f}/{quantum:.0f}  peers={session.num_peers}",
                end="\r",
                flush=True,
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bpm", type=float, default=120.0, help="Initial tempo")
    parser.add_argument("--duration", type=float, default=12.0, help="Seconds to run")
    parser.add_argument(
        "--quantum", type=float, default=4.0, help="Beats per bar / sync quantum"
    )
    parser.add_argument("--channels", type=int, default=2, help="Output channels")
    parser.add_argument(
        "--sample-rate", type=float, default=44100.0, help="Sample rate"
    )
    parser.add_argument("--gain", type=float, default=0.3, help="Amplitude 0.0-1.0")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Do not play audio; just print the Link timeline",
    )
    args = parser.parse_args()

    with link.LinkSession(bpm=args.bpm) as session:
        session.enabled = True  # start network discovery / sync
        print(
            f"Link enabled at {args.bpm:.0f} BPM (quantum={args.quantum:.0f}). "
            f"Peers: {session.num_peers}"
        )

        if args.print_only:
            print_timeline(session, args.quantum, args.duration)
            return 0

        stream = AudioOutputStream(
            channels=args.channels, sample_rate=args.sample_rate, buffer_size=512
        )
        stream.set_generator(
            make_link_generator(
                session,
                sample_rate=args.sample_rate,
                channels=args.channels,
                quantum=args.quantum,
                gain=args.gain,
            )
        )
        try:
            print(
                f"Sequencing {len(PATTERN)} steps for {args.duration:.0f} s... "
                f"Ctrl-C to stop."
            )
            with stream:
                time.sleep(args.duration)
        except KeyboardInterrupt:
            print("\nStopped.")
        except Exception as exc:  # noqa: BLE001 - fall back to the visual timeline
            print(f"Audio output unavailable ({exc}); showing timeline instead.")
            print_timeline(session, args.quantum, args.duration)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
