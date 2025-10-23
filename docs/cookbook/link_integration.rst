Ableton Link Integration
========================

Recipes for tempo synchronization using Ableton Link.

.. contents:: Topics
   :local:
   :depth: 2

Basic Link Usage
----------------

Create Link Session
^^^^^^^^^^^^^^^^^^^

Start a Link session for tempo synchronization:

.. code-block:: python

   import coremusic as cm

   # Create Link session with context manager
   with cm.link.LinkSession(bpm=120.0) as session:
       print(f"Link enabled: {session.enabled}")
       print(f"Connected peers: {session.num_peers}")

       # Get current state
       state = session.capture_app_session_state()
       print(f"Tempo: {state.tempo:.1f} BPM")
       print(f"Playing: {state.is_playing}")

Query Tempo and Beat
^^^^^^^^^^^^^^^^^^^^

Get current tempo and beat position:

.. code-block:: python

   import coremusic as cm
   import time

   with cm.link.LinkSession(bpm=120.0) as session:
       # Start transport
       state = session.capture_app_session_state()
       state.set_is_playing(True, session.clock.micros())
       session.commit_app_session_state(state)

       # Monitor beat position
       for i in range(10):
           time.sleep(0.5)
           state = session.capture_app_session_state()
           current_time = session.clock.micros()
           beat = state.beat_at_time(current_time, 4.0)  # 4/4 time
           print(f"Beat: {beat:.2f}, Tempo: {state.tempo:.1f} BPM")

Change Tempo
^^^^^^^^^^^^

Modify tempo during playback:

.. code-block:: python

   import coremusic as cm
   import time

   with cm.link.LinkSession(bpm=120.0) as session:
       state = session.capture_app_session_state()
       state.set_is_playing(True, session.clock.micros())
       session.commit_app_session_state(state)

       # Gradually increase tempo
       for bpm in range(120, 140, 2):
           state = session.capture_app_session_state()
           state.set_tempo(float(bpm), session.clock.micros())
           session.commit_app_session_state(state)
           time.sleep(1.0)
           print(f"Tempo: {bpm} BPM")

AudioPlayer Integration
-----------------------

Sync AudioPlayer to Link
^^^^^^^^^^^^^^^^^^^^^^^^^

Synchronize audio playback with Link:

.. code-block:: python

   import coremusic as cm
   import time

   # Create Link session
   with cm.link.LinkSession(bpm=120.0) as session:
       # Create AudioPlayer with Link
       player = cm.AudioPlayer(link_session=session)
       player.load_file("audio.wav")
       player.setup_output()

       # Query Link timing
       timing = player.get_link_timing(quantum=4.0)
       print(f"Beat: {timing['beat']:.2f}")
       print(f"Phase: {timing['phase']:.2f}")
       print(f"Tempo: {timing['tempo']:.1f} BPM")
       print(f"Playing: {timing['is_playing']}")

       # Start playback
       player.play()

       # Monitor sync while playing
       for _ in range(10):
           time.sleep(0.5)
           timing = player.get_link_timing(quantum=4.0)
           print(f"Beat: {timing['beat']:.2f}, Phase: {timing['phase']:.2f}")

       player.stop()

Beat-Accurate Playback Start
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start playback on a specific beat:

.. code-block:: python

   import coremusic as cm
   import time

   with cm.link.LinkSession(bpm=120.0) as session:
       player = cm.AudioPlayer(link_session=session)
       player.load_file("loop.wav")
       player.setup_output()

       # Wait for start of next bar (beat 0)
       state = session.capture_app_session_state()
       current_time = session.clock.micros()
       current_beat = state.beat_at_time(current_time, 4.0)

       # Calculate time to next bar
       next_bar_beat = (int(current_beat / 4) + 1) * 4
       next_bar_time = state.time_at_beat(next_bar_beat, 4.0)

       # Wait until next bar
       wait_micros = next_bar_time - current_time
       time.sleep(wait_micros / 1000000.0)

       # Start playback on the beat
       player.play()
       print(f"Started playback on beat {next_bar_beat}")

       time.sleep(5.0)
       player.stop()

MIDI Clock Sync
---------------

Send MIDI Clock
^^^^^^^^^^^^^^^

Synchronize external MIDI devices to Link:

.. code-block:: python

   import coremusic as cm
   from coremusic import link_midi
   import time

   # Create MIDI output
   client = cm.capi.midi_client_create("Link MIDI Clock")
   port = cm.capi.midi_output_port_create(client, "Clock Out")
   dest = cm.capi.midi_get_destination(0)  # First MIDI device

   # Create Link session
   with cm.link.LinkSession(bpm=120.0) as session:
       # Create MIDI clock synchronized to Link
       clock = link_midi.LinkMIDIClock(session, port, dest)

       # Start sending MIDI clock
       clock.start()
       print("Sending MIDI clock at 120 BPM")

       # Let it run for 10 seconds
       time.sleep(10)

       # Change tempo
       state = session.capture_app_session_state()
       state.set_tempo(140.0, session.clock.micros())
       session.commit_app_session_state(state)
       print("Changed tempo to 140 BPM")

       time.sleep(5)

       # Stop clock
       clock.stop()

   # Cleanup
   cm.capi.midi_port_dispose(port)
   cm.capi.midi_client_dispose(client)

Beat-Accurate MIDI Sequencing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Schedule MIDI events at specific beat positions:

.. code-block:: python

   import coremusic as cm
   from coremusic import link_midi
   import time

   # Create MIDI output
   client = cm.capi.midi_client_create("Link Sequencer")
   port = cm.capi.midi_output_port_create(client, "Seq Out")
   dest = cm.capi.midi_get_destination(0)

   # Create Link session
   with cm.link.LinkSession(bpm=120.0) as session:
       # Create MIDI sequencer
       sequencer = link_midi.LinkMIDISequencer(session, port, dest)

       # Schedule notes at specific beats
       # Beat 0: C (60)
       sequencer.schedule_note(beat=0.0, channel=0, note=60, velocity=100, duration=0.9)

       # Beat 1: E (64)
       sequencer.schedule_note(beat=1.0, channel=0, note=64, velocity=100, duration=0.9)

       # Beat 2: G (67)
       sequencer.schedule_note(beat=2.0, channel=0, note=67, velocity=100, duration=0.9)

       # Beat 3: C (72)
       sequencer.schedule_note(beat=3.0, channel=0, note=72, velocity=100, duration=0.9)

       # Schedule CC automation
       sequencer.schedule_cc(beat=0.0, channel=0, controller=7, value=100)  # Volume
       sequencer.schedule_cc(beat=2.0, channel=0, controller=7, value=80)

       # Start sequencer
       sequencer.start()
       print("Sequencer started")

       # Let it play
       time.sleep(5)

       # Stop sequencer
       sequencer.stop()

   # Cleanup
   cm.capi.midi_port_dispose(port)
   cm.capi.midi_client_dispose(client)

Multi-Device Sync
-----------------

Sync Multiple Applications
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Connect multiple Link-enabled applications:

.. code-block:: python

   import coremusic as cm
   import time

   # Create first Link session (e.g., for drums)
   with cm.link.LinkSession(bpm=120.0) as session1:
       session1.enabled = True

       # Wait for peer connections
       time.sleep(2)
       print(f"Session 1 - Peers: {session1.num_peers}")

       # Create second Link session (e.g., for bass)
       with cm.link.LinkSession(bpm=120.0) as session2:
           session2.enabled = True

           time.sleep(1)
           print(f"Session 1 - Peers: {session1.num_peers}")
           print(f"Session 2 - Peers: {session2.num_peers}")

           # Both sessions are now synchronized
           state1 = session1.capture_app_session_state()
           state2 = session2.capture_app_session_state()

           current_time = session1.clock.micros()
           beat1 = state1.beat_at_time(current_time, 4.0)
           beat2 = state2.beat_at_time(current_time, 4.0)

           print(f"Session 1 beat: {beat1:.2f}")
           print(f"Session 2 beat: {beat2:.2f}")
           print(f"Synchronized: {abs(beat1 - beat2) < 0.01}")

Transport Control
^^^^^^^^^^^^^^^^^

Control playback state across multiple devices:

.. code-block:: python

   import coremusic as cm
   import time

   with cm.link.LinkSession(bpm=120.0) as session:
       # Enable start/stop sync
       session.start_stop_sync_enabled = True

       # Start transport
       state = session.capture_app_session_state()
       state.set_is_playing(True, session.clock.micros())
       session.commit_app_session_state(state)
       print("Transport started")

       time.sleep(3)

       # Stop transport
       state = session.capture_app_session_state()
       state.set_is_playing(False, session.clock.micros())
       session.commit_app_session_state(state)
       print("Transport stopped")

Advanced Beat Mapping
---------------------

Map Timeline to Beats
^^^^^^^^^^^^^^^^^^^^^

Convert between sample positions and beat positions:

.. code-block:: python

   import coremusic as cm

   with cm.link.LinkSession(bpm=120.0) as session:
       state = session.capture_app_session_state()
       current_time = session.clock.micros()

       # Get current beat
       beat = state.beat_at_time(current_time, 4.0)
       print(f"Current beat: {beat:.2f}")

       # Get phase within bar (0.0 - 4.0 for 4/4 time)
       phase = state.phase_at_time(current_time, 4.0)
       print(f"Phase: {phase:.2f}")

       # Calculate time for future beat
       future_beat = beat + 8.0  # 2 bars from now
       future_time = state.time_at_beat(future_beat, 4.0)
       wait_micros = future_time - current_time
       print(f"2 bars from now in {wait_micros / 1000000.0:.2f} seconds")

Request Beat Alignment
^^^^^^^^^^^^^^^^^^^^^^

Align beat grid to specific events:

.. code-block:: python

   import coremusic as cm

   with cm.link.LinkSession(bpm=120.0) as session:
       state = session.capture_app_session_state()
       current_time = session.clock.micros()

       # Request that beat 0 occurs now
       state.request_beat_at_time(0.0, current_time, 4.0)
       session.commit_app_session_state(state)
       print("Beat grid aligned to current time")

       # Or align to start of playback
       state = session.capture_app_session_state()
       state.request_beat_at_start_playing_time(0.0, 4.0)
       session.commit_app_session_state(state)
       print("Beat 0 will occur when transport starts")

Tempo-Synced Loops
^^^^^^^^^^^^^^^^^^

Create loops that stay synchronized:

.. code-block:: python

   import coremusic as cm
   import time

   with cm.link.LinkSession(bpm=120.0) as session:
       # 4-bar loop
       loop_length_beats = 16.0

       state = session.capture_app_session_state()
       state.set_is_playing(True, session.clock.micros())
       session.commit_app_session_state(state)

       # Monitor loop position
       for _ in range(20):
           time.sleep(0.5)
           state = session.capture_app_session_state()
           current_time = session.clock.micros()

           # Get beat position
           beat = state.beat_at_time(current_time, 4.0)

           # Calculate loop position
           loop_beat = beat % loop_length_beats
           bar = int(loop_beat / 4) + 1
           beat_in_bar = (loop_beat % 4) + 1

           print(f"Bar {bar}, Beat {beat_in_bar:.1f}")

Complete Example: Drum Machine
-------------------------------

Full example of a Link-synchronized drum machine:

.. code-block:: python

   import coremusic as cm
   from coremusic import link_midi
   import time

   def create_drum_pattern():
       """Create a simple drum pattern"""
       pattern = []

       # 4 bars of 4/4 time
       for bar in range(4):
           bar_start = bar * 4.0

           # Kick on beats 1 and 3
           pattern.append((bar_start + 0.0, 36, 100))  # Beat 1
           pattern.append((bar_start + 2.0, 36, 100))  # Beat 3

           # Snare on beats 2 and 4
           pattern.append((bar_start + 1.0, 38, 100))  # Beat 2
           pattern.append((bar_start + 3.0, 38, 100))  # Beat 4

           # Hi-hat every half beat
           for eighth in range(8):
               pattern.append((bar_start + eighth * 0.5, 42, 80))

       return pattern

   # Setup MIDI
   client = cm.capi.midi_client_create("Drum Machine")
   port = cm.capi.midi_output_port_create(client, "Drums")
   dest = cm.capi.midi_get_destination(0)

   # Create Link session
   with cm.link.LinkSession(bpm=120.0) as session:
       # Create sequencer
       sequencer = link_midi.LinkMIDISequencer(session, port, dest)

       # Load pattern
       pattern = create_drum_pattern()
       for beat, note, velocity in pattern:
           sequencer.schedule_note(
               beat=beat,
               channel=9,  # MIDI channel 10 (index 9) for drums
               note=note,
               velocity=velocity,
               duration=0.1
           )

       print(f"Loaded {len(pattern)} drum hits")
       print(f"Link tempo: {session.capture_app_session_state().tempo:.1f} BPM")
       print(f"Connected peers: {session.num_peers}")

       # Start playback
       sequencer.start()
       print("Drum machine started!")

       # Run for 16 bars
       time.sleep(16 * 4 * 60.0 / 120.0)  # 16 bars at 120 BPM

       # Stop
       sequencer.stop()
       print("Drum machine stopped")

   # Cleanup
   cm.capi.midi_port_dispose(port)
   cm.capi.midi_client_dispose(client)

Best Practices
--------------

Session Management
^^^^^^^^^^^^^^^^^^

Always use context managers:

.. code-block:: python

   # Good: Automatic cleanup
   with cm.link.LinkSession(bpm=120.0) as session:
       # Use session
       pass

   # Avoid: Manual management
   session = cm.link.LinkSession(bpm=120.0)
   try:
       session.enabled = True
       # Use session
   finally:
       session.enabled = False

State Capture and Commit
^^^^^^^^^^^^^^^^^^^^^^^^^

Capture state, modify, then commit:

.. code-block:: python

   with cm.link.LinkSession(bpm=120.0) as session:
       # Capture current state
       state = session.capture_app_session_state()

       # Modify state
       state.set_tempo(140.0, session.clock.micros())
       state.set_is_playing(True, session.clock.micros())

       # Commit changes
       session.commit_app_session_state(state)

Timing Precision
^^^^^^^^^^^^^^^^

Use microsecond precision for accurate timing:

.. code-block:: python

   with cm.link.LinkSession(bpm=120.0) as session:
       # Always use clock.micros() for current time
       current_time = session.clock.micros()

       state = session.capture_app_session_state()
       beat = state.beat_at_time(current_time, 4.0)

       # Don't use time.time() - it's not precise enough for audio

Thread Safety
^^^^^^^^^^^^^

Link operations are thread-safe, but use audio thread for time-critical operations:

.. code-block:: python

   import threading
   import coremusic as cm

   with cm.link.LinkSession(bpm=120.0) as session:
       def audio_thread():
           """This runs on audio thread"""
           # Capture state on audio thread for low latency
           state = session.capture_audio_session_state()
           current_time = session.clock.micros()
           beat = state.beat_at_time(current_time, 4.0)
           # Process audio...

       def ui_thread():
           """This runs on UI thread"""
           # Capture state on UI thread for UI updates
           state = session.capture_app_session_state()
           tempo = state.tempo
           # Update UI...

See Also
--------

- :doc:`/api/index` - Complete API reference
- :doc:`audiounit_hosting` - AudioUnit plugin hosting
- :doc:`midi_processing` - MIDI I/O and processing
- Ableton Link documentation: https://ableton.github.io/link/
