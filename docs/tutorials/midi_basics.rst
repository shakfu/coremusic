MIDI Basics
===========

This tutorial covers MIDI fundamentals with coremusic, including sending, receiving, and processing MIDI messages.

Prerequisites
-------------

- coremusic installed and built
- Basic Python knowledge
- Optional: A MIDI controller or virtual MIDI device

Understanding MIDI
------------------

MIDI (Musical Instrument Digital Interface) is a protocol for communicating musical information:

- **Note On/Off**: When keys are pressed/released
- **Control Change (CC)**: Knobs, sliders, pedals
- **Program Change**: Patch/preset selection
- **Pitch Bend**: Pitch wheel position
- **Aftertouch**: Pressure after key press

MIDI Devices
------------

Listing Devices
^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def list_midi_devices():
       """List all MIDI devices, sources, and destinations."""
       # Count devices
       num_devices = cm.midi_get_number_of_devices()
       num_sources = cm.midi_get_number_of_sources()
       num_destinations = cm.midi_get_number_of_destinations()

       print(f"MIDI System Overview:")
       print(f"  Devices: {num_devices}")
       print(f"  Sources (inputs): {num_sources}")
       print(f"  Destinations (outputs): {num_destinations}")
       print()

       # List sources (inputs)
       print("MIDI Sources (Inputs):")
       for i in range(num_sources):
           source = cm.midi_get_source(i)
           try:
               name = cm.midi_object_get_string_property(
                   source, cm.get_midi_property_name()
               )
               print(f"  [{i}] {name}")
           except:
               print(f"  [{i}] <unknown>")

       print()

       # List destinations (outputs)
       print("MIDI Destinations (Outputs):")
       for i in range(num_destinations):
           dest = cm.midi_get_destination(i)
           try:
               name = cm.midi_object_get_string_property(
                   dest, cm.get_midi_property_name()
               )
               print(f"  [{i}] {name}")
           except:
               print(f"  [{i}] <unknown>")

   list_midi_devices()

Using the CLI
^^^^^^^^^^^^^

.. code-block:: bash

   # List all MIDI devices
   coremusic midi list

   # Get detailed device info
   coremusic midi device info "Device Name"

Creating a MIDI Client
----------------------

All MIDI operations require a client:

.. code-block:: python

   import coremusic as cm

   # Create MIDI client
   client = cm.MIDIClient("My Application")

   try:
       # Use the client...
       print(f"Created MIDI client: {client.name}")

   finally:
       # Always dispose when done
       client.dispose()

Or use context manager:

.. code-block:: python

   import coremusic as cm

   with cm.MIDIClient("My Application") as client:
       print(f"MIDI client active: {client.name}")
       # Client is automatically disposed when exiting

Sending MIDI Messages
---------------------

Creating an Output Port
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def setup_midi_output():
       """Set up MIDI output."""
       client = cm.MIDIClient("MIDI Sender")
       output_port = client.create_output_port("Output")

       return client, output_port

   client, port = setup_midi_output()

Sending Note Messages
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   import time

   def send_note(client, port, note, velocity=100, duration=0.5, channel=0):
       """Send a note on/off pair."""
       # Get first destination
       if cm.midi_get_number_of_destinations() == 0:
           print("No MIDI destinations available")
           return

       dest = cm.midi_get_destination(0)

       # Create Note On message
       # Status byte: 0x90 + channel (Note On on channel)
       note_on = bytes([0x90 + channel, note, velocity])

       # Create Note Off message
       # Status byte: 0x80 + channel (Note Off on channel)
       note_off = bytes([0x80 + channel, note, 0])

       # Send Note On
       port.send(dest, note_on)
       print(f"Note On: {note} velocity={velocity}")

       # Wait for duration
       time.sleep(duration)

       # Send Note Off
       port.send(dest, note_off)
       print(f"Note Off: {note}")

   # Send middle C
   client = cm.MIDIClient("Note Sender")
   port = client.create_output_port("Output")

   send_note(client, port, note=60, velocity=100, duration=0.5)

   client.dispose()

Sending Control Change
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def send_cc(client, port, controller, value, channel=0):
       """Send Control Change message."""
       dest = cm.midi_get_destination(0)

       # CC message: 0xB0 + channel, controller number, value
       cc_msg = bytes([0xB0 + channel, controller, value])
       port.send(dest, cc_msg)

       print(f"CC {controller}: {value}")

   # Common CC numbers:
   # CC 1  = Modulation wheel
   # CC 7  = Volume
   # CC 10 = Pan
   # CC 64 = Sustain pedal
   # CC 123 = All Notes Off

   client = cm.MIDIClient("CC Sender")
   port = client.create_output_port("Output")

   # Send modulation
   send_cc(client, port, controller=1, value=64)

   # Send volume
   send_cc(client, port, controller=7, value=100)

   client.dispose()

Playing a Melody
^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   import time

   def play_melody(notes, durations, tempo_bpm=120):
       """Play a simple melody."""
       client = cm.MIDIClient("Melody Player")
       port = client.create_output_port("Output")

       if cm.midi_get_number_of_destinations() == 0:
           print("No MIDI destinations available")
           client.dispose()
           return

       dest = cm.midi_get_destination(0)

       # Calculate beat duration
       beat_duration = 60.0 / tempo_bpm

       try:
           for note, duration in zip(notes, durations):
               # Note On
               port.send(dest, bytes([0x90, note, 100]))

               # Wait
               time.sleep(duration * beat_duration)

               # Note Off
               port.send(dest, bytes([0x80, note, 0]))

       finally:
           client.dispose()

   # Play "Twinkle Twinkle Little Star"
   notes = [60, 60, 67, 67, 69, 69, 67,  # C C G G A A G
            65, 65, 64, 64, 62, 62, 60]  # F F E E D D C
   durations = [1, 1, 1, 1, 1, 1, 2,
                1, 1, 1, 1, 1, 1, 2]

   play_melody(notes, durations, tempo_bpm=100)

Receiving MIDI Messages
-----------------------

Creating an Input Port
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm

   def midi_callback(packet_list, src_conn_ref):
       """Callback for incoming MIDI data."""
       print(f"Received MIDI from connection {src_conn_ref}")
       # Process packets...

   client = cm.MIDIClient("MIDI Receiver")
   input_port = client.create_input_port("Input", callback=midi_callback)

   # Connect to all sources
   for i in range(cm.midi_get_number_of_sources()):
       source = cm.midi_get_source(i)
       input_port.connect_source(source)

Simple MIDI Monitor
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import coremusic as cm
   import time

   class MIDIMonitor:
       """Monitor and display incoming MIDI messages."""

       def __init__(self):
           self.client = cm.MIDIClient("MIDI Monitor")
           self.running = True

       def parse_message(self, data):
           """Parse MIDI message bytes."""
           if len(data) == 0:
               return None

           status = data[0]
           channel = status & 0x0F
           msg_type = status & 0xF0

           if msg_type == 0x90 and len(data) >= 3:
               # Note On
               note, velocity = data[1], data[2]
               if velocity > 0:
                   return f"Note On  ch={channel} note={note} vel={velocity}"
               else:
                   return f"Note Off ch={channel} note={note}"

           elif msg_type == 0x80 and len(data) >= 3:
               # Note Off
               note = data[1]
               return f"Note Off ch={channel} note={note}"

           elif msg_type == 0xB0 and len(data) >= 3:
               # Control Change
               cc, value = data[1], data[2]
               return f"CC       ch={channel} cc={cc} val={value}"

           elif msg_type == 0xC0 and len(data) >= 2:
               # Program Change
               program = data[1]
               return f"Program  ch={channel} prog={program}"

           elif msg_type == 0xE0 and len(data) >= 3:
               # Pitch Bend
               lsb, msb = data[1], data[2]
               value = (msb << 7) | lsb
               return f"PitchBnd ch={channel} val={value}"

           else:
               return f"Unknown  {' '.join(f'{b:02X}' for b in data)}"

       def callback(self, packet_list, src_conn_ref):
           """Handle incoming MIDI."""
           for packet in packet_list:
               msg = self.parse_message(packet.data)
               if msg:
                   print(msg)

       def start(self):
           """Start monitoring."""
           input_port = self.client.create_input_port(
               "Monitor Input",
               callback=self.callback
           )

           # Connect to all sources
           num_sources = cm.midi_get_number_of_sources()
           print(f"Monitoring {num_sources} MIDI sources...")
           print("Press Ctrl+C to stop\n")

           for i in range(num_sources):
               source = cm.midi_get_source(i)
               input_port.connect_source(source)

           try:
               while self.running:
                   time.sleep(0.1)
           except KeyboardInterrupt:
               print("\nStopping...")

           self.client.dispose()

   # Run monitor
   monitor = MIDIMonitor()
   monitor.start()

Using the CLI
^^^^^^^^^^^^^

.. code-block:: bash

   # Monitor MIDI input
   coremusic midi input monitor

   # Monitor specific source
   coremusic midi input monitor 0

MIDI Message Reference
----------------------

Note Messages
^^^^^^^^^^^^^

.. code-block:: python

   # Note On: 0x90 + channel, note, velocity
   note_on = bytes([0x90, 60, 100])   # Middle C, velocity 100

   # Note Off: 0x80 + channel, note, velocity
   note_off = bytes([0x80, 60, 0])    # Middle C off

   # Note numbers: 0-127
   # Middle C (C4) = 60
   # A440 = 69

Control Change
^^^^^^^^^^^^^^

.. code-block:: python

   # CC: 0xB0 + channel, controller, value
   modulation = bytes([0xB0, 1, 64])   # Mod wheel to 50%
   volume = bytes([0xB0, 7, 100])      # Volume to 100
   pan = bytes([0xB0, 10, 64])         # Pan center
   sustain_on = bytes([0xB0, 64, 127]) # Sustain on
   sustain_off = bytes([0xB0, 64, 0])  # Sustain off
   all_off = bytes([0xB0, 123, 0])     # All Notes Off

Program Change
^^^^^^^^^^^^^^

.. code-block:: python

   # Program Change: 0xC0 + channel, program
   piano = bytes([0xC0, 0])     # Program 0 (Piano)
   strings = bytes([0xC0, 48])  # Program 48 (Strings)

Pitch Bend
^^^^^^^^^^

.. code-block:: python

   # Pitch Bend: 0xE0 + channel, LSB, MSB
   # Value range: 0-16383, center = 8192

   center = 8192
   bend_up = bytes([0xE0, center & 0x7F, (center >> 7) & 0x7F])

   max_up = 16383
   bend_max = bytes([0xE0, max_up & 0x7F, (max_up >> 7) & 0x7F])

Complete Example: MIDI Keyboard
-------------------------------

A simple MIDI keyboard using computer keys:

.. code-block:: python

   import coremusic as cm
   import sys
   import termios
   import tty

   class MIDIKeyboard:
       """Computer keyboard to MIDI converter."""

       # Map computer keys to MIDI notes
       KEY_MAP = {
           'a': 60,  # C4
           'w': 61,  # C#4
           's': 62,  # D4
           'e': 63,  # D#4
           'd': 64,  # E4
           'f': 65,  # F4
           't': 66,  # F#4
           'g': 67,  # G4
           'y': 68,  # G#4
           'h': 69,  # A4
           'u': 70,  # A#4
           'j': 71,  # B4
           'k': 72,  # C5
       }

       def __init__(self):
           self.client = cm.MIDIClient("MIDI Keyboard")
           self.port = self.client.create_output_port("Output")
           self.active_notes = set()

       def get_destination(self):
           """Get first MIDI destination."""
           if cm.midi_get_number_of_destinations() == 0:
               return None
           return cm.midi_get_destination(0)

       def note_on(self, note, velocity=100):
           """Send Note On."""
           dest = self.get_destination()
           if dest and note not in self.active_notes:
               self.port.send(dest, bytes([0x90, note, velocity]))
               self.active_notes.add(note)
               print(f"Note On: {note}")

       def note_off(self, note):
           """Send Note Off."""
           dest = self.get_destination()
           if dest and note in self.active_notes:
               self.port.send(dest, bytes([0x80, note, 0]))
               self.active_notes.discard(note)
               print(f"Note Off: {note}")

       def all_notes_off(self):
           """Turn off all active notes."""
           dest = self.get_destination()
           if dest:
               for note in list(self.active_notes):
                   self.port.send(dest, bytes([0x80, note, 0]))
               self.active_notes.clear()

       def run(self):
           """Run keyboard input loop."""
           print("MIDI Keyboard")
           print("=" * 40)
           print("Keys: A-S-D-F-G-H-J-K = C-D-E-F-G-A-B-C")
           print("Black keys: W-E-T-Y-U")
           print("Press 'q' to quit")
           print()

           # Set terminal to raw mode
           old_settings = termios.tcgetattr(sys.stdin)

           try:
               tty.setraw(sys.stdin.fileno())

               while True:
                   char = sys.stdin.read(1).lower()

                   if char == 'q':
                       break

                   if char in self.KEY_MAP:
                       note = self.KEY_MAP[char]
                       self.note_on(note)

           except KeyboardInterrupt:
               pass

           finally:
               # Restore terminal
               termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
               self.all_notes_off()
               self.client.dispose()
               print("\nGoodbye!")

   # Run keyboard
   keyboard = MIDIKeyboard()
   keyboard.run()

Troubleshooting
---------------

No MIDI Devices Found
^^^^^^^^^^^^^^^^^^^^^

1. Check Audio MIDI Setup.app for device visibility
2. Ensure MIDI devices are connected and powered on
3. Try unplugging and reconnecting USB MIDI devices
4. Check for driver requirements

Messages Not Received
^^^^^^^^^^^^^^^^^^^^^

1. Verify source is connected to input port
2. Check device is sending on expected channel
3. Use MIDI Monitor to verify messages

Messages Not Sending
^^^^^^^^^^^^^^^^^^^^

1. Verify destination exists
2. Check receiving device/software is listening
3. Try sending to different destination

Next Steps
----------

- :doc:`midi_transform` - Transform and process MIDI
- :doc:`../cookbook/midi_processing` - MIDI processing recipes
- :doc:`../cookbook/link_integration` - Sync MIDI with Ableton Link

See Also
--------

- :doc:`../api/index` - Complete API reference
- :doc:`../guides/cli` - CLI MIDI commands
