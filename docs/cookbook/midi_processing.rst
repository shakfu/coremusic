MIDI Processing
===============

Recipes for MIDI input/output and processing with CoreMIDI.

.. contents:: Topics
   :local:
   :depth: 2

MIDI Device Discovery
---------------------

List MIDI Devices
^^^^^^^^^^^^^^^^^

Discover available MIDI sources and destinations:

.. code-block:: python

   import coremusic.capi as capi

   # List MIDI sources (input devices)
   num_sources = capi.midi_get_number_of_sources()
   print(f"MIDI Sources: {num_sources}")

   for i in range(num_sources):
       source = capi.midi_get_source(i)
       name = capi.midi_object_get_string_property(source, "name")
       print(f"  {i}: {name}")

   # List MIDI destinations (output devices)
   num_dests = capi.midi_get_number_of_destinations()
   print(f"\nMIDI Destinations: {num_dests}")

   for i in range(num_dests):
       dest = capi.midi_get_destination(i)
       name = capi.midi_object_get_string_property(dest, "name")
       print(f"  {i}: {name}")

Find Device by Name
^^^^^^^^^^^^^^^^^^^

Locate a specific MIDI device:

.. code-block:: python

   import coremusic.capi as capi

   def find_midi_source(device_name):
       """Find MIDI source by name"""
       num_sources = capi.midi_get_number_of_sources()

       for i in range(num_sources):
           source = capi.midi_get_source(i)
           name = capi.midi_object_get_string_property(source, "name")
           if device_name.lower() in name.lower():
               return source, name

       return None, None

   # Find a specific device
   source, name = find_midi_source("Keyboard")
   if source:
       print(f"Found: {name}")
   else:
       print("Device not found")

MIDI Input
----------

Receive MIDI Messages
^^^^^^^^^^^^^^^^^^^^^

Set up MIDI input and receive messages:

.. code-block:: python

   import coremusic.capi as capi
   import time

   def midi_callback(packet_list, src_conn_ref_con):
       """Handle incoming MIDI messages"""
       num_packets = capi.midi_packet_list_get_num_packets(packet_list)

       for i in range(num_packets):
           packet = capi.midi_packet_list_get_packet(packet_list, i)
           data = capi.midi_packet_get_data(packet)
           timestamp = capi.midi_packet_get_timestamp(packet)

           # Parse MIDI message
           if len(data) >= 1:
               status = data[0]
               message_type = status & 0xF0
               channel = status & 0x0F

               if message_type == 0x90 and len(data) >= 3:  # Note On
                   note = data[1]
                   velocity = data[2]
                   print(f"Note On: ch={channel}, note={note}, vel={velocity}")

               elif message_type == 0x80 and len(data) >= 3:  # Note Off
                   note = data[1]
                   velocity = data[2]
                   print(f"Note Off: ch={channel}, note={note}, vel={velocity}")

               elif message_type == 0xB0 and len(data) >= 3:  # Control Change
                   controller = data[1]
                   value = data[2]
                   print(f"CC: ch={channel}, ctrl={controller}, val={value}")

   # Create MIDI client and input port
   client = capi.midi_client_create("MIDI Input")
   input_port = capi.midi_input_port_create(client, "Input", midi_callback)

   # Connect to first MIDI source
   source = capi.midi_get_source(0)
   capi.midi_port_connect_source(input_port, source)

   print("Listening for MIDI... (Press Ctrl+C to stop)")
   try:
       while True:
           time.sleep(0.1)
   except KeyboardInterrupt:
       print("\nStopped")

   # Cleanup
   capi.midi_port_disconnect_source(input_port, source)
   capi.midi_port_dispose(input_port)
   capi.midi_client_dispose(client)

Filter MIDI Messages
^^^^^^^^^^^^^^^^^^^^

Filter specific MIDI message types:

.. code-block:: python

   import coremusic.capi as capi

   class MIDIFilter:
       def __init__(self, filter_notes=False, filter_cc=False):
           self.filter_notes = filter_notes
           self.filter_cc = filter_cc

       def callback(self, packet_list, src_conn_ref_con):
           num_packets = capi.midi_packet_list_get_num_packets(packet_list)

           for i in range(num_packets):
               packet = capi.midi_packet_list_get_packet(packet_list, i)
               data = capi.midi_packet_get_data(packet)

               if len(data) >= 1:
                   status = data[0]
                   message_type = status & 0xF0

                   # Filter note messages
                   if message_type in [0x80, 0x90] and self.filter_notes:
                       continue

                   # Filter CC messages
                   if message_type == 0xB0 and self.filter_cc:
                       continue

                   # Process remaining messages
                   print(f"MIDI: {[hex(b) for b in data]}")

   # Create filter that blocks notes but allows CC
   midi_filter = MIDIFilter(filter_notes=True, filter_cc=False)

   client = capi.midi_client_create("Filtered Input")
   input_port = capi.midi_input_port_create(client, "Input", midi_filter.callback)

   # Connect and listen...

MIDI Output
-----------

Send MIDI Messages
^^^^^^^^^^^^^^^^^^

Send MIDI messages to an output device:

.. code-block:: python

   import coremusic.capi as capi
   import time

   # Create MIDI client and output port
   client = capi.midi_client_create("MIDI Output")
   output_port = capi.midi_output_port_create(client, "Output")

   # Get first MIDI destination
   dest = capi.midi_get_destination(0)

   # Send Note On
   note_on = bytes([0x90, 60, 100])  # Channel 1, Middle C, Velocity 100
   capi.midi_send(output_port, dest, note_on)
   print("Sent Note On")

   time.sleep(1.0)

   # Send Note Off
   note_off = bytes([0x80, 60, 0])  # Channel 1, Middle C
   capi.midi_send(output_port, dest, note_off)
   print("Sent Note Off")

   # Cleanup
   capi.midi_port_dispose(output_port)
   capi.midi_client_dispose(client)

Play MIDI Sequence
^^^^^^^^^^^^^^^^^^

Send a sequence of MIDI notes:

.. code-block:: python

   import coremusic.capi as capi
   import time

   def play_note(port, dest, channel, note, velocity, duration):
       """Play a single note"""
       # Note On
       note_on = bytes([0x90 | channel, note, velocity])
       capi.midi_send(port, dest, note_on)

       # Wait
       time.sleep(duration)

       # Note Off
       note_off = bytes([0x80 | channel, note, 0])
       capi.midi_send(port, dest, note_off)

   # Setup
   client = capi.midi_client_create("Sequencer")
   output_port = capi.midi_output_port_create(client, "Output")
   dest = capi.midi_get_destination(0)

   # Play C major scale
   scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C

   for note in scale:
       play_note(output_port, dest, channel=0, note=note, velocity=100, duration=0.5)
       time.sleep(0.1)  # Gap between notes

   # Cleanup
   capi.midi_port_dispose(output_port)
   capi.midi_client_dispose(client)

Send Control Changes
^^^^^^^^^^^^^^^^^^^^

Send MIDI CC messages for automation:

.. code-block:: python

   import coremusic.capi as capi
   import time

   client = capi.midi_client_create("CC Controller")
   output_port = capi.midi_output_port_create(client, "Output")
   dest = capi.midi_get_destination(0)

   # Start a note
   note_on = bytes([0x90, 60, 100])
   capi.midi_send(output_port, dest, note_on)

   # Fade volume (CC 7) from 127 to 0
   for volume in range(127, -1, -5):
       cc = bytes([0xB0, 7, volume])  # Channel 1, CC 7 (Volume), value
       capi.midi_send(output_port, dest, cc)
       time.sleep(0.05)

   # Stop note
   note_off = bytes([0x80, 60, 0])
   capi.midi_send(output_port, dest, note_off)

   # Cleanup
   capi.midi_port_dispose(output_port)
   capi.midi_client_dispose(client)

MIDI Routing
------------

MIDI Thru
^^^^^^^^^

Route MIDI input directly to output:

.. code-block:: python

   import coremusic.capi as capi
   import time

   # Create client with input and output ports
   client = capi.midi_client_create("MIDI Thru")

   # Output port
   output_port = capi.midi_output_port_create(client, "Output")
   dest = capi.midi_get_destination(0)

   # Input callback that forwards to output
   def thru_callback(packet_list, src_conn_ref_con):
       num_packets = capi.midi_packet_list_get_num_packets(packet_list)

       for i in range(num_packets):
           packet = capi.midi_packet_list_get_packet(packet_list, i)
           data = capi.midi_packet_get_data(packet)

           # Forward to output
           capi.midi_send(output_port, dest, data)

   # Input port
   input_port = capi.midi_input_port_create(client, "Input", thru_callback)
   source = capi.midi_get_source(0)
   capi.midi_port_connect_source(input_port, source)

   print("MIDI thru active... (Press Ctrl+C to stop)")
   try:
       while True:
           time.sleep(0.1)
   except KeyboardInterrupt:
       print("\nStopped")

   # Cleanup
   capi.midi_port_disconnect_source(input_port, source)
   capi.midi_port_dispose(input_port)
   capi.midi_port_dispose(output_port)
   capi.midi_client_dispose(client)

Channel Routing
^^^^^^^^^^^^^^^

Route MIDI from one channel to another:

.. code-block:: python

   import coremusic.capi as capi

   class ChannelRouter:
       def __init__(self, output_port, dest, input_channel, output_channel):
           self.output_port = output_port
           self.dest = dest
           self.input_channel = input_channel
           self.output_channel = output_channel

       def callback(self, packet_list, src_conn_ref_con):
           num_packets = capi.midi_packet_list_get_num_packets(packet_list)

           for i in range(num_packets):
               packet = capi.midi_packet_list_get_packet(packet_list, i)
               data = list(capi.midi_packet_get_data(packet))

               if len(data) >= 1:
                   status = data[0]
                   message_type = status & 0xF0
                   channel = status & 0x0F

                   # Only process messages on input channel
                   if channel == self.input_channel:
                       # Change to output channel
                       data[0] = message_type | self.output_channel

                       # Forward modified message
                       capi.midi_send(self.output_port, self.dest, bytes(data))

   # Route channel 0 → channel 1
   client = capi.midi_client_create("Channel Router")
   output_port = capi.midi_output_port_create(client, "Output")
   dest = capi.midi_get_destination(0)

   router = ChannelRouter(output_port, dest, input_channel=0, output_channel=1)

   input_port = capi.midi_input_port_create(client, "Input", router.callback)
   source = capi.midi_get_source(0)
   capi.midi_port_connect_source(input_port, source)

   # Let it run...

MIDI Transformation
-------------------

Transpose Notes
^^^^^^^^^^^^^^^

Transpose all incoming notes:

.. code-block:: python

   import coremusic.capi as capi

   class Transposer:
       def __init__(self, output_port, dest, semitones):
           self.output_port = output_port
           self.dest = dest
           self.semitones = semitones

       def callback(self, packet_list, src_conn_ref_con):
           num_packets = capi.midi_packet_list_get_num_packets(packet_list)

           for i in range(num_packets):
               packet = capi.midi_packet_list_get_packet(packet_list, i)
               data = list(capi.midi_packet_get_data(packet))

               if len(data) >= 3:
                   status = data[0]
                   message_type = status & 0xF0

                   # Transpose note on/off messages
                   if message_type in [0x80, 0x90]:  # Note On/Off
                       original_note = data[1]
                       transposed_note = max(0, min(127, original_note + self.semitones))
                       data[1] = transposed_note

                       print(f"Transposed: {original_note} → {transposed_note}")

                   # Forward modified message
                   capi.midi_send(self.output_port, self.dest, bytes(data))

   # Transpose up one octave
   client = capi.midi_client_create("Transposer")
   output_port = capi.midi_output_port_create(client, "Output")
   dest = capi.midi_get_destination(0)

   transposer = Transposer(output_port, dest, semitones=12)

   input_port = capi.midi_input_port_create(client, "Input", transposer.callback)
   source = capi.midi_get_source(0)
   capi.midi_port_connect_source(input_port, source)

   # Let it run...

Velocity Scaling
^^^^^^^^^^^^^^^^

Scale note velocities:

.. code-block:: python

   import coremusic.capi as capi

   class VelocityScaler:
       def __init__(self, output_port, dest, scale_factor):
           self.output_port = output_port
           self.dest = dest
           self.scale_factor = scale_factor

       def callback(self, packet_list, src_conn_ref_con):
           num_packets = capi.midi_packet_list_get_num_packets(packet_list)

           for i in range(num_packets):
               packet = capi.midi_packet_list_get_packet(packet_list, i)
               data = list(capi.midi_packet_get_data(packet))

               if len(data) >= 3:
                   status = data[0]
                   message_type = status & 0xF0

                   if message_type == 0x90:  # Note On
                       original_vel = data[2]
                       scaled_vel = int(original_vel * self.scale_factor)
                       scaled_vel = max(1, min(127, scaled_vel))  # Clamp to 1-127
                       data[2] = scaled_vel

                       print(f"Velocity: {original_vel} → {scaled_vel}")

                   # Forward message
                   capi.midi_send(self.output_port, self.dest, bytes(data))

   # Scale velocities to 80% (softer)
   scaler = VelocityScaler(output_port, dest, scale_factor=0.8)

   # Setup and run...

MIDI Recording
--------------

Record MIDI Messages
^^^^^^^^^^^^^^^^^^^^

Record MIDI to a list with timestamps:

.. code-block:: python

   import coremusic.capi as capi
   import time

   class MIDIRecorder:
       def __init__(self):
           self.recording = False
           self.start_time = None
           self.recorded_messages = []

       def start(self):
           self.recording = True
           self.start_time = time.time()
           self.recorded_messages = []
           print("Recording started")

       def stop(self):
           self.recording = False
           print(f"Recording stopped: {len(self.recorded_messages)} messages")

       def callback(self, packet_list, src_conn_ref_con):
           if not self.recording:
               return

           current_time = time.time() - self.start_time
           num_packets = capi.midi_packet_list_get_num_packets(packet_list)

           for i in range(num_packets):
               packet = capi.midi_packet_list_get_packet(packet_list, i)
               data = capi.midi_packet_get_data(packet)

               self.recorded_messages.append({
                   'time': current_time,
                   'data': bytes(data)
               })

       def save(self, filename):
           """Save recorded messages to file"""
           import json

           with open(filename, 'w') as f:
               messages = [
                   {'time': msg['time'], 'data': list(msg['data'])}
                   for msg in self.recorded_messages
               ]
               json.dump(messages, f, indent=2)

           print(f"Saved to {filename}")

   # Setup recorder
   recorder = MIDIRecorder()

   client = capi.midi_client_create("Recorder")
   input_port = capi.midi_input_port_create(client, "Input", recorder.callback)
   source = capi.midi_get_source(0)
   capi.midi_port_connect_source(input_port, source)

   # Record for 10 seconds
   recorder.start()
   time.sleep(10)
   recorder.stop()

   # Save recording
   recorder.save("recorded_midi.json")

   # Cleanup
   capi.midi_port_disconnect_source(input_port, source)
   capi.midi_port_dispose(input_port)
   capi.midi_client_dispose(client)

Playback Recorded MIDI
^^^^^^^^^^^^^^^^^^^^^^^

Play back recorded MIDI messages:

.. code-block:: python

   import coremusic.capi as capi
   import json
   import time

   def playback_midi(filename, output_port, dest):
       """Play back recorded MIDI"""
       # Load recorded messages
       with open(filename, 'r') as f:
           messages = json.load(f)

       if not messages:
           print("No messages to play")
           return

       print(f"Playing back {len(messages)} messages...")
       start_time = time.time()

       for msg in messages:
           # Wait until scheduled time
           target_time = start_time + msg['time']
           wait_time = target_time - time.time()

           if wait_time > 0:
               time.sleep(wait_time)

           # Send message
           data = bytes(msg['data'])
           capi.midi_send(output_port, dest, data)

       print("Playback complete")

   # Setup playback
   client = capi.midi_client_create("Playback")
   output_port = capi.midi_output_port_create(client, "Output")
   dest = capi.midi_get_destination(0)

   # Play recording
   playback_midi("recorded_midi.json", output_port, dest)

   # Cleanup
   capi.midi_port_dispose(output_port)
   capi.midi_client_dispose(client)

Complete Example: MIDI Monitor
-------------------------------

Full-featured MIDI monitor with message parsing:

.. code-block:: python

   import coremusic.capi as capi
   import time

   class MIDIMonitor:
       def __init__(self):
           self.message_count = 0

       def parse_message(self, data):
           """Parse and format MIDI message"""
           if len(data) == 0:
               return "Empty message"

           status = data[0]
           message_type = status & 0xF0
           channel = (status & 0x0F) + 1

           if message_type == 0x80:  # Note Off
               return f"Note Off  | Ch {channel:2d} | Note {data[1]:3d} | Vel {data[2]:3d}"
           elif message_type == 0x90:  # Note On
               if data[2] == 0:  # Velocity 0 = Note Off
                   return f"Note Off  | Ch {channel:2d} | Note {data[1]:3d} | Vel {data[2]:3d}"
               return f"Note On   | Ch {channel:2d} | Note {data[1]:3d} | Vel {data[2]:3d}"
           elif message_type == 0xA0:  # Poly Aftertouch
               return f"Poly AT   | Ch {channel:2d} | Note {data[1]:3d} | Pressure {data[2]:3d}"
           elif message_type == 0xB0:  # Control Change
               return f"CC        | Ch {channel:2d} | Ctrl {data[1]:3d} | Val {data[2]:3d}"
           elif message_type == 0xC0:  # Program Change
               return f"Program   | Ch {channel:2d} | Program {data[1]:3d}"
           elif message_type == 0xD0:  # Channel Aftertouch
               return f"Channel AT| Ch {channel:2d} | Pressure {data[1]:3d}"
           elif message_type == 0xE0:  # Pitch Bend
               value = data[1] + (data[2] << 7)
               return f"Pitch Bend| Ch {channel:2d} | Value {value:5d}"
           elif status == 0xF8:  # Clock
               return "MIDI Clock"
           elif status == 0xFA:  # Start
               return "MIDI Start"
           elif status == 0xFB:  # Continue
               return "MIDI Continue"
           elif status == 0xFC:  # Stop
               return "MIDI Stop"
           else:
               hex_data = ' '.join(f'{b:02X}' for b in data)
               return f"Unknown   | {hex_data}"

       def callback(self, packet_list, src_conn_ref_con):
           num_packets = capi.midi_packet_list_get_num_packets(packet_list)

           for i in range(num_packets):
               packet = capi.midi_packet_list_get_packet(packet_list, i)
               data = capi.midi_packet_get_data(packet)

               self.message_count += 1
               message = self.parse_message(data)
               print(f"[{self.message_count:6d}] {message}")

   # Setup monitor
   monitor = MIDIMonitor()

   client = capi.midi_client_create("MIDI Monitor")
   input_port = capi.midi_input_port_create(client, "Monitor Input", monitor.callback)

   # Connect to all MIDI sources
   num_sources = capi.midi_get_number_of_sources()
   print(f"Monitoring {num_sources} MIDI source(s)\n")

   for i in range(num_sources):
       source = capi.midi_get_source(i)
       capi.midi_port_connect_source(input_port, source)
       name = capi.midi_object_get_string_property(source, "name")
       print(f"Connected to: {name}")

   print("\nMIDI Monitor - Press Ctrl+C to stop")
   print("-" * 70)

   try:
       while True:
           time.sleep(0.1)
   except KeyboardInterrupt:
       print(f"\n\nStopped - Received {monitor.message_count} messages")

   # Cleanup
   for i in range(num_sources):
       source = capi.midi_get_source(i)
       capi.midi_port_disconnect_source(input_port, source)

   capi.midi_port_dispose(input_port)
   capi.midi_client_dispose(client)

Best Practices
--------------

Resource Management
^^^^^^^^^^^^^^^^^^^

Always dispose of MIDI resources:

.. code-block:: python

   # Create resources
   client = capi.midi_client_create("App")
   port = capi.midi_output_port_create(client, "Out")

   try:
       # Use resources
       pass
   finally:
       # Always cleanup
       capi.midi_port_dispose(port)
       capi.midi_client_dispose(client)

Error Handling
^^^^^^^^^^^^^^

Handle MIDI errors gracefully:

.. code-block:: python

   try:
       dest = capi.midi_get_destination(0)
   except IndexError:
       print("No MIDI destinations available")
       return

   try:
       capi.midi_send(port, dest, data)
   except Exception as e:
       print(f"Failed to send MIDI: {e}")

Timing Precision
^^^^^^^^^^^^^^^^

Use timestamps for accurate timing:

.. code-block:: python

   # Get current host time for precise scheduling
   timestamp = capi.midi_get_current_time()

   # Schedule MIDI message with timestamp
   # (Note: Requires using MIDIPacketList directly)

Thread Safety
^^^^^^^^^^^^^

MIDI callbacks run on separate threads - use thread-safe operations:

.. code-block:: python

   import threading

   class ThreadSafeMIDIProcessor:
       def __init__(self):
           self.lock = threading.Lock()
           self.buffer = []

       def callback(self, packet_list, src_conn_ref_con):
           with self.lock:
               # Process MIDI safely
               pass

See Also
--------

- :doc:`/api/index` - Complete API reference
- :doc:`audiounit_hosting` - AudioUnit plugin hosting (instruments)
- :doc:`link_integration` - Ableton Link tempo sync
- CoreMIDI documentation: https://developer.apple.com/documentation/coremidi
