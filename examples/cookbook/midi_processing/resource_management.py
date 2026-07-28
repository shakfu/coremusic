#!/usr/bin/env python3
"""Resource Management."""

# --8<-- [start:example]
from coremusic import capi

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
# --8<-- [end:example]
