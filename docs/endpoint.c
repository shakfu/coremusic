// Assuming 'endpoint' is a valid MIDIEndpointRef
CFStringRef displayNameRef = NULL;
OSStatus status = MIDIObjectGetStringProperty(endpoint, kMIDIPropertyDisplayName, &displayNameRef);

if (status == noErr && displayNameRef != NULL) {
    // Convert CFStringRef to a C string for use
    const char* displayName = CFStringGetCStringPtr(displayNameRef, kCFStringEncodingUTF8);
    if (displayName != NULL) {
        printf("MIDI Endpoint Display Name: %s\n", displayName);
    } else {
        // Handle cases where CFStringGetCStringPtr returns NULL (e.g., non-ASCII characters)
        // You would typically use CFStringGetCString to copy the string into a buffer.
        CFIndex length = CFStringGetLength(displayNameRef);
        CFIndex maxSize = CFStringGetMaximumSizeForEncoding(length, kCFStringEncodingUTF8) + 1;
        char* buffer = (char*)malloc(maxSize);
        if (CFStringGetCString(displayNameRef, buffer, maxSize, kCFStringEncodingUTF8)) {
            printf("MIDI Endpoint Display Name: %s\n", buffer);
        }
        free(buffer);
    }
    CFRelease(displayNameRef); // Release the CFStringRef
} else {
    // Handle error
    fprintf(stderr, "Error getting display name: %d\n", (int)status);
}
