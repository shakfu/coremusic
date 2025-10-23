#!/usr/bin/env python3
"""Demo: OSStatus Error Translation - Human-Readable Error Messages

This demo shows how the new os_status module provides human-readable error
messages with actionable recovery suggestions for CoreAudio errors.
"""

import coremusic as cm
from coremusic import os_status


def main():
    print("=" * 70)
    print("CoreAudio OSStatus Error Translation Demo")
    print("=" * 70)
    print()

    # Demo 1: Basic error translation
    print("Demo 1: Basic Error Translation")
    print("-" * 70)
    error_codes = [
        (-43, "File not found"),
        (-50, "Invalid parameter"),
        (-128, "User canceled/Security restriction"),
        (0x7479703F, "Unsupported file type (FourCC 'typ?')"),
        (-10875, "AudioUnit invalid property"),
        (-66687, "AudioQueue invalid device"),
    ]

    for code, context in error_codes:
        translation = os_status.os_status_to_string(code)
        print(f"\n{context}:")
        print(f"  Status Code: {code}")
        print(f"  Translation: {translation}")

    # Demo 2: Error suggestions
    print("\n\n" + "=" * 70)
    print("Demo 2: Recovery Suggestions")
    print("-" * 70)

    error_codes_with_suggestions = [
        (-43, "open audio file"),
        (0x70726D3F, "write to file"),  # 'prm?' - permissions
        (-10879, "initialize AudioUnit"),
        (-66687, "create audio queue"),
    ]

    for code, operation in error_codes_with_suggestions:
        error_str = os_status.os_status_to_string(code)
        suggestion = os_status.get_error_suggestion(code)

        print(f"\nOperation: {operation}")
        print(f"  Error: {error_str}")
        if suggestion:
            print(f"  💡 Suggestion: {suggestion}")
        else:
            print(f"  (No specific suggestion available)")

    # Demo 3: Formatted error messages
    print("\n\n" + "=" * 70)
    print("Demo 3: Complete Formatted Error Messages")
    print("-" * 70)

    operations = [
        (-43, "open audio file '/path/to/missing.wav'"),
        (-50, "create audio queue with invalid format"),
        (0x7479703F, "parse unknown audio format"),
    ]

    for code, operation in operations:
        formatted = os_status.format_os_status_error(code, operation)
        print(f"\n{formatted}")

    # Demo 4: Using with exceptions
    print("\n\n" + "=" * 70)
    print("Demo 4: Enhanced Exception Classes")
    print("-" * 70)

    print("\nBEFORE (old style):")
    print('  raise RuntimeError(f"AudioFileOpenURL failed with status: {-43}")')
    print("  RuntimeError: AudioFileOpenURL failed with status: -43")

    print("\nAFTER (new style):")
    print('  exc = cm.AudioFileError.from_os_status(-43, "open audio file")')
    exc = cm.AudioFileError.from_os_status(-43, "open audio file")
    print(f"  {exc.__class__.__name__}: {exc}")
    print(f"  Status code preserved: exc.status_code = {exc.status_code}")

    # Demo 5: Error info tuple
    print("\n\n" + "=" * 70)
    print("Demo 5: Structured Error Information")
    print("-" * 70)

    error_codes = [-43, -10875, -66687]

    for code in error_codes:
        name, description, suggestion = os_status.get_error_info(code)
        print(f"\nStatus {code}:")
        print(f"  Error Name: {name}")
        print(f"  Description: {description}")
        if suggestion:
            print(f"  Suggestion: {suggestion}")

    # Demo 6: Real-world scenario
    print("\n\n" + "=" * 70)
    print("Demo 6: Real-World Error Handling")
    print("-" * 70)

    print("\nScenario: Trying to open a non-existent audio file")
    print()

    try:
        # This will fail because the file doesn't exist
        file = cm.AudioFile("/path/to/nonexistent.wav")
        file.open()
    except cm.AudioFileError as e:
        print("OLD ERROR MESSAGE (what user would see before):")
        print("  AudioFileError: Failed to open file")
        print()
        print("NEW ERROR MESSAGE (with os_status integration):")
        print(f"  {e.__class__.__name__}: {e}")
        print(f"  Status Code: {e.status_code}")
        print()
        # Show what it would look like with the new API
        enhanced_exc = cm.AudioFileError.from_os_status(e.status_code, "open audio file")
        print("ENHANCED MESSAGE (using from_os_status):")
        print(f"  {enhanced_exc}")

    # Demo 7: All error categories
    print("\n\n" + "=" * 70)
    print("Demo 7: Error Categories Overview")
    print("-" * 70)

    categories = {
        "Hardware": [(0x73746F70, "Hardware not running"), (0x21646576, "Bad device")],
        "File I/O": [(-43, "File not found"), (0x70726D3F, "Permissions error")],
        "AudioUnit": [(-10875, "Invalid property"), (-10879, "Failed init")],
        "AudioQueue": [(-66680, "Invalid buffer"), (-66687, "Invalid device")],
        "System": [(-50, "Invalid parameter"), (-108, "Out of memory")],
    }

    for category, errors in categories.items():
        print(f"\n{category} Errors:")
        for code, _ in errors:
            translation = os_status.os_status_to_string(code)
            print(f"  {code:6d}: {translation}")

    print("\n" + "=" * 70)
    print("Summary:")
    print("  ✓ 100+ error codes from all CoreAudio frameworks")
    print("  ✓ Human-readable error names and descriptions")
    print("  ✓ 30+ actionable recovery suggestions")
    print("  ✓ Easy integration with existing exception classes")
    print("  ✓ Zero dependencies - pure Python implementation")
    print("=" * 70)


if __name__ == "__main__":
    main()
