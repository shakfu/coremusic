#!/usr/bin/env python3
"""Find AudioUnits by name.

Usage:
    python find_by_name.py [name]
"""

import sys
import coremusic.capi as capi


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "AUDelay"

    print(f"Searching for: {name}")
    codes = capi.find_audio_unit_by_name(name)

    if codes:
        type_code, subtype_code, manufacturer = codes
        print(f"Found: type={type_code}, subtype={subtype_code}, mfr={manufacturer}")
    else:
        print("Not found")

        # Show available units
        print("\nAvailable AudioUnits:")
        units = capi.list_available_audio_units()
        for unit in units[:10]:
            print(f"  {unit['name']} ({unit['type']}/{unit['subtype']})")
        if len(units) > 10:
            print(f"  ... and {len(units) - 10} more")


if __name__ == "__main__":
    main()
