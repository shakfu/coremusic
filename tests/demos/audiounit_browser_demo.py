#!/usr/bin/env python3
"""AudioUnit Browser Demo

Interactive demo showcasing AudioUnit plugin discovery and parameter control.
Demonstrates:
- Plugin discovery and enumeration
- Plugin information retrieval
- Parameter listing and manipulation
- Factory preset browsing
"""

import sys
import coremusic.capi as capi


def list_plugins_by_category():
    """List all AudioUnit plugins organized by category"""
    print("\n" + "=" * 70)
    print(" AudioUnit Plugin Browser")
    print("=" * 70)

    categories = {
        'Effects': 'aufx',
        'Instruments': 'aumu',
        'Generators': 'augn',
        'Mixers': 'aumx',
        'Output': 'auou',
    }

    total_plugins = 0

    for category_name, category_type in categories.items():
        components = capi.audio_unit_find_all_components(component_type=category_type)
        count = len(components)
        total_plugins += count

        print(f"\n{category_name}: {count} plugins")
        print("-" * 70)

        # Show first 10 plugins in each category
        for i, comp_id in enumerate(components[:10]):
            try:
                info = capi.audio_unit_get_component_info(comp_id)
                print(f"  {i+1:2d}. {info['name']:<40} ({info['manufacturer']})")
            except Exception as e:
                print(f"  {i+1:2d}. [Error getting info: {e}]")

        if count > 10:
            print(f"      ... and {count - 10} more")

    print(f"\nTotal: {total_plugins} AudioUnit plugins found on this system")
    return total_plugins


def browse_plugin_details():
    """Browse detailed information about a specific plugin"""
    print("\n" + "=" * 70)
    print(" Plugin Detail Browser")
    print("=" * 70)

    # Get all effects
    components = capi.audio_unit_find_all_components(component_type='aufx')

    if len(components) == 0:
        print("\nNo effect plugins found!")
        return

    print(f"\nFound {len(components)} effect plugins")
    print("\nShowing first 20:")
    print("-" * 70)

    # List first 20
    for i, comp_id in enumerate(components[:20]):
        try:
            info = capi.audio_unit_get_component_info(comp_id)
            print(f"  [{i}] {info['name']}")
        except Exception as e:
            print(f"  [{i}] Error: {e}")

    # Let user select a plugin
    print()
    choice = input(f"Select plugin to examine (0-{min(19, len(components)-1)}) or 'q' to quit: ")

    if choice.lower() == 'q':
        return

    try:
        index = int(choice)
        if index < 0 or index >= len(components):
            print("Invalid selection")
            return
    except ValueError:
        print("Invalid input")
        return

    # Get selected component
    comp_id = components[index]
    info = capi.audio_unit_get_component_info(comp_id)

    print("\n" + "=" * 70)
    print(f" Plugin: {info['name']}")
    print("=" * 70)

    print(f"\nBasic Information:")
    print(f"  Name: {info['name']}")
    print(f"  Type: {info['type']}")
    print(f"  Subtype: {info['subtype']}")
    print(f"  Manufacturer: {info['manufacturer']}")
    print(f"  Version: {info['version']}")

    # Try to instantiate and get parameters
    print("\nInstantiating plugin...")
    try:
        unit_id = capi.audio_component_instance_new(comp_id)
        print("  ✓ Instance created")

        print("  Initializing...")
        capi.audio_unit_initialize(unit_id)
        print("  ✓ Initialized")

        # Get parameters
        print("\nQuerying parameters...")
        params = capi.audio_unit_get_parameter_list(unit_id)
        print(f"  Found {len(params)} parameters")

        if len(params) > 0:
            print("\nParameters:")
            print("-" * 70)

            for param_id in params[:20]:  # Show first 20
                try:
                    param_info = capi.audio_unit_get_parameter_info(unit_id, param_id)
                    value = capi.audio_unit_get_parameter(unit_id, param_id)

                    unit_str = f" {param_info['unit_name']}" if param_info['unit_name'] else ""
                    print(f"\n  {param_info['name']}:")
                    print(f"    ID: {param_id}")
                    print(f"    Current: {value:.3f}{unit_str}")
                    print(f"    Range: {param_info['min_value']:.3f} - {param_info['max_value']:.3f}")
                    print(f"    Default: {param_info['default_value']:.3f}")

                except Exception as e:
                    print(f"\n  Parameter {param_id}: Error - {e}")

            if len(params) > 20:
                print(f"\n  ... and {len(params) - 20} more parameters")

        # Get factory presets
        print("\nQuerying factory presets...")
        presets = capi.audio_unit_get_factory_presets(unit_id)
        print(f"  Found {len(presets)} factory presets")

        if len(presets) > 0:
            print("\nFactory Presets:")
            print("-" * 70)
            for preset in presets[:10]:  # Show first 10
                print(f"  [{preset['number']}] {preset['name']}")

            if len(presets) > 10:
                print(f"  ... and {len(presets) - 10} more presets")

        # Cleanup
        print("\nCleaning up...")
        capi.audio_unit_uninitialize(unit_id)
        capi.audio_component_instance_dispose(unit_id)
        print("  ✓ Cleanup complete")

    except Exception as e:
        print(f"  ✗ Error: {e}")


def interactive_parameter_control():
    """Interactive parameter control demo"""
    print("\n" + "=" * 70)
    print(" Interactive Parameter Control")
    print("=" * 70)

    # Find Apple's effects (most reliable)
    components = capi.audio_unit_find_all_components(
        component_type='aufx',
        manufacturer='appl'
    )

    if len(components) == 0:
        print("\nNo Apple effect plugins found!")
        return

    print(f"\nApple Effect Plugins ({len(components)}):")
    print("-" * 70)

    for i, comp_id in enumerate(components):
        try:
            info = capi.audio_unit_get_component_info(comp_id)
            print(f"  [{i}] {info['name']}")
        except Exception as e:
            print(f"  [{i}] Error: {e}")

    # Let user select
    print()
    choice = input(f"Select plugin (0-{len(components)-1}) or 'q' to quit: ")

    if choice.lower() == 'q':
        return

    try:
        index = int(choice)
        if index < 0 or index >= len(components):
            print("Invalid selection")
            return
    except ValueError:
        print("Invalid input")
        return

    # Create instance
    comp_id = components[index]
    info = capi.audio_unit_get_component_info(comp_id)

    print(f"\nLoading {info['name']}...")

    try:
        unit_id = capi.audio_component_instance_new(comp_id)
        capi.audio_unit_initialize(unit_id)
        print("  ✓ Plugin loaded")

        params = capi.audio_unit_get_parameter_list(unit_id)

        if len(params) == 0:
            print("\nThis plugin has no parameters to control")
            capi.audio_unit_uninitialize(unit_id)
            capi.audio_component_instance_dispose(unit_id)
            return

        print(f"\nParameters ({len(params)}):")
        print("-" * 70)

        param_infos = []
        for param_id in params:
            try:
                param_info = capi.audio_unit_get_parameter_info(unit_id, param_id)
                value = capi.audio_unit_get_parameter(unit_id, param_id)
                param_infos.append((param_id, param_info, value))
                print(f"  [{len(param_infos)-1}] {param_info['name']}: {value:.3f}")
            except:
                pass

        # Interactive control loop
        while True:
            print("\nCommands:")
            print("  [number] - Select parameter to modify")
            print("  [q] - Quit")

            cmd = input("\nCommand: ").strip().lower()

            if cmd == 'q':
                break

            try:
                param_index = int(cmd)
                if param_index < 0 or param_index >= len(param_infos):
                    print("Invalid parameter index")
                    continue

                param_id, param_info, _ = param_infos[param_index]

                print(f"\nParameter: {param_info['name']}")
                print(f"  Current: {capi.audio_unit_get_parameter(unit_id, param_id):.3f}")
                print(f"  Range: {param_info['min_value']:.3f} - {param_info['max_value']:.3f}")

                new_value = input(f"New value: ").strip()
                try:
                    value = float(new_value)
                    capi.audio_unit_set_parameter(unit_id, param_id, value)
                    print(f"  ✓ Set to {value:.3f}")
                except ValueError:
                    print("  ✗ Invalid value")

            except ValueError:
                print("Invalid command")

        # Cleanup
        capi.audio_unit_uninitialize(unit_id)
        capi.audio_component_instance_dispose(unit_id)
        print("\n✓ Plugin unloaded")

    except Exception as e:
        print(f"  ✗ Error: {e}")


def main():
    """Main demo menu"""
    print("\n" + "=" * 70)
    print(" CoreMusic AudioUnit Browser")
    print("=" * 70)
    print("\nThis demo showcases AudioUnit plugin hosting capabilities:")
    print("  • Plugin discovery and enumeration")
    print("  • Plugin information retrieval")
    print("  • Parameter discovery and control")
    print("  • Factory preset browsing")

    while True:
        print("\n" + "=" * 70)
        print(" Main Menu")
        print("=" * 70)
        print("\n[1] List all plugins by category")
        print("[2] Browse plugin details")
        print("[3] Interactive parameter control")
        print("[q] Quit")

        choice = input("\nChoice: ").strip().lower()

        if choice == '1':
            list_plugins_by_category()
        elif choice == '2':
            browse_plugin_details()
        elif choice == '3':
            interactive_parameter_control()
        elif choice == 'q':
            break
        else:
            print("Invalid choice")

    print("\n" + "=" * 70)
    print(" Demo Complete!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Plugin discovery across all categories")
    print("  ✓ Plugin metadata retrieval")
    print("  ✓ Parameter enumeration and info")
    print("  ✓ Real-time parameter control")
    print("  ✓ Factory preset browsing")
    print("\nNext steps:")
    print("  • Audio routing and processing")
    print("  • Preset save/load")
    print("  • Link tempo synchronization")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(0)
