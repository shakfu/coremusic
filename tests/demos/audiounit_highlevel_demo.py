#!/usr/bin/env python3
"""AudioUnit High-Level API Demo

Demonstrates the Pythonic object-oriented API for AudioUnit plugin hosting.

Features:
- Simple plugin discovery and loading
- Automatic resource management with context managers
- Dictionary-style parameter access
- Clean, intuitive API
"""

import sys
import coremusic as cm


def demo_plugin_discovery():
    """Demo 1: Plugin Discovery"""
    print("\n" + "=" * 70)
    print(" Demo 1: Plugin Discovery")
    print("=" * 70)

    # Create host
    host = cm.AudioUnitHost()
    print(f"\nInitialized: {host}")

    # Get plugin counts
    counts = host.get_plugin_count()
    print(f"\nPlugin inventory:")
    for plugin_type, count in counts.items():
        print(f"  {plugin_type.capitalize()}: {count}")

    # Discover effects
    print("\nDiscovering Apple effects...")
    effects = host.discover_plugins(type='effect', manufacturer='appl')
    print(f"Found {len(effects)} Apple effect plugins:")
    for plugin in effects[:10]:
        print(f"  • {plugin['name']}")
    if len(effects) > 10:
        print(f"  ... and {len(effects) - 10} more")

    # Discover instruments
    print("\nDiscovering instruments...")
    instruments = host.discover_plugins(type='instrument')
    print(f"Found {len(instruments)} instrument plugins:")
    for plugin in instruments[:5]:
        print(f"  • {plugin['name']} ({plugin['manufacturer']})")
    if len(instruments) > 5:
        print(f"  ... and {len(instruments) - 5} more")

    print("\n✓ Demo complete")


def demo_plugin_loading():
    """Demo 2: Plugin Loading and Context Managers"""
    print("\n" + "=" * 70)
    print(" Demo 2: Plugin Loading")
    print("=" * 70)

    host = cm.AudioUnitHost()

    # Load plugin with context manager (automatic cleanup)
    print("\nLoading Apple AUBandpass...")
    with host.load_plugin("Bandpass", type='effect') as plugin:
        print(f"\n✓ Loaded: {plugin}")
        print(f"  Name: {plugin.name}")
        print(f"  Manufacturer: {plugin.manufacturer}")
        print(f"  Type: {plugin.type}")
        print(f"  Version: {plugin.version}")
        print(f"  Initialized: {plugin.is_initialized}")

    print("\n✓ Plugin automatically cleaned up after 'with' block")


def demo_parameter_control():
    """Demo 3: Parameter Discovery and Control"""
    print("\n" + "=" * 70)
    print(" Demo 3: Parameter Control")
    print("=" * 70)

    print("\nLoading plugin with parameters...")

    with cm.AudioUnitPlugin.from_name("Bandpass", component_type='aufx') as plugin:
        print(f"\nPlugin: {plugin.name}")
        print(f"Parameters: {len(plugin.parameters)}")

        if len(plugin.parameters) > 0:
            print("\nParameter details:")
            print("-" * 70)
            for param in plugin.parameters:
                print(f"\n  {param.name}:")
                print(f"    Current value: {param.value:.2f} {param.unit_name}")
                print(f"    Range: {param.min_value:.2f} - {param.max_value:.2f}")
                print(f"    Default: {param.default_value:.2f}")

            # Method 1: Using parameter object
            print("\n\nSetting parameters (method 1: parameter object):")
            param = plugin.parameters[0]
            original = param.value
            param.value = (param.min_value + param.max_value) / 2.0
            print(f"  Set '{param.name}' from {original:.2f} to {param.value:.2f}")

            # Method 2: Using dictionary access
            print("\nSetting parameters (method 2: dictionary style):")
            param_name = plugin.parameters[0].name
            plugin[param_name] = original  # Restore original value
            print(f"  plugin['{param_name}'] = {plugin[param_name]:.2f}")

            # Method 3: Using set_parameter method
            print("\nSetting parameters (method 3: set_parameter):")
            plugin.set_parameter(param_name, param.max_value * 0.75)
            print(f"  Set '{param_name}' to {plugin[param_name]:.2f}")

        print("\n✓ Demo complete")


def demo_preset_management():
    """Demo 4: Preset Management"""
    print("\n" + "=" * 70)
    print(" Demo 4: Factory Presets")
    print("=" * 70)

    # Find a plugin with presets (try AUReverb)
    print("\nLooking for plugins with factory presets...")

    try:
        with cm.AudioUnitPlugin.from_name("Reverb", component_type='aufx') as plugin:
            print(f"\nPlugin: {plugin.name}")

            presets = plugin.factory_presets
            print(f"Factory presets: {len(presets)}")

            if len(presets) > 0:
                print("\nAvailable presets:")
                print("-" * 70)
                for i, preset in enumerate(presets[:10]):
                    print(f"  [{i}] {preset.name}")
                if len(presets) > 10:
                    print(f"  ... and {len(presets) - 10} more")

                # Load a preset
                print(f"\nLoading preset: '{presets[0].name}'")
                plugin.load_preset(presets[0])
                print("✓ Preset loaded")

                if len(plugin.parameters) > 0:
                    print("\nParameter values after preset load:")
                    for param in plugin.parameters[:5]:
                        print(f"  {param.name}: {param.value:.2f}")

    except ValueError as e:
        print(f"Could not find Reverb plugin: {e}")

    print("\n✓ Demo complete")


def demo_multiple_plugins():
    """Demo 5: Working with Multiple Plugins"""
    print("\n" + "=" * 70)
    print(" Demo 5: Multiple Plugins")
    print("=" * 70)

    host = cm.AudioUnitHost()

    print("\nLoading multiple plugins simultaneously...")

    # Load multiple plugins
    with host.load_plugin("Bandpass", type='effect') as filter, \
         host.load_plugin("Reverb", type='effect') as reverb:

        print(f"\n✓ Loaded:")
        print(f"  1. {filter.name} ({len(filter.parameters)} params)")
        print(f"  2. {reverb.name} ({len(reverb.parameters)} params)")

        print("\nThis demonstrates:")
        print("  • Multiple simultaneous plugin instances")
        print("  • Independent parameter control")
        print("  • Automatic cleanup of all plugins")

    print("\n✓ All plugins automatically cleaned up")


def demo_complete_workflow():
    """Demo 6: Complete Workflow"""
    print("\n" + "=" * 70)
    print(" Demo 6: Complete Workflow")
    print("=" * 70)

    print("\nDemonstrating a complete plugin hosting workflow...")

    host = cm.AudioUnitHost()

    # 1. Discover plugins
    print("\n1. Discovering plugins...")
    effects = host.discover_plugins(type='effect', manufacturer='appl')
    print(f"   Found {len(effects)} Apple effects")

    # 2. Load plugin
    print("\n2. Loading plugin...")
    with host.load_plugin("Bandpass", type='effect') as plugin:
        print(f"   ✓ {plugin.name}")

        # 3. Inspect parameters
        print("\n3. Inspecting parameters...")
        print(f"   Found {len(plugin.parameters)} parameters")
        for param in plugin.parameters:
            print(f"     • {param.name}")

        # 4. Set parameters
        if len(plugin.parameters) >= 2:
            print("\n4. Configuring parameters...")
            # Set center frequency
            plugin[plugin.parameters[0].name] = 1000.0
            print(f"     ✓ {plugin.parameters[0].name} = 1000.0 Hz")

            # Set bandwidth
            if len(plugin.parameters) > 1:
                plugin[plugin.parameters[1].name] = 500.0
                print(f"     ✓ {plugin.parameters[1].name} = 500.0 Hz")

        # 5. Check presets
        print("\n5. Checking factory presets...")
        print(f"   {len(plugin.factory_presets)} presets available")

        print("\n6. Plugin ready for audio processing!")
        print("   (Audio routing will be demonstrated in a separate example)")

    print("\n✓ Complete workflow demonstrated")


def main():
    """Main demo runner"""
    print("\n" + "=" * 70)
    print(" CoreMusic AudioUnit High-Level API Demo")
    print("=" * 70)
    print("\nThis demo showcases the Pythonic object-oriented API for AudioUnit")
    print("plugin hosting. The API provides:")
    print("  • Simple, intuitive plugin discovery")
    print("  • Automatic resource management")
    print("  • Dictionary-style parameter access")
    print("  • Context manager support")

    demos = [
        ("Plugin Discovery", demo_plugin_discovery),
        ("Plugin Loading", demo_plugin_loading),
        ("Parameter Control", demo_parameter_control),
        ("Preset Management", demo_preset_management),
        ("Multiple Plugins", demo_multiple_plugins),
        ("Complete Workflow", demo_complete_workflow),
    ]

    while True:
        print("\n" + "=" * 70)
        print(" Demo Menu")
        print("=" * 70)
        for i, (name, _) in enumerate(demos, 1):
            print(f"  [{i}] {name}")
        print("  [a] Run all demos")
        print("  [q] Quit")

        choice = input("\nChoice: ").strip().lower()

        if choice == 'q':
            break
        elif choice == 'a':
            for name, demo_func in demos:
                demo_func()
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(demos):
                demos[idx][1]()
            else:
                print("Invalid choice")
        else:
            print("Invalid choice")

    print("\n" + "=" * 70)
    print(" Demo Complete!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  ✓ AudioUnitHost provides simple plugin discovery")
    print("  ✓ AudioUnitPlugin wraps plugins with clean API")
    print("  ✓ Context managers handle cleanup automatically")
    print("  ✓ Parameters accessible via dict notation")
    print("  ✓ Factory presets easy to browse and load")
    print("\nNext steps:")
    print("  • Check out the audio processing examples")
    print("  • Explore Link integration for tempo sync")
    print("  • Build your own plugin chains")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(0)
