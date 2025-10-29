#!/usr/bin/env python3
"""Demo: Audio Slicing and Recombination

Demonstrates the audio slicing and recombination capabilities of CoreMusic:
- Onset detection slicing
- Transient detection slicing
- Zero-crossing slicing
- Grid-based slicing
- Manual slicing
- Slice collection manipulation
- Creative recombination strategies
- Real-world workflows

Requires NumPy and SciPy.
"""

import sys
from pathlib import Path

# Add src to path for demo purposes
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import coremusic as cm

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available - demos will be skipped")

try:
    from scipy import signal

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not available - demos will be skipped")

if NUMPY_AVAILABLE and SCIPY_AVAILABLE:
    from coremusic.audio.slicing import (
        AudioSlicer,
        SliceCollection,
        SliceRecombinator,
    )


def demo_onset_slicing():
    """Demo 1: Onset detection slicing."""
    print("\n" + "=" * 70)
    print("DEMO 1: Onset Detection Slicing")
    print("=" * 70)

    print("\nSlicing audio using onset detection...")
    slicer = AudioSlicer("tests/amen.wav", method="onset", sensitivity=0.5)
    slices = slicer.detect_slices(min_slice_duration=0.05, max_slices=16)

    print(f"\nResults:")
    print(f"  Number of slices: {len(slices)}")
    print(f"  Slicing method: onset detection")
    print(f"  Sensitivity: 0.5")

    if len(slices) > 0:
        print(f"\n  First 5 slices:")
        for i, s in enumerate(slices[:5], 1):
            print(f"    Slice {i}: {s.start:.3f}s - {s.end:.3f}s ({s.duration:.3f}s)")

        # Statistics
        durations = [s.duration for s in slices]
        print(f"\n  Slice duration statistics:")
        print(f"    Shortest: {min(durations):.3f}s")
        print(f"    Longest: {max(durations):.3f}s")
        print(f"    Average: {np.mean(durations):.3f}s")


def demo_transient_slicing():
    """Demo 2: Transient detection slicing."""
    print("\n" + "=" * 70)
    print("DEMO 2: Transient Detection Slicing")
    print("=" * 70)

    print("\nSlicing audio using transient detection...")
    slicer = AudioSlicer("tests/amen.wav", method="transient", sensitivity=0.6)
    slices = slicer.detect_slices(window_size=0.02, threshold_db=-40.0)

    print(f"\nResults:")
    print(f"  Number of slices: {len(slices)}")
    print(f"  Slicing method: transient detection")
    print(f"  Window size: 0.02s")
    print(f"  Threshold: -40.0 dB")

    if len(slices) > 0:
        print(f"\n  First 5 transient slices:")
        for i, s in enumerate(slices[:5], 1):
            print(
                f"    Slice {i}: {s.start:.3f}s - {s.end:.3f}s "
                f"(confidence: {s.confidence:.2f})"
            )


def demo_zero_crossing_slicing():
    """Demo 3: Zero-crossing detection slicing."""
    print("\n" + "=" * 70)
    print("DEMO 3: Zero-Crossing Slicing")
    print("=" * 70)

    print("\nSlicing audio at zero crossings (glitch-free)...")
    slicer = AudioSlicer("tests/amen.wav", method="zero_crossing")
    slices = slicer.detect_slices(target_slices=16, snap_to_zero=True)

    print(f"\nResults:")
    print(f"  Number of slices: {len(slices)}")
    print(f"  Slicing method: zero-crossing")
    print(f"  Snap to zero: Yes (glitch-free)")

    if len(slices) > 0:
        durations = [s.duration for s in slices]
        print(f"\n  Slice durations are roughly equal:")
        print(f"    Average: {np.mean(durations):.3f}s")
        print(f"    Std dev: {np.std(durations):.4f}s")
        print(f"\n  First 3 slices:")
        for i, s in enumerate(slices[:3], 1):
            print(f"    Slice {i}: {s.start:.3f}s - {s.end:.3f}s")


def demo_grid_slicing():
    """Demo 4: Grid-based slicing."""
    print("\n" + "=" * 70)
    print("DEMO 4: Grid-Based Slicing")
    print("=" * 70)

    print("\nSlicing audio into equal-duration segments...")
    slicer = AudioSlicer("tests/amen.wav", method="grid")
    slices = slicer.detect_slices(divisions=8)

    print(f"\nResults:")
    print(f"  Number of slices: {len(slices)}")
    print(f"  Slicing method: regular grid")
    print(f"  Divisions: 8")

    if len(slices) > 0:
        durations = [s.duration for s in slices]
        print(f"\n  All slices have equal duration:")
        print(f"    Duration: {durations[0]:.3f}s")

        print(f"\n  Grid slices:")
        for i, s in enumerate(slices, 1):
            print(f"    Slice {i}: {s.start:.3f}s - {s.end:.3f}s")


def demo_manual_slicing():
    """Demo 5: Manual slicing."""
    print("\n" + "=" * 70)
    print("DEMO 5: Manual Slicing")
    print("=" * 70)

    print("\nSlicing audio at manually specified time points...")
    slicer = AudioSlicer("tests/amen.wav", method="manual")
    slice_points = [0.0, 0.5, 1.0, 1.5, 2.0]
    slices = slicer.detect_slices(slice_points=slice_points)

    print(f"\nResults:")
    print(f"  Number of slices: {len(slices)}")
    print(f"  Slicing method: manual")
    print(f"  Specified points: {slice_points}")

    if len(slices) > 0:
        print(f"\n  Manual slices:")
        for i, s in enumerate(slices, 1):
            print(f"    Slice {i}: {s.start:.3f}s - {s.end:.3f}s ({s.duration:.3f}s)")


def demo_slice_collection():
    """Demo 6: Slice collection manipulation."""
    print("\n" + "=" * 70)
    print("DEMO 6: Slice Collection Manipulation")
    print("=" * 70)

    print("\nDemonstrating slice collection operations...")

    # Create slices
    slicer = AudioSlicer("tests/amen.wav", method="onset", sensitivity=0.5)
    slices = slicer.detect_slices(max_slices=16)
    collection = SliceCollection(slices)

    print(f"\nOriginal collection:")
    print(f"  Number of slices: {len(collection)}")

    # Shuffle
    shuffled = collection.shuffle()
    print(f"\n1. Shuffle:")
    print(f"  Randomized order: {len(shuffled)} slices")

    # Reverse
    reversed_collection = collection.reverse()
    print(f"\n2. Reverse:")
    print(
        f"  First slice index: {reversed_collection[0].index} "
        f"(was {collection[0].index})"
    )

    # Repeat
    repeated = collection.select([0, 2, 4]).repeat(3)
    print(f"\n3. Select + Repeat:")
    print(f"  Selected slices 0, 2, 4 and repeated 3x: {len(repeated)} slices")

    # Filter
    filtered = collection.filter(lambda s: s.duration > 0.1)
    print(f"\n4. Filter:")
    print(f"  Slices with duration > 0.1s: {len(filtered)} slices")

    # Sort
    sorted_collection = collection.sort_by_duration()
    print(f"\n5. Sort by Duration:")
    print(
        f"  Shortest: {sorted_collection[0].duration:.3f}s, "
        f"Longest: {sorted_collection[-1].duration:.3f}s"
    )

    # Apply pattern
    patterned = collection.apply_pattern([0, 3, 1, 2, 3, 0])
    print(f"\n6. Apply Pattern:")
    print(f"  Pattern [0,3,1,2,3,0]: {len(patterned)} slices")

    # Chaining operations
    complex_result = collection.select([0, 1, 2, 3]).shuffle().repeat(2)
    print(f"\n7. Chaining Operations:")
    print(f"  select([0,1,2,3]).shuffle().repeat(2): {len(complex_result)} slices")


def demo_recombination():
    """Demo 7: Slice recombination strategies."""
    print("\n" + "=" * 70)
    print("DEMO 7: Slice Recombination Strategies")
    print("=" * 70)

    print("\nDemonstrating different recombination methods...")

    # Create slices
    slicer = AudioSlicer("tests/amen.wav", method="grid")
    slices = slicer.detect_slices(divisions=16)
    collection = SliceCollection(slices)

    print(f"\nCreated {len(slices)} grid slices")

    # Original order
    recombinator = SliceRecombinator(collection)
    original = recombinator.recombine(method="original", crossfade_duration=0.005)
    print(f"\n1. Original Order:")
    print(f"  Duration: {len(original) / 44100:.3f}s")
    print(f"  Method: original sequence with 5ms crossfade")

    # Random recombination
    random_audio = recombinator.recombine(
        method="random", crossfade_duration=0.01, num_slices=8
    )
    print(f"\n2. Random Recombination:")
    print(f"  Duration: {len(random_audio) / 44100:.3f}s")
    print(f"  Method: random selection of 8 slices with 10ms crossfade")

    # Reverse
    reversed_audio = recombinator.recombine(
        method="reverse", crossfade_duration=0.005, normalize=True
    )
    print(f"\n3. Reverse:")
    print(f"  Duration: {len(reversed_audio) / 44100:.3f}s")
    print(f"  Method: reversed order, normalized")

    # Pattern-based
    pattern = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14]
    patterned_audio = recombinator.recombine(
        method="pattern", crossfade_duration=0.003, pattern=pattern
    )
    print(f"\n4. Pattern-Based:")
    print(f"  Duration: {len(patterned_audio) / 44100:.3f}s")
    print(f"  Method: custom pattern with 3ms crossfade")

    # Custom ordering
    def by_duration_desc(slices_list):
        return sorted(slices_list, key=lambda s: s.duration, reverse=True)

    custom_audio = recombinator.recombine(
        method="custom", crossfade_duration=0.01, order_func=by_duration_desc
    )
    print(f"\n5. Custom Ordering:")
    print(f"  Duration: {len(custom_audio) / 44100:.3f}s")
    print(f"  Method: sorted by duration (longest first)")


def demo_creative_workflow():
    """Demo 8: Creative workflow example."""
    print("\n" + "=" * 70)
    print("DEMO 8: Creative Workflow Example")
    print("=" * 70)

    print("\nCreative breakbeat manipulation workflow...")

    print("\n  Step 1: Slice drum break using onset detection")
    slicer = AudioSlicer("tests/amen.wav", method="onset", sensitivity=0.6)
    slices = slicer.detect_slices(min_slice_duration=0.05, max_slices=16)
    print(f"    Detected {len(slices)} hits")

    print("\n  Step 2: Filter out short slices")
    collection = SliceCollection(slices)
    filtered = collection.filter(lambda s: s.duration > 0.08)
    print(f"    Filtered to {len(filtered)} slices (duration > 0.08s)")

    print("\n  Step 3: Select strongest hits")
    selected = filtered.select([0, 2, 4, 6])
    print(f"    Selected {len(selected)} strong hits")

    print("\n  Step 4: Create stuttering pattern")
    pattern = [0, 0, 1, 2, 2, 3, 1, 0]
    patterned = selected.apply_pattern(pattern)
    print(f"    Applied pattern: {len(patterned)} slices")

    print("\n  Step 5: Recombine with tight crossfading")
    recombinator = SliceRecombinator(patterned)
    result = recombinator.recombine(
        method="original", crossfade_duration=0.003, normalize=True
    )
    print(f"    Final audio: {len(result) / 44100:.3f}s, normalized")

    print("\n  Workflow complete! Created rhythmic variation.")


def demo_comparison():
    """Demo 9: Compare slicing methods."""
    print("\n" + "=" * 70)
    print("DEMO 9: Slicing Method Comparison")
    print("=" * 70)

    print("\nComparing different slicing methods on the same audio...")

    methods = {
        "onset": {"method": "onset", "kwargs": {"min_slice_duration": 0.05}},
        "transient": {
            "method": "transient",
            "kwargs": {"window_size": 0.02, "threshold_db": -40.0},
        },
        "zero_crossing": {
            "method": "zero_crossing",
            "kwargs": {"target_slices": 16, "snap_to_zero": True},
        },
        "grid": {"method": "grid", "kwargs": {"divisions": 16}},
    }

    results = {}
    for name, config in methods.items():
        slicer = AudioSlicer(
            "tests/amen.wav", method=config["method"], sensitivity=0.5
        )
        slices = slicer.detect_slices(**config["kwargs"])
        results[name] = slices

    print(f"\n  Comparison Results:")
    print(f"  {'Method':<15} {'Slices':<8} {'Avg Duration':<12} {'Std Dev'}")
    print(f"  {'-' * 50}")

    for name, slices in results.items():
        durations = [s.duration for s in slices]
        avg_duration = np.mean(durations)
        std_duration = np.std(durations)
        print(
            f"  {name:<15} {len(slices):<8} {avg_duration:<12.3f} {std_duration:.4f}"
        )

    print(f"\n  Observations:")
    print(
        f"    - Onset detection: Captures rhythmic hits ({len(results['onset'])} slices)"
    )
    print(
        f"    - Transient detection: More detailed segmentation "
        f"({len(results['transient'])} slices)"
    )
    print(f"    - Zero-crossing: Glitch-free equal divisions (16 slices)")
    print(f"    - Grid: Perfect equal divisions (16 slices)")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("COREMUSIC AUDIO SLICING DEMO")
    print("=" * 70)
    print("\nThis demo showcases audio slicing and recombination:")
    print("- Onset detection slicing")
    print("- Transient detection slicing")
    print("- Zero-crossing slicing")
    print("- Grid-based slicing")
    print("- Manual slicing")
    print("- Slice collection manipulation")
    print("- Creative recombination strategies")
    print("- Real-world workflows")

    if not NUMPY_AVAILABLE:
        print("\nERROR: NumPy is required for audio slicing.")
        print("Install with: pip install numpy")
        return

    if not SCIPY_AVAILABLE:
        print("\nERROR: SciPy is required for audio slicing.")
        print("Install with: pip install scipy")
        return

    try:
        demo_onset_slicing()
        demo_transient_slicing()
        demo_zero_crossing_slicing()
        demo_grid_slicing()
        demo_manual_slicing()
        demo_slice_collection()
        demo_recombination()
        demo_creative_workflow()
        demo_comparison()

        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("- AudioSlicer supports 5 different slicing methods")
        print("- Onset detection is ideal for rhythmic material")
        print("- Transient detection captures dynamic changes")
        print("- Zero-crossing provides glitch-free slicing")
        print("- Grid slicing creates equal-duration segments")
        print("- Manual slicing offers precise control")
        print("- SliceCollection enables fluent manipulation")
        print("- SliceRecombinator supports creative workflows")
        print("- Multiple recombination strategies available")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
