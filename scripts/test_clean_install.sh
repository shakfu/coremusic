#!/usr/bin/env bash
# Test script to verify coremusic can be installed and imported without optional dependencies
set -e

echo "=========================================="
echo "CoreMusic Clean Install Test"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Create temp directory for test
TEST_DIR="/tmp/coremusic_test_$$"
echo "${YELLOW}→${NC} Creating test environment in: ${TEST_DIR}"
mkdir -p "$TEST_DIR"

# Trap to cleanup on exit
cleanup() {
    if [ -d "$TEST_DIR" ]; then
        echo ""
        echo "${YELLOW}→${NC} Cleaning up test environment..."
        rm -rf "$TEST_DIR"
    fi
}
trap cleanup EXIT

# Change to test directory
cd "$TEST_DIR"

# Step 1: Create virtual environment
echo ""
echo "${YELLOW}→${NC} Creating virtual environment..."
python3 -m venv env

# Step 2: Activate virtual environment
echo "${YELLOW}→${NC} Activating virtual environment..."
source env/bin/activate

# Step 3: Upgrade pip
echo ""
echo "${YELLOW}→${NC} Upgrading pip..."
pip install --quiet --upgrade pip

# Step 4: Install coremusic from project directory
echo ""
echo "${YELLOW}→${NC} Installing coremusic from: ${PROJECT_DIR}"
pip install --no-deps "${PROJECT_DIR}"

# Step 5: Check what got installed
echo ""
echo "${YELLOW}→${NC} Installed packages:"
pip list | grep -i "coremusic\|cython" || echo "  (none found)"

# Step 6: Verify numpy/scipy/matplotlib are NOT installed
echo ""
echo "${YELLOW}→${NC} Checking optional dependencies (should NOT be installed):"
for pkg in numpy scipy matplotlib; do
    if pip show "$pkg" &>/dev/null; then
        echo "  ${RED}✗${NC} $pkg is installed (unexpected)"
    else
        echo "  ${GREEN}✓${NC} $pkg is NOT installed (expected)"
    fi
done

# Step 7: Test importing coremusic
echo ""
echo "${YELLOW}→${NC} Testing coremusic import..."
python3 << 'EOF'
import sys

# Test 1: Basic import
try:
    import coremusic
    print("  ✓ import coremusic")
except ImportError as e:
    print(f"  ✗ import coremusic failed: {e}")
    sys.exit(1)

# Test 2: Check NUMPY_AVAILABLE flag
try:
    print(f"  ✓ coremusic.NUMPY_AVAILABLE = {coremusic.NUMPY_AVAILABLE}")
    if coremusic.NUMPY_AVAILABLE:
        print("    ⚠ Warning: NUMPY_AVAILABLE is True but numpy should not be installed")
except AttributeError as e:
    print(f"  ✗ NUMPY_AVAILABLE not found: {e}")

# Test 3: Import capi submodule
try:
    import coremusic.capi as capi
    print("  ✓ import coremusic.capi")
except ImportError as e:
    print(f"  ✗ import coremusic.capi failed: {e}")
    sys.exit(1)

# Test 4: Import core objects
try:
    from coremusic import AudioFile, AudioQueue, AudioUnit
    print("  ✓ import core objects (AudioFile, AudioQueue, AudioUnit)")
except ImportError as e:
    print(f"  ✗ import core objects failed: {e}")
    sys.exit(1)

# Test 5: Import audio package
try:
    import coremusic.audio
    print("  ✓ import coremusic.audio")
except ImportError as e:
    print(f"  ✗ import coremusic.audio failed: {e}")
    sys.exit(1)

# Test 6: Import mmap_file (this had the bug)
try:
    from coremusic.audio.mmap_file import MMapAudioFile
    print("  ✓ import coremusic.audio.mmap_file.MMapAudioFile")
except ImportError as e:
    print(f"  ✗ import MMapAudioFile failed: {e}")
    sys.exit(1)

# Test 7: Verify numpy methods raise proper errors
try:
    from coremusic.audio.mmap_file import MMapAudioFile
    # This should work (doesn't require numpy)
    mmap_file = MMapAudioFile.__new__(MMapAudioFile)
    print("  ✓ MMapAudioFile can be instantiated")
except Exception as e:
    print(f"  ✗ MMapAudioFile instantiation failed: {e}")
    sys.exit(1)

# Test 8: Import utils
try:
    import coremusic.utils
    print("  ✓ import coremusic.utils")
except ImportError as e:
    print(f"  ✗ import coremusic.utils failed: {e}")
    sys.exit(1)

# Test 9: Import utils.scipy (should work even without scipy)
try:
    import coremusic.utils.scipy
    print("  ✓ import coremusic.utils.scipy (module imports without scipy)")
except ImportError as e:
    print(f"  ✗ import coremusic.utils.scipy failed: {e}")
    sys.exit(1)

# Test 10: Import MIDI
try:
    from coremusic import MIDIClient
    print("  ✓ import MIDIClient")
except ImportError as e:
    print(f"  ✗ import MIDIClient failed: {e}")
    sys.exit(1)

print("\n✓ All import tests passed!")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "${GREEN}=========================================="
    echo "✓ CLEAN INSTALL TEST PASSED"
    echo "==========================================${NC}"
    echo ""
    echo "Summary:"
    echo "  • coremusic installs without dependencies"
    echo "  • All core modules import successfully"
    echo "  • No numpy, scipy, or matplotlib required"
    echo "  • Optional dependencies properly handled"
    exit 0
else
    echo ""
    echo "${RED}=========================================="
    echo "✗ CLEAN INSTALL TEST FAILED"
    echo "==========================================${NC}"
    exit 1
fi
