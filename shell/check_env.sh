#!/bin/bash

################################################################################
# SPMM Environment Check Script for Ascend 910B3 - SR-BCRS Format
# Validates the environment before compilation and execution
################################################################################

echo "=========================================="
echo "SPMM SR-BCRS Environment Check for Ascend 910B3"
echo "=========================================="

# 尝试自动加载环境变量
DEFAULT_ASCEND_PATH="/usr/local/Ascend/ascend-toolkit/set_env.sh"

# 如果环境变量未设置，且默认路径下的脚本存在，则自动 source
if [ -z "$ASCEND_HOME_PATH" ]; then
    if [ -f "$DEFAULT_ASCEND_PATH" ]; then
        echo "Detected missing environment. Sourcing $DEFAULT_ASCEND_PATH ..."
        source "$DEFAULT_ASCEND_PATH"
    fi
fi

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

pass_count=0
warn_count=0
fail_count=0

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((pass_count++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((warn_count++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((fail_count++))
}

# Check environment variables
echo "[1/5] Checking environment variables..."

if [ -n "$ASCEND_HOME_PATH" ]; then
    check_pass "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
else
    check_fail "ASCEND_HOME_PATH is not set!"
    echo "   Please run: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
fi

if [ -n "$LD_LIBRARY_PATH" ]; then
    check_pass "LD_LIBRARY_PATH is set"
else
    check_warn "LD_LIBRARY_PATH is empty (may cause issues)"
fi

# Check binary tools
echo "[2/5] Checking compiler tools..."

if command -v ccec &> /dev/null; then
    ccec_version=$(ccec --version 2>&1 | head -1)
    check_pass "ccec compiler found: $ccec_version"
else
    check_fail "ccec compiler not found in PATH"
    echo "   Expected location: $ASCEND_HOME_PATH/bin/ccec"
fi

if command -v g++ &> /dev/null; then
    gpp_version=$(g++ --version | head -1)
    check_pass "g++ compiler found: $gpp_version"
else
    check_fail "g++ compiler not found"
fi

if command -v ld.lld &> /dev/null; then
    check_pass "ld.lld linker found"
else
    check_fail "ld.lld linker not found"
fi

# Check NPU device
echo "[3/5] Checking NPU devices..."

if command -v npu-smi &> /dev/null; then
    check_pass "npu-smi tool found"

    npu_count=$(npu-smi info -l 2>/dev/null | grep -c "NPU ID")
    if [ "$npu_count" -gt 0 ]; then
        check_pass "Found $npu_count NPU device(s)"

        # Check 910B3 specifically
        npu_type=$(npu-smi info -t board -i 0 2>/dev/null | grep "Name" | grep -o "910[^ ]*" || echo "")
        if [[ "$npu_type" == *"910B3"* ]]; then
            check_pass "NPU Type: 910B3 detected"
        else
            check_warn "NPU Type: $npu_type (expected 910B3 for optimal performance)"
        fi
    else
        check_fail "No NPU devices detected"
    fi
else
    check_fail "npu-smi not found"
fi

# Check Python and dependencies
echo "[4/5] Checking Python environment..."

if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1)
    check_pass "Python 3 found: $python_version"

    # Check numpy
    python3 -c "import numpy" 2>/dev/null
    if [ $? -eq 0 ]; then
        numpy_version=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null)
        check_pass "NumPy found: $numpy_version"
    else
        check_warn "NumPy not installed (required for data generation)"
        echo "   Install with: pip3 install numpy"
    fi

    # Check scipy
    python3 -c "import scipy" 2>/dev/null
    if [ $? -eq 0 ]; then
        scipy_version=$(python3 -c "import scipy; print(scipy.__version__)" 2>/dev/null)
        check_pass "SciPy found: $scipy_version"
    else
        check_warn "SciPy not installed (required for CSR format)"
        echo "   Install with: pip3 install scipy"
    fi
else
    check_fail "Python 3 not found"
fi

# Check required files
echo "[5/5] Checking source files..."

required_files=(
    "../src/spmm_kernel.cce"
    "../src/spmm.h"
    "../src/main.cpp"
    "../src/utils/sr_bcrs_utils.h"
    "../src/utils/file_utils.h"
    "../include/handle.cc"
    "makefile"
    "run.sh"
    "build.sh"
    "check_env.sh"
    "spmm_gen_data.py"
    "prof.py"
    "README.md"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        check_pass "$file found"
    else
        check_fail "$file missing"
    fi
done

# Summary
echo ""
echo "=========================================="
echo "Environment Check Summary"
echo "=========================================="
echo -e "${GREEN}Passed:${NC} $pass_count"
echo -e "${YELLOW}Warnings:${NC} $warn_count"
echo -e "${RED}Failed:${NC} $fail_count"
echo "=========================================="

safe_exit() {
    if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
        return "$1"
    else
        exit "$1"
    fi
}

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}Environment check passed!${NC}"
    echo "You can now build and run SPMM."
    echo ""
    echo "Quick start:"
    echo "  ./build.sh  # Clean, compile, and test"
    safe_exit 0
else
    echo -e "${RED}Environment check failed!${NC}"
    echo "Please fix the issues above before building."
    safe_exit 1
fi