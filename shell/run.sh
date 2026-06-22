#!/bin/bash

################################################################################
# SPMM Run Script for Ascend 910B
#
# Usage:
#   ./run.sh <M> <N> <K> [sparsity] [d] [mode] [device_id] [rebuild]
#
# Modes:
#   gen    : generate or use data into ascblas/data/, then exit
#   verify : reuse matching data if present; otherwise prepare data, optionally compile, verify
#   prof   : reuse matching data if present; otherwise prepare data, optionally compile, msprof
#
# Args:
#   rebuild: 1 = run make clean && make before execution (default)
#            0 = reuse existing build/spmm and kernel .o files
#
# Data policy:
#   1) If SPMM_FULL_DATA_CASE is set, use datasets_full/<case> via tools/use_full_dataset.py.
#   2) Otherwise, if ../data/.spmm_case_meta matches M/N/K/sparsity/d, reuse ../data.
#   3) Otherwise, run spmm_gen_data.py and write ../data/.spmm_case_meta.
#
# Optional env:
#   SPMM_FORCE_REGEN=1    Force regenerate temporary data even if metadata matches.
#   SPMM_SKIP_DATA_GEN=1  Skip data preparation and trust existing ascblas/data.
#   SPMM_FULL_DATA_CASE=<case-name>
#                         Use ascblas/datasets_full/<case-name>/.
#   SPMM_VERIFY_LEVEL=<0|1|2>
#                         Override verify level; use 2 to print kernel debug dumps.
#   KERNEL_CFLAGS="..."
#                         Extra global kernel flags for experiments. In the
#                         multi-tiling dispatch build, tiling defines are owned
#                         by the Makefile variants and should not be supplied here.
#   HOST_CFLAGS="..."     Optional extra host flags. Usually not needed.
################################################################################

set -e

function check_error {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed!"
        exit 1
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- Environment Setup ---
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH}

EXECUTABLE="spmm"

# --- Parse Arguments ---
if [ $# -lt 3 ]; then
    echo "Usage: $0 <M> <N> <K> [sparsity] [d] [mode] [device_id] [rebuild]"
    echo "Modes: gen | verify | prof"
    echo "rebuild: 1 = compile before run (default), 0 = reuse existing build"
    exit 1
fi

M=$1
N=$2
K=$3
sparsity=${4:-85}
d=${5:-16}
MODE=${6:-verify}
device_id=${7:-0}
rebuild=${8:-1}

if [ "${MODE}" != "gen" ] && [ "${MODE}" != "verify" ] && [ "${MODE}" != "prof" ]; then
    echo "Invalid mode: ${MODE}"
    echo "Valid modes: gen | verify | prof"
    exit 1
fi

if [ "${rebuild}" != "0" ] && [ "${rebuild}" != "1" ]; then
    echo "Invalid rebuild flag: ${rebuild}"
    echo "Valid rebuild values: 1 = compile before run, 0 = reuse existing build"
    exit 1
fi

# --- Configuration ---
verifyLevel=1
if [ "${MODE}" == "prof" ]; then
    verifyLevel=0
fi
if [ -n "${SPMM_VERIFY_LEVEL:-}" ]; then
    verifyLevel="${SPMM_VERIFY_LEVEL}"
fi

DATA_DIR="../data"
META_FILE="${DATA_DIR}/.spmm_case_meta"
EXPECTED_META="M=${M};N=${N};K=${K};sparsity=${sparsity};d=${d}"

function write_meta {
    local source_desc="$1"
    echo "${EXPECTED_META};source=${source_desc}" > "${META_FILE}"
}

function meta_matches {
    if [ ! -f "${META_FILE}" ]; then
        return 1
    fi
    local current_meta
    current_meta="$(cat "${META_FILE}")"
    if [[ "${current_meta}" == "${EXPECTED_META};"* ]]; then
        return 0
    fi
    return 1
}

function prepare_data {
    if [ "${SPMM_SKIP_DATA_GEN:-0}" == "1" ]; then
        echo "  SPMM_SKIP_DATA_GEN=1: skip data preparation and trust existing ${DATA_DIR}/"
        return 0
    fi

    if [ -n "${SPMM_FULL_DATA_CASE:-}" ]; then
        echo "  Using full dataset: ${SPMM_FULL_DATA_CASE}"
        python3 ../tools/use_full_dataset.py "${SPMM_FULL_DATA_CASE}"
        check_error "Use full dataset"
        write_meta "full:${SPMM_FULL_DATA_CASE}"
        return 0
    fi

    if [ "${SPMM_FORCE_REGEN:-0}" != "1" ] && meta_matches; then
        echo "  Reusing existing data in ${DATA_DIR}/"
        echo "  Metadata: $(cat "${META_FILE}")"
        return 0
    fi

    if [ "${SPMM_FORCE_REGEN:-0}" == "1" ]; then
        echo "  SPMM_FORCE_REGEN=1: force regenerate temporary data into ${DATA_DIR}/"
    else
        echo "  No matching data metadata found. Generate temporary data into ${DATA_DIR}/"
        if [ -f "${META_FILE}" ]; then
            echo "  Existing metadata: $(cat "${META_FILE}")"
            echo "  Expected metadata: ${EXPECTED_META}"
        fi
    fi

    python3 spmm_gen_data.py "${M}" "${N}" "${K}" "${sparsity}" "${d}"
    check_error "Data generation"
    write_meta "temporary"
}

function compile_build {
    echo "[2/3] Compiling..."
    make clean

    if [ -n "${HOST_CFLAGS+x}" ]; then
        make KERNEL_CFLAGS="${KERNEL_CFLAGS:-}" HOST_CFLAGS="${HOST_CFLAGS}"
    else
        make KERNEL_CFLAGS="${KERNEL_CFLAGS:-}"
    fi
    check_error "Compilation"
}

function check_existing_build {
    if [ ! -x "build/${EXECUTABLE}" ]; then
        echo "Error: rebuild=0 but build/${EXECUTABLE} does not exist or is not executable."
        echo "Run once with rebuild=1 first, for example:"
        echo "  ./run.sh ${M} ${N} ${K} ${sparsity} ${d} ${MODE} ${device_id} 1"
        exit 1
    fi

    local kernel_obj_count
    kernel_obj_count=$(find build -maxdepth 1 -name 'spmm_kernel*.o' 2>/dev/null | wc -l)
    if [ "${kernel_obj_count}" -eq 0 ]; then
        echo "Error: rebuild=0 but no build/spmm_kernel*.o kernel binary was found."
        echo "Run once with rebuild=1 first."
        exit 1
    fi

    echo "[2/3] Reusing existing build artifacts."
    echo "  Executable: build/${EXECUTABLE}"
    echo "  Kernel objects found: ${kernel_obj_count}"
}

# --- Print configuration ---
echo "=========================================="
echo "SPMM Configuration:"
echo "  Dimensions: M=${M}, N=${N}, K=${K}"
echo "  Sparsity:   ${sparsity}%"
echo "  Block dim:  d=${d}"
echo "  Mode:       ${MODE}"
echo "  Device:     ${device_id}"
echo "  Rebuild:    ${rebuild}"
if [ -n "${SPMM_FULL_DATA_CASE:-}" ]; then
    echo "  Dataset:    ${SPMM_FULL_DATA_CASE}"
else
    echo "  Dataset:    temporary data in ascblas/data/"
fi
echo "=========================================="

# --- Prepare directories ---
mkdir -p ../data
mkdir -p build
mkdir -p prof

# --- Step 1: Prepare data ---
echo "[1/3] Preparing test data..."
prepare_data

if [ "${MODE}" == "gen" ]; then
    echo ""
    echo "=========================================="
    echo "Data is ready in ascblas/data/."
    echo "Mode is gen, skip compile and kernel run."
    echo "=========================================="
    exit 0
fi

# --- Step 2: Compile or reuse ---
if [ "${rebuild}" == "1" ]; then
    compile_build
else
    check_existing_build
fi

# --- Step 3: Run ---
echo "[3/3] Running SPMM kernel..."
cd build

if [ "${MODE}" == "prof" ]; then
    echo "  Running performance analysis with msprof..."

    rm -rf ../prof/*

    msprof \
        --application="./${EXECUTABLE} ${M} ${N} ${K} ${verifyLevel} ${device_id}" \
        --output=../prof \
        --task-time=on \
        --aicpu=on \
        --l2=on

    check_error "Performance profiling"

    cd ..

    echo ""
    echo "Profiling completed. Output directory: prof/"
    ls -lh prof/

    if [ -f "./prof.py" ]; then
        echo ""
        echo "Profiling summary: M N K sparsity avg_ms effective_tflops dense_equiv_tflops"
        python3 ./prof.py "${M}" "${N}" "${K}" "${sparsity}" --prof-dir ./prof
    else
        echo ""
        echo "Warning: prof.py not found in shell/. Skip command-line profiling summary."
    fi

    echo ""
    echo "Tip: Open prof/timeline/msprof*timeline.json in Chrome://tracing if needed."
else
    echo "  Running functional verification..."
    ./${EXECUTABLE} "${M}" "${N}" "${K}" "${verifyLevel}" "${device_id}"
    check_error "Kernel execution"

    cd ..
fi

echo ""
echo "=========================================="
echo "SPMM execution completed successfully!"
echo "=========================================="
