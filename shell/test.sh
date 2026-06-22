#!/bin/bash

################################################################################
# Multi-case SPMM profiling with runtime tiling dispatch.
#
# Purpose:
#   This script is for the engineering build that compiles all tiling-specialized
#   kernels once and lets host code select the right kernel at runtime.
#
# For each case it:
#   1) materializes datasets_full/<case> into ascblas/data,
#   2) computes the expected adaptive tiling from A_indptr.bin for reporting,
#   3) profiles build/spmm with msprof,
#   4) appends one compact summary row.
#
# It can compile once at the beginning, or reuse an existing build.
#
# Recommended location:
#   ascblas/shell/test.sh
#
# Usage:
#   ./test.sh <cases_json> [device_id] [rebuild] [case_name ...]
#
# Arguments:
#   rebuild=1  Rebuild once before running all cases. This is the default.
#   rebuild=0  Do not rebuild; reuse existing build/spmm and kernel objects.
#
# Compatibility:
#   If the third argument is not 0 or 1, it is treated as the first case_name
#   and rebuild defaults to 1.
#
# Output:
#   prof_multi_cases/summary.tsv
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}

EXECUTABLE="spmm"
DATA_DIR="../data"
DATASET_ROOT="../datasets_full"
PROF_ROOT="./prof_multi_cases"
RESULTS_TSV="${PROF_ROOT}/summary.tsv"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <cases_json> [device_id] [rebuild] [case_name ...]"
    echo "  rebuild=1: rebuild once before running all cases (default)"
    echo "  rebuild=0: reuse existing build/spmm and kernel objects"
    exit 1
fi

CASES_JSON="$1"
shift || true

device_id="0"
if [ $# -gt 0 ]; then
    device_id="$1"
    shift || true
fi

# Rebuild flag. Defaults to 1. For backwards compatibility, if the next
# argument is not 0 or 1, it is treated as the first case filter.
rebuild="${SPMM_MULTI_REBUILD:-1}"
if [ $# -gt 0 ]; then
    if [ "$1" == "0" ] || [ "$1" == "1" ]; then
        rebuild="$1"
        shift || true
    fi
fi

if [ "${rebuild}" != "0" ] && [ "${rebuild}" != "1" ]; then
    echo "Error: rebuild must be 0 or 1, got: ${rebuild}"
    exit 1
fi

CASE_FILTERS=("$@")

if [ ! -f "${CASES_JSON}" ]; then
    echo "Error: cases_json not found: ${CASES_JSON}"
    exit 1
fi

function abs_path {
    python3 - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
}

CASES_JSON_ABS="$(abs_path "${CASES_JSON}")"

function load_cases {
    python3 - "${CASES_JSON_ABS}" "${CASE_FILTERS[@]}" <<'PY'
import json
import sys
from pathlib import Path

cases_path = Path(sys.argv[1])
filters = set(sys.argv[2:])
with cases_path.open("r", encoding="utf-8") as f:
    payload = json.load(f)
cases = payload["cases"] if isinstance(payload, dict) else payload

matched = 0
for case in cases:
    name = str(case["name"])
    if filters and name not in filters:
        continue
    matched += 1
    print("\t".join([
        name,
        str(case["M"]),
        str(case["N"]),
        str(case["K"]),
        str(case["sparsity"]),
        str(case.get("d", 16)),
    ]))

if filters and matched != len(filters):
    known = {str(c["name"]) for c in cases}
    missing = sorted(filters - known)
    raise SystemExit("unknown case name(s): " + ", ".join(missing))
PY
}

function ensure_dataset {
    local case_name="$1"
    if [ -d "${DATASET_ROOT}/${case_name}" ]; then
        return 0
    fi

    if [ "${SPMM_MULTI_GEN_MISSING:-0}" == "1" ]; then
        echo "  Dataset missing, generating: ${case_name}"
        python3 ../tools/generate_full_datasets.py \
            --cases "${CASES_JSON_ABS}" \
            --only "${case_name}"
        return 0
    fi

    echo "Error: dataset not found: ${DATASET_ROOT}/${case_name}"
    echo "Hint: generate it first, for example:"
    echo "  cd ../tools && python3 generate_full_datasets.py --cases ${CASES_JSON_ABS} --only ${case_name}"
    echo "Or set SPMM_MULTI_GEN_MISSING=1 before running this script."
    exit 1
}

function prepare_full_dataset {
    local case_name="$1"
    python3 ../tools/use_full_dataset.py "${case_name}" >"${PROF_ROOT}/${case_name}_use_dataset.log"
}

function select_case_tiling_for_report {
    local indptr_file="${DATA_DIR}/vector_csr/A_indptr.bin"
    if [ ! -f "${indptr_file}" ]; then
        echo "Error: tiling report requires ${indptr_file}" >&2
        exit 1
    fi

    # This is only for reporting in summary.tsv. Runtime kernel selection is done
    # inside the host code. Keep this threshold table in sync with spmm.h.
    local selector_output
    if ! selector_output="$(python3 - "${indptr_file}" <<'PYTILING'
from array import array
from pathlib import Path
import sys

path = Path(sys.argv[1])
raw = path.read_bytes()
if len(raw) < 8 or len(raw) % 4 != 0:
    raise SystemExit(f"invalid int32 indptr file: {path}")

offsets = array("i")
offsets.frombytes(raw)
if sys.byteorder != "little":
    offsets.byteswap()

previous = offsets[0]
if previous < 0:
    raise SystemExit("indptr must start with a non-negative offset")

max_vectors = 0
for current in offsets[1:]:
    if current < previous:
        raise SystemExit("indptr must be monotonically non-decreasing")
    max_vectors = max(max_vectors, current - previous)
    previous = current

if max_vectors <= 256:
    aic_n_batch, sparse_k_tile = 8, 256
elif max_vectors <= 512:
    aic_n_batch, sparse_k_tile = 4, 512
elif max_vectors <= 1024:
    aic_n_batch, sparse_k_tile = 2, 1024
else:
    aic_n_batch, sparse_k_tile = 1, 2048

print(f"SPMM_AUTO_MAX_VECTORS={max_vectors}")
print(f"SPMM_AUTO_AIC_N_BATCH={aic_n_batch}")
print(f"SPMM_AUTO_SPARSE_K_TILE={sparse_k_tile}")
PYTILING
    )"; then
        echo "Error: adaptive SPMM tiling report failed for ${indptr_file}." >&2
        exit 1
    fi

    eval "${selector_output}"
}

function ensure_existing_build {
    if [ ! -x "build/${EXECUTABLE}" ]; then
        echo "Error: rebuild=0 but build/${EXECUTABLE} does not exist or is not executable."
        echo "Hint: run once with rebuild=1:"
        echo "  $0 ${CASES_JSON} ${device_id} 1"
        exit 1
    fi

    if ! compgen -G "build/spmm_kernel*.o" >/dev/null; then
        echo "Error: rebuild=0 but no build/spmm_kernel*.o files were found."
        echo "Hint: run once with rebuild=1:"
        echo "  $0 ${CASES_JSON} ${device_id} 1"
        exit 1
    fi
}

function compile_once {
    local compile_log="${PROF_ROOT}/compile.log"

    if [ "${rebuild}" == "0" ]; then
        echo "Skipping compilation and reusing existing build/."
        ensure_existing_build
        return 0
    fi

    echo "Compiling all tiling-specialized kernels once..."
    if ! {
        make clean
        make KERNEL_CFLAGS="${KERNEL_CFLAGS:-}" HOST_CFLAGS="${HOST_CFLAGS:-}"
    } >"${compile_log}" 2>&1; then
        echo "Error: compilation failed."
        echo "Compile log: ${compile_log}"
        echo "----- compile.log tail -----"
        tail -n 160 "${compile_log}" || true
        echo "----------------------------"
        exit 1
    fi
}

function run_one_case {
    local case_name="$1"
    local M="$2"
    local N="$3"
    local K="$4"
    local sparsity="$5"
    local d="$6"

    echo ""
    echo "[case] ${case_name}: M=${M} N=${N} K=${K} sparsity=${sparsity} d=${d}"

    ensure_dataset "${case_name}"

    local case_prof_dir="${PROF_ROOT}/${case_name}"
    rm -rf "${case_prof_dir}"
    mkdir -p "${case_prof_dir}"

    prepare_full_dataset "${case_name}"
    select_case_tiling_for_report

    echo "  max_vectors_per_row=${SPMM_AUTO_MAX_VECTORS}"
    echo "  runtime dispatch should select AIC_N_BATCH=${SPMM_AUTO_AIC_N_BATCH}, SPARSE_K_TILE=${SPMM_AUTO_SPARSE_K_TILE}"

    echo "  Profiling with msprof..."
    if ! (
        cd build
        msprof \
            --application="./${EXECUTABLE} ${M} ${N} ${K} 0 ${device_id}" \
            --output="../${case_prof_dir}" \
            --task-time=on \
            --aicpu=on \
            --l2=on
    ) >"${case_prof_dir}/msprof.log" 2>&1; then
        echo "Error: msprof failed for ${case_name}."
        echo "msprof log: ${case_prof_dir}/msprof.log"
        echo "----- msprof.log tail -----"
        tail -n 120 "${case_prof_dir}/msprof.log" || true
        echo "---------------------------"
        exit 1
    fi

    local prof_args=("${M}" "${N}" "${K}" "${sparsity}" --prof-dir "${case_prof_dir}")
    if [ -n "${SPMM_MULTI_WARMUP:-}" ]; then
        prof_args+=(--warmup "${SPMM_MULTI_WARMUP}")
    fi
    if [ -n "${SPMM_MULTI_RUNS:-}" ]; then
        prof_args+=(--runs "${SPMM_MULTI_RUNS}")
    fi

    local prof_line
    if ! prof_line="$(python3 ./prof.py "${prof_args[@]}" | tail -n 1)"; then
        echo "Error: prof.py failed for ${case_name}."
        exit 1
    fi

    local out_M out_N out_K out_sparsity avg_ms effective_tflops dense_equiv_tflops
    read -r out_M out_N out_K out_sparsity avg_ms effective_tflops dense_equiv_tflops <<< "${prof_line}"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${case_name}" \
        "${out_M}" "${out_N}" "${out_K}" "${out_sparsity}" \
        "${avg_ms}" "${effective_tflops}" "${dense_equiv_tflops}" \
        "${SPMM_AUTO_MAX_VECTORS}" "${SPMM_AUTO_AIC_N_BATCH}" "${SPMM_AUTO_SPARSE_K_TILE}" \
        >> "${RESULTS_TSV}"

    echo "  summary: ${case_name} ${prof_line} max_vectors=${SPMM_AUTO_MAX_VECTORS} AIC_N_BATCH=${SPMM_AUTO_AIC_N_BATCH} SPARSE_K_TILE=${SPMM_AUTO_SPARSE_K_TILE}"
}

mkdir -p "${DATA_DIR}" build
if [ "${SPMM_MULTI_KEEP_OLD:-0}" != "1" ]; then
    rm -rf "${PROF_ROOT}"
fi
mkdir -p "${PROF_ROOT}"

printf "case_name\tM\tN\tK\tsparsity\tavg_ms\teffective_tflops\tdense_equiv_tflops\tmax_vectors_per_row\tAIC_N_BATCH\tSPARSE_K_TILE\n" > "${RESULTS_TSV}"

CASE_COUNT="$(load_cases | wc -l | tr -d ' ')"

echo "=========================================="
echo "SPMM Multi-Case Runtime-Dispatch Profiling"
echo "  Cases JSON: ${CASES_JSON_ABS}"
echo "  Selected cases: ${CASE_COUNT}"
echo "  Device:     ${device_id}"
echo "  Output:     ${RESULTS_TSV}"
if [ "${rebuild}" == "1" ]; then
    echo "  Compile:    rebuild once, all tiling variants"
else
    echo "  Compile:    skip, reuse existing build/"
fi
echo "=========================================="

compile_once

while IFS=$'\t' read -r case_name M N K sparsity d; do
    run_one_case "${case_name}" "${M}" "${N}" "${K}" "${sparsity}" "${d}"
done < <(load_cases)

echo ""
echo "Final summary"
echo "  Saved to: ${RESULTS_TSV}"
echo ""
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "${RESULTS_TSV}"
else
    cat "${RESULTS_TSV}"
fi

echo ""
echo "Done."
