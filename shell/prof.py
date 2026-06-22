#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse msprof task_time_*.csv and print one compact profiling line:

    M N K sparsity avg_ms effective_tflops dense_equiv_tflops

Default policy:
  - match kernel_name == spmm_kernel or variants whose names start with
    spmm_kernel_ (e.g. spmm_kernel_aic8_ktile256)
  - prefer kernel_type == MIX_AIC when present
  - prof mode launches 21 kernels: skip the first warmup, average the next 20
"""

import argparse
import csv
import glob
import os
import sys
from typing import Dict, List, Optional


def _to_float(value: str) -> float:
    return float(str(value).strip().replace("\t", ""))


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def find_latest_task_time_csv(prof_dir: str) -> str:
    patterns = [
        os.path.join(prof_dir, "PROF_*", "mindstudio_profiler_output", "task_time_*.csv"),
        os.path.join(prof_dir, "**", "task_time_*.csv"),
    ]

    candidates: List[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern, recursive=True))

    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find task_time_*.csv under prof_dir={prof_dir!r}"
        )

    return max(candidates, key=lambda p: os.path.getmtime(p))


def kernel_name_matches(actual: str, requested: str, exact: bool) -> bool:
    if exact:
        return actual == requested
    if actual == requested:
        return True
    # Runtime-dispatch build launches specialized kernels such as
    # spmm_kernel_aic8_ktile256.  Keep prof.py backward-compatible by letting
    # the default requested name match all spmm_kernel_* variants.
    if requested == "spmm_kernel" and actual.startswith("spmm_kernel_"):
        return True
    return False


def read_kernel_rows(
    csv_path: str,
    kernel_name: str,
    preferred_kernel_type: str,
    exact_kernel_name: bool,
) -> List[Dict[str, str]]:
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            name = str(row.get("kernel_name", "")).strip()
            if not kernel_name_matches(name, kernel_name, exact_kernel_name):
                continue

            try:
                task_time_us = _to_float(row.get("task_time(us)", "0"))
            except Exception:
                continue

            if task_time_us <= 0.0:
                continue

            row["_task_time_us"] = task_time_us
            row["_task_start_us"] = _to_float(row.get("task_start(us)", "0") or "0")
            row["_task_id_int"] = _to_int(row.get("task_id", "0"))
            row["_kernel_name"] = name
            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No valid kernel rows found in {csv_path!r} for kernel_name={kernel_name!r}"
        )

    preferred = [
        r for r in rows
        if str(r.get("kernel_type", "")).strip() == preferred_kernel_type
    ]

    # In MIX kernels, task_time usually reports the AIC side as the full kernel
    # duration. Prefer MIX_AIC when available to avoid double-counting AIV/AIC rows.
    selected = preferred if preferred else rows

    selected.sort(key=lambda r: (r.get("_task_start_us", 0.0), r.get("_task_id_int", 0)))
    return selected


def compute_average_ms(
    rows: List[Dict[str, str]],
    warmup: int,
    expected_runs: int,
    strict: bool,
) -> float:
    required = warmup + expected_runs

    if len(rows) <= warmup:
        raise RuntimeError(
            f"Only {len(rows)} kernel rows found, cannot skip warmup={warmup}"
        )

    if len(rows) < required:
        message = (
            f"Warning: expected at least {required} kernel rows "
            f"({warmup} warmup + {expected_runs} measured), got {len(rows)}. "
            f"Averaging all rows after warmup."
        )
        if strict:
            raise RuntimeError(message)
        print(message, file=sys.stderr)
        measured = rows[warmup:]
    else:
        measured = rows[warmup:required]

    avg_us = sum(float(r["_task_time_us"]) for r in measured) / float(len(measured))
    return avg_us / 1000.0


def infer_actual_nnz_from_vector_csr(path: str) -> Optional[int]:
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    # vector_csr/A_data.bin is generated as float32 values, one value per real
    # nonzero A element.
    if size <= 0 or size % 4 != 0:
        return None
    return size // 4


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print SPMM profiling summary: M N K sparsity avg_ms effective_tflops dense_equiv_tflops"
    )
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("sparsity", type=float)
    parser.add_argument("--prof-dir", default="prof")
    parser.add_argument("--csv", default=None, help="Optional explicit task_time_*.csv path")
    parser.add_argument("--kernel-name", default="spmm_kernel")
    parser.add_argument("--exact-kernel-name", action="store_true")
    parser.add_argument("--preferred-kernel-type", default="MIX_AIC")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--header", action="store_true", help="Print a header before the data line")
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--a-data", default="../../data/vector_csr/A_data.bin",
                        help="vector_csr/A_data.bin path used to infer actual nnz(A)")

    args = parser.parse_args()

    csv_path = args.csv if args.csv else find_latest_task_time_csv(args.prof_dir)
    rows = read_kernel_rows(
        csv_path,
        args.kernel_name,
        args.preferred_kernel_type,
        args.exact_kernel_name,
    )
    avg_ms = compute_average_ms(rows, args.warmup, args.runs, args.strict)

    dense_ops = 2.0 * float(args.M) * float(args.N) * float(args.K)
    dense_equiv_tflops = dense_ops / (avg_ms * 1.0e9)

    actual_nnz = infer_actual_nnz_from_vector_csr(args.a_data)
    if actual_nnz is None:
        actual_nnz = int(round(float(args.M) * float(args.K) * (1.0 - args.sparsity / 100.0)))
    effective_ops = 2.0 * float(actual_nnz) * float(args.N)
    effective_tflops = effective_ops / (avg_ms * 1.0e9)

    if args.header:
        print("M N K sparsity avg_ms effective_tflops dense_equiv_tflops")

    sparsity_text = (
        str(int(args.sparsity))
        if float(args.sparsity).is_integer()
        else str(args.sparsity)
    )
    print(
        f"{args.M} {args.N} {args.K} {sparsity_text} "
        f"{avg_ms:.{args.precision}f} "
        f"{effective_tflops:.{args.precision}f} "
        f"{dense_equiv_tflops:.{args.precision}f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[prof.py] Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
