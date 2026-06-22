#!/usr/bin/env python3
"""Generate reusable full SpMM datasets.

Each case contains the complete files needed by the current full-verification
host path:

  A_dense.bin, B_dense.bin, C_dense.bin,
  csr/A_*.bin, vector_csr/A_*.bin, and metadata.

The implementation intentionally calls shell/spmm_gen_data.py's
generate_spmm_data(...) so the pre-generated datasets follow the same logic as
the known-correct full data path.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


TOOL_DIR = Path(__file__).resolve().parent
ASCBLAS_DIR = TOOL_DIR.parent
SHELL_DIR = ASCBLAS_DIR / "shell"
DEFAULT_CASES = TOOL_DIR / "full_benchmark_cases.json"
DEFAULT_OUT = ASCBLAS_DIR / "datasets_full"
CASE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["cases"] if isinstance(payload, dict) else payload


def build_custom_case(args: argparse.Namespace) -> Optional[dict]:
    values = {
        "name": args.name,
        "M": args.m,
        "N": args.n,
        "K": args.k,
        "sparsity": args.sparsity,
        "d": args.d,
    }
    provided = {key for key, value in values.items() if value is not None}
    if not provided:
        return None

    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise ValueError(
            "custom case requires all of --name, --m, --n, --k, --sparsity, and --d; "
            f"missing: {', '.join('--' + key.lower() for key in missing)}"
        )

    name = str(args.name)
    if not CASE_NAME_RE.match(name):
        raise ValueError(
            "--name may only contain letters, digits, underscore, dot, and dash, "
            "and must start with a letter or digit"
        )

    for key in ("m", "n", "k", "d"):
        if getattr(args, key) <= 0:
            raise ValueError(f"--{key} must be a positive integer")

    if args.sparsity < 0 or args.sparsity >= 100:
        raise ValueError("--sparsity must be in [0, 100)")

    if args.m % args.d != 0:
        raise ValueError("--m must be divisible by --d for vector-CSR row blocks")

    return {
        "name": name,
        "M": args.m,
        "N": args.n,
        "K": args.k,
        "sparsity": args.sparsity,
        "d": args.d,
        "source_hint": args.source_hint or "custom full dataset",
    }


def case_bytes(case: dict) -> dict:
    m, n, k = case["M"], case["N"], case["K"]
    return {
        "A_dense_bytes": m * k * 4,
        "B_dense_bytes": k * n * 4,
        "C_dense_bytes": m * n * 4,
        "dense_abc_bytes": (m * k + k * n + m * n) * 4
    }


def write_manifest(case_dir: Path, case: dict) -> None:
    manifest = {
        **case,
        "mode": "full",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "storage_estimate": case_bytes(case),
        "layout": {
            "A_dense.bin": "M x K row-major float32",
            "B_dense.bin": "K x N row-major float32",
            "C_dense.bin": "M x N row-major float32 reference, computed as A @ B",
            "vector_csr/A_data.bin": "vector blocks, float32, d contiguous values per vector",
            "vector_csr/A_cols.bin": "int32 column index per vector block",
            "vector_csr/A_indptr.bin": "int32 row-block pointer, length M/d + 1"
        },
        "files": sorted(
            str(p.relative_to(case_dir)).replace("\\", "/")
            for p in case_dir.rglob("*")
            if p.is_file()
        )
    }
    with (case_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def generate_case(case: dict, out_root: Path, force: bool) -> None:
    sys.path.insert(0, str(SHELL_DIR))
    import spmm_gen_data

    case_dir = out_root / case["name"]
    manifest = case_dir / "manifest.json"
    if manifest.exists() and not force:
        print(f"[skip] {case['name']}: already exists")
        return

    if case_dir.exists() and force:
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[gen] {case['name']}: M={case['M']} N={case['N']} K={case['K']} "
        f"sparsity={case['sparsity']} d={case['d']}"
    )
    print(f"      dense ABC estimate: {case_bytes(case)['dense_abc_bytes'] / (1024 ** 3):.2f} GiB")

    spmm_gen_data.generate_spmm_data(
        case["M"],
        case["N"],
        case["K"],
        case["sparsity"],
        case["d"],
        str(case_dir),
    )
    write_manifest(case_dir, case)
    print(f"[ok]  {case['name']}: {manifest}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate reusable full SpMM datasets with A/B/C, CSR, and vector-CSR files."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="JSON case list")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output dataset root")
    parser.add_argument("--only", nargs="*", help="Generate only named cases from --cases")
    parser.add_argument("--force", action="store_true", help="Regenerate existing output directories")
    parser.add_argument("--name", help="Custom case name")
    parser.add_argument("--m", type=int, help="Custom M dimension")
    parser.add_argument("--n", type=int, help="Custom N dimension")
    parser.add_argument("--k", type=int, help="Custom K dimension")
    parser.add_argument("--sparsity", type=int, help="Custom A sparsity percentage")
    parser.add_argument("--d", type=int, help="Custom vector block dimension")
    parser.add_argument("--source-hint", help="Optional custom case description")
    args = parser.parse_args()

    try:
        custom_case = build_custom_case(args)
    except ValueError as exc:
        parser.error(str(exc))
    if custom_case is not None and args.only:
        parser.error("--only cannot be used together with custom --name/--m/--n/--k/--sparsity/--d")

    args.out.mkdir(parents=True, exist_ok=True)

    if custom_case is not None:
        generate_case(custom_case, args.out, args.force)
        return 0

    selected = set(args.only) if args.only else None
    matched = 0
    for case in load_cases(args.cases):
        if selected is not None and case["name"] not in selected:
            continue
        matched += 1
        generate_case(case, args.out, args.force)
    if selected is not None and matched != len(selected):
        parser.error(f"unknown case name(s): {', '.join(sorted(selected))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
