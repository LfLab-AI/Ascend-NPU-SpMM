#!/usr/bin/env python3
"""Copy one full dataset case into ascblas/data for full verification/profiling."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent
ASCBLAS_DIR = TOOL_DIR.parent
DEFAULT_DATASETS = ASCBLAS_DIR / "datasets_full"
DEFAULT_TARGET = ASCBLAS_DIR / "data"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", help="Case name under ascblas/datasets_full")
    parser.add_argument("--datasets", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()

    source = args.datasets / args.case
    if not source.exists():
        raise FileNotFoundError(f"dataset case not found: {source}")

    if args.target.exists():
        shutil.rmtree(args.target)
    shutil.copytree(source, args.target)

    manifest_path = args.target / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(json.dumps({
            "case": manifest.get("name", args.case),
            "mode": manifest.get("mode", "full"),
            "M": manifest.get("M"),
            "N": manifest.get("N"),
            "K": manifest.get("K"),
            "sparsity": manifest.get("sparsity"),
            "d": manifest.get("d"),
            "target": str(args.target)
        }, indent=2, ensure_ascii=False))
    else:
        print(f"materialized {source} -> {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
