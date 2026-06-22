import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SELECTOR_PATH = ROOT / "ascblas" / "shell" / "select_spmm_tiling.py"


def load_selector():
    spec = importlib.util.spec_from_file_location("select_spmm_tiling", SELECTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selects_tiling_from_actual_max_vectors():
    selector = load_selector()

    assert selector.select_default_tiling(238) == (8, 256)
    assert selector.select_default_tiling(410) == (4, 512)
    assert selector.select_default_tiling(819) == (2, 1024)
    assert selector.select_default_tiling(2048) == (1, 2048)
    assert selector.select_default_tiling(4096) == (1, 2048)


def test_reads_max_vectors_from_indptr(tmp_path):
    selector = load_selector()
    indptr = np.array([0, 184, 387, 600], dtype=np.int32)
    path = tmp_path / "A_indptr.bin"
    indptr.tofile(path)

    assert selector.read_max_vectors_per_row(path) == 213


def test_manual_aic_batch_is_preserved_and_missing_k_tile_is_clamped():
    selector = load_selector()

    result = selector.resolve_tiling(
        max_vectors=410,
        kernel_cflags="-DSPMM_PACK_B_ROW_MAJOR=1 -DSPMM_AIC_N_BATCH=8",
    )

    assert result.aic_n_batch == 8
    assert result.sparse_k_tile == 256
    assert result.added_flags == ("-DSPMM_SPARSE_K_TILE=256",)
    assert result.source == "partial-manual"


def test_manual_k_tile_is_preserved_and_missing_aic_batch_is_clamped():
    selector = load_selector()

    result = selector.resolve_tiling(
        max_vectors=238,
        kernel_cflags="-DSPMM_PACK_B_ROW_MAJOR=1 -DSPMM_SPARSE_K_TILE=512",
    )

    assert result.aic_n_batch == 4
    assert result.sparse_k_tile == 512
    assert result.added_flags == ("-DSPMM_AIC_N_BATCH=4",)
    assert result.source == "partial-manual"


def test_unsafe_manual_pair_is_rejected():
    selector = load_selector()

    with pytest.raises(ValueError, match="exceeds 64KB"):
        selector.resolve_tiling(
            max_vectors=410,
            kernel_cflags="-DSPMM_AIC_N_BATCH=8 -DSPMM_SPARSE_K_TILE=512",
        )


