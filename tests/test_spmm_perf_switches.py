from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "ascblas" / "src" / "spmm_kernel.cce"
MAIN = ROOT / "ascblas" / "src" / "main.cpp"
MAKEFILE = ROOT / "ascblas" / "shell" / "makefile"
RUN_SH = ROOT / "ascblas" / "shell" / "run.sh"
TILING_SELECTOR = ROOT / "ascblas" / "shell" / "select_spmm_tiling.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_debug_detail_is_limited_to_first_core_and_first_tile():
    kernel = read(KERNEL)

    assert "bool trace_row = debug_enabled && (core_id == 0 && row_block == 0);" in kernel
    assert "bool write_detail = trace_block;" in kernel
    assert "core_id != 0" not in kernel


def test_batch_sizes_are_compile_time_overrides_shared_by_host_and_kernel():
    kernel = read(KERNEL)
    main = read(MAIN)
    makefile = read(MAKEFILE)

    assert "#ifndef SPMM_B_PACK_N_BATCH" in kernel
    assert "#ifndef SPMM_AIC_N_BATCH" in kernel
    assert "constexpr int32_t B_PACK_N_BATCH = SPMM_B_PACK_N_BATCH;" in kernel
    assert "constexpr int32_t AIC_N_BATCH = SPMM_AIC_N_BATCH;" in kernel

    assert "#ifndef SPMM_B_PACK_N_BATCH" in main
    assert "const int64_t B_PACK_N_BATCH_HOST = SPMM_B_PACK_N_BATCH;" in main

    assert "KERNEL_CFLAGS ?=" in makefile
    assert "$(KERNEL_CFLAGS)" in makefile
    assert "static_assert(B_PACK_N_BATCH % AIC_N_BATCH == 0" in kernel
    assert "static_assert(B_PACK_N_BATCH >= AIC_N_BATCH" in kernel


def test_profile_modes_and_pipe_barrier_switch_are_available():
    kernel = read(KERNEL)
    makefile = read(MAKEFILE)

    assert "#ifndef SPMM_PROFILE_AIC_ONLY" in kernel
    assert "#ifndef SPMM_PROFILE_AIV_ONLY" in kernel
    assert "#ifndef SPMM_DISABLE_PIPE_ALL_BARRIER" in kernel
    assert "spmm_pipe_all_barrier" in kernel

    assert "-DSPMM_PROFILE_AIC_ONLY" in makefile
    assert "-DSPMM_PROFILE_AIV_ONLY" in makefile
    assert "-DSPMM_DISABLE_PIPE_ALL_BARRIER" in makefile


def test_aic_only_mode_zeroes_workspace_before_kernel_reads_it():
    main = read(MAIN)

    assert "#ifndef SPMM_PROFILE_AIC_ONLY" in main
    assert "#if SPMM_PROFILE_AIC_ONLY" in main
    assert "aclrtMemset(d_b_pack_workspace, b_pack_workspace_size, 0, b_pack_workspace_size)" in main


def test_full_nbatch_pack_uses_one_strided_gm_write():
    kernel = read(KERNEL)
    start = kernel.index("void pack_b_selected_panels_full_nbatch_aiv")
    end = kernel.index("void pack_b_selected_panels_aiv", start)
    body = kernel[start:end]

    assert "B_PACK_BATCH_COLS" in body
    assert """ascblas_matrix_ubuf2gm(
        B_pack_batch_dst + vec_base,
        trans_base,
        B_PACK_SUB_TILE,
        B_PACK_BATCH_COLS,
        B_PACK_SUB_TILE,
        row_k_round
    );""" in body
    assert "for (int32_t batch_i = 0; batch_i < B_PACK_N_BATCH; ++batch_i) {\n        ascblas_matrix_ubuf2gm" not in body


def test_row_major_b_pack_experiment_changes_both_aiv_and_aic_layouts():
    kernel = read(KERNEL)

    assert "#ifndef SPMM_PACK_B_ROW_MAJOR" in kernel
    assert "#ifndef SPMM_PACK_B_ROW_MAJOR_L0A_TRANSPOSE" in kernel
    assert "#if SPMM_PACK_B_ROW_MAJOR" in kernel
    assert "copy_b_pack_row_major_to_l1" in kernel
    assert "pack_b_selected_panels_full_nbatch_row_major_aiv" in kernel
    assert "B_pack_batch_dst + (int64_t)vec_base * B_PACK_BATCH_COLS" in kernel
    assert "B_pack +\n                    row_pack_offset +\n                    (int64_t)pipe_slot * (int64_t)row_k_round * B_PACK_BATCH_COLS +\n                    (int64_t)local_n_block * BLOCK_SIZE" in kernel
    assert "(uint16_t)(batch_n_frac * k_tile_fracs)" in kernel
    assert "#if SPMM_PACK_B_ROW_MAJOR_L0A_TRANSPOSE\n                        true,\n#else\n                        false,\n#endif" in kernel
    assert "(SPMM_PACK_B_ROW_MAJOR_L0A_TRANSPOSE != 0)" not in kernel


def test_row_major_debug_dump_is_available_on_single_k_fast_path():
    kernel = read(KERNEL)
    start = kernel.index("if (single_k_tile_fast_path) {", kernel.index("for (int32_t n_block_idx = 0;"))
    end = kernel.index("continue;", start)
    body = kernel[start:end]

    assert "DBG_BC_DUMP_START" in body
    assert "DBG_L1A_DUMP_START" in body
    assert "DBG_L1B_DUMP_START" in body
    assert "detail_dumped = true;" in body


def test_run_script_can_override_verify_level_for_debug_dumps():
    run_sh = read(RUN_SH)

    assert "SPMM_VERIFY_LEVEL" in run_sh
    assert 'verifyLevel="${SPMM_VERIFY_LEVEL}"' in run_sh


def test_row_major_aiv_tail_tiles_do_not_fall_back_to_column_major_pack():
    kernel = read(KERNEL)

    assert "pack_b_selected_panels_row_major_direct_aiv" in kernel
    assert "B_pack_batch_dst +\n                                    (int64_t)vec_base * B_PACK_BATCH_COLS" in kernel
    assert "B_pack_batch_dst =\n                        workspace1" in kernel


def test_sparse_k_tile_is_bounded_by_l0a_and_l1_partition_capacity():
    kernel = read(KERNEL)
    main = read(MAIN)

    assert "#ifndef SPMM_SPARSE_K_TILE" in kernel
    assert "#define SPMM_SPARSE_K_TILE 256" in kernel
    assert "constexpr int32_t SPARSE_K_TILE = SPMM_SPARSE_K_TILE;" in kernel
    assert "constexpr int32_t L0A_CAPACITY_HALF_ELEMS" in kernel
    assert "constexpr int32_t L1_A_REGION_HALF_ELEMS" in kernel
    assert "AIC_BATCH_COLS * SPARSE_K_TILE <= L0A_CAPACITY_HALF_ELEMS" in kernel
    assert "AIC_BATCH_COLS * SPARSE_K_TILE <= L1_A_REGION_HALF_ELEMS" in kernel
    assert "#ifndef SPMM_SPARSE_K_TILE" in main
    assert "const int64_t SPARSE_K_TILE_HOST = SPMM_SPARSE_K_TILE;" in main


def test_run_script_selects_tiling_after_data_preparation():
    run_sh = read(RUN_SH)

    prepare_pos = run_sh.index("prepare_data")
    selector_pos = run_sh.index("select_spmm_tiling.py", prepare_pos)
    compile_pos = run_sh.index('echo "[2/3] Compiling..."', selector_pos)

    assert selector_pos < compile_pos
    assert '--kernel-cflags="${KERNEL_CFLAGS:-}"' in run_sh
    assert 'KERNEL_CFLAGS="${KERNEL_CFLAGS:-} ${SPMM_AUTO_CFLAGS}"' in run_sh
    assert 'make KERNEL_CFLAGS="${KERNEL_CFLAGS}" HOST_CFLAGS="${HOST_CFLAGS:-${KERNEL_CFLAGS}}"' in run_sh
    assert TILING_SELECTOR.name == "select_spmm_tiling.py"

