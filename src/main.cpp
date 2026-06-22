#include <acl/acl.h>
#include "spmm.h"
#include "sr_bcrs_utils.h"
#include "file_utils.h"
#include "data_utils.h"

#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <chrono>

#ifndef SPMM_B_PACK_N_BATCH
#define SPMM_B_PACK_N_BATCH 64
#endif

#ifndef SPMM_AIC_N_BATCH
#define SPMM_AIC_N_BATCH 8
#endif

#ifndef SPMM_B_PACK_PIPE_DEPTH
#define SPMM_B_PACK_PIPE_DEPTH 2
#endif

#ifndef SPMM_SPARSE_K_TILE
#define SPMM_SPARSE_K_TILE 256
#endif

#ifndef SPMM_PROFILE_AIC_ONLY
#define SPMM_PROFILE_AIC_ONLY 0
#endif

using HostClock = std::chrono::high_resolution_clock;

struct HostWallTiming
{
    double acl_init_handle = 0.0;
    double host_malloc_init = 0.0;
    double read_a_vector_csr = 0.0;
    double read_b_dense = 0.0;
    double read_c_reference = 0.0;
    double analyze_sparse_metadata = 0.0;
    double pack_b_host_once = 0.0;
    double debug_host_alloc_init = 0.0;
    double device_malloc = 0.0;
    double h2d_memcpy = 0.0;
    double device_memset = 0.0;
    double kernel_loop_sync = 0.0;
    double d2h_c_memcpy = 0.0;
    double d2h_debug_memcpy = 0.0;
    double debug_dump_printing = 0.0;
    double verification = 0.0;
    double debug_summary_printing = 0.0;
    double cleanup = 0.0;
};

static double elapsed_ms(HostClock::time_point start, HostClock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct ScopedTimer
{
    explicit ScopedTimer(double &target) : target(target), start(HostClock::now()) {}
    ~ScopedTimer()
    {
        target += elapsed_ms(start, HostClock::now());
    }

    double &target;
    HostClock::time_point start;
};

static void print_host_wall_time_breakdown(
    const HostWallTiming &timing,
    int launch_count,
    double total_inside_main_ms)
{
    double kernel_avg = launch_count > 0
        ? timing.kernel_loop_sync / static_cast<double>(launch_count)
        : 0.0;

    printf("\n===== Host Wall-Time Breakdown =====\n");
    printf("  %-24s : %.3f ms\n", "acl init + handle", timing.acl_init_handle);
    printf("  %-24s : %.3f ms\n", "host malloc/init", timing.host_malloc_init);
    printf("  %-24s : %.3f ms\n", "read A vector CSR", timing.read_a_vector_csr);
    printf("  %-24s : %.3f ms\n", "read B dense", timing.read_b_dense);
    printf("  %-24s : %.3f ms\n", "read C reference", timing.read_c_reference);
    printf("  %-24s : %.3f ms\n", "analyze sparse metadata", timing.analyze_sparse_metadata);
    printf("  %-24s : %.3f ms\n", "pack B on host (once)", timing.pack_b_host_once);
    printf("  %-24s : %.3f ms\n", "debug host alloc/init", timing.debug_host_alloc_init);
    printf("  %-24s : %.3f ms\n", "device malloc", timing.device_malloc);
    printf("  %-24s : %.3f ms\n", "H2D memcpy", timing.h2d_memcpy);
    printf("  %-24s : %.3f ms\n", "device memset", timing.device_memset);
    printf("  %-24s : %.3f ms\n", "kernel loop + sync", timing.kernel_loop_sync);
    printf("  %-24s : %.3f ms (%d launches)\n", "kernel avg per launch", kernel_avg, launch_count);
    printf("  %-24s : %.3f ms\n", "D2H C memcpy", timing.d2h_c_memcpy);
    printf("  %-24s : %.3f ms\n", "D2H debug memcpy", timing.d2h_debug_memcpy);
    printf("  %-24s : %.3f ms\n", "debug dump printing", timing.debug_dump_printing);
    printf("  %-24s : %.3f ms\n", "verification", timing.verification);
    printf("  %-24s : %.3f ms\n", "debug summary printing", timing.debug_summary_printing);
    printf("  %-24s : %.3f ms\n", "cleanup", timing.cleanup);
    printf("  %-24s : %.3f ms\n", "total inside main", total_inside_main_ms);
    printf("====================================\n\n");
}

// deviceId 表示程序运行在第几号卡上
int deviceId = 0;

// 0: 不验证结果，1: 验证结果并打印少量错误信息，2: 验证结果并打印更多错误信息
int verifyLevel = 0;

int main(int argc, char **argv)
{
    HostWallTiming timing;
    HostClock::time_point main_start = HostClock::now();
    int launch_count = 0;

    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K> [verifyLevel] [deviceId]" << std::endl;
        return -1;
    }

    int32_t M = std::stoi(argv[1]);
    int32_t N = std::stoi(argv[2]);
    int32_t K = std::stoi(argv[3]);

    if (argc > 4)
        verifyLevel = std::stoi(argv[4]);
    if (argc > 5)
        deviceId = std::stoi(argv[5]);

    const int vec_length = 16;
    // These are common across all compiled tiling variants.
    const int64_t B_PACK_N_BATCH_HOST = SPMM_B_PACK_N_BATCH;
    const int64_t B_PACK_PIPE_DEPTH_HOST = SPMM_B_PACK_PIPE_DEPTH;
    const int64_t B_PACK_BATCH_COLS_HOST = B_PACK_N_BATCH_HOST * vec_length;
    const bool needVerify = verifyLevel > 0;
    const bool needDebugDump = verifyLevel >= 2;
    const int32_t debug_enable = verifyLevel > 0 ? 1 : 0;

    if (M <= 0 || N <= 0 || K <= 0)
    {
        std::cerr << "Invalid matrix shape: M=" << M << ", N=" << N << ", K=" << K << std::endl;
        return -1;
    }

    const int32_t N_original = N;
    int32_t N_kernel = N_original;
    bool n_padded = false;
    int64_t AIC_BATCH_COLS_HOST = vec_length;
    SpmmTilingConfig selected_tiling = SelectSpmmTilingConfig(0);

    size_t B_count_input = 0;
    size_t B_count_kernel = 0;
    size_t C_count_ref = 0;
    size_t C_count_kernel = 0;
    size_t B_size_input = 0;
    size_t B_size_kernel = 0;
    size_t C_size_ref = 0;
    size_t C_size_kernel = 0;

    aclrtStream stream;
    ascblasHandle_t handle;
    {
        ScopedTimer timer(timing.acl_init_handle);
        CALL_RT(aclInit(nullptr));
        CALL_RT(aclrtSetDevice(deviceId));
        ascblasCreate(&handle);
        ascblasGetStream(handle, &stream);
    }

    __fp16 *h_B = nullptr;

    // h_C_ref: C_dense.bin 参考结果，当前数据文件表现为 row-major
    // h_C_out: kernel 输出结果，目标布局为 column-major
    float *h_C_ref = nullptr;
    float *h_C_out = nullptr;

    int32_t *debug_out_dev = nullptr;
    int32_t *debug_out_host = nullptr;

    std::vector<__fp16> values;
    std::vector<int32_t> col_indices;
    std::vector<int32_t> row_offsets;

    int32_t A_num_vectors = 0;
    int32_t A_d = 0;

    {
        ScopedTimer timer(timing.read_a_vector_csr);
        ReadAMatrixVectorCSR(M, A_d, values, col_indices, row_offsets, A_num_vectors);
    }
    int64_t num_row_blocks = 0;
    int64_t num_n_blocks = 0;
    int64_t max_vectors_per_row = 0;
    int64_t groupDim = 0;
    int64_t total_b_pack_elems = 0;
    std::vector<int64_t> b_pack_offsets;

    {
    ScopedTimer timer(timing.analyze_sparse_metadata);

    num_row_blocks = (static_cast<int64_t>(M) + vec_length - 1) / vec_length;

    if (static_cast<int64_t>(row_offsets.size()) < num_row_blocks + 1)
    {
        std::cerr << "Invalid row_offsets size. row_offsets.size() = "
                  << row_offsets.size()
                  << ", expected at least "
                  << (num_row_blocks + 1)
                  << std::endl;
        return -1;
    }

    if (A_d != vec_length)
    {
        std::cerr << "Warning: A_d = " << A_d
                  << ", expected vec_length = " << vec_length
                  << ". Continue anyway." << std::endl;
    }

    for (int64_t i = 0; i < num_row_blocks; ++i)
    {
        int64_t start = static_cast<int64_t>(row_offsets[i]);
        int64_t end = static_cast<int64_t>(row_offsets[i + 1]);
        int64_t cnt = end - start;

        if (cnt < 0)
        {
            std::cerr << "Invalid row_offsets: row_offsets[" << i << "] = " << start
                      << ", row_offsets[" << (i + 1) << "] = " << end << std::endl;
            return -1;
        }

        if (cnt > max_vectors_per_row)
        {
            max_vectors_per_row = cnt;
        }
    }

    selected_tiling = SelectSpmmTilingConfig(max_vectors_per_row);
    AIC_BATCH_COLS_HOST = static_cast<int64_t>(selected_tiling.aic_n_batch) * vec_length;
    N_kernel = static_cast<int32_t>(
        ((static_cast<int64_t>(N_original) + AIC_BATCH_COLS_HOST - 1) /
         AIC_BATCH_COLS_HOST) *
        AIC_BATCH_COLS_HOST);
    n_padded = (N_kernel != N_original);
    num_n_blocks = (static_cast<int64_t>(N_kernel) + vec_length - 1) / vec_length;

    B_count_input = static_cast<size_t>(K) * static_cast<size_t>(N_original);
    B_count_kernel = static_cast<size_t>(K) * static_cast<size_t>(N_kernel);
    C_count_ref = static_cast<size_t>(M) * static_cast<size_t>(N_original);
    C_count_kernel = static_cast<size_t>(M) * static_cast<size_t>(N_kernel);
    B_size_input = B_count_input * sizeof(__fp16);
    B_size_kernel = B_count_kernel * sizeof(__fp16);
    C_size_ref = C_count_ref * sizeof(float);
    C_size_kernel = C_count_kernel * sizeof(float);

    std::cout << "Runtime tiling dispatch: kernel=" << selected_tiling.kernel_name
              << ", aic_n_batch=" << selected_tiling.aic_n_batch
              << ", sparse_k_tile=" << selected_tiling.sparse_k_tile
              << ", max_vectors_per_row=" << max_vectors_per_row
              << std::endl;
    if (n_padded)
    {
        std::cout << "Host-side N padding enabled: N_original=" << N_original
                  << ", N_kernel=" << N_kernel
                  << " (pad " << (N_kernel - N_original) << " columns, "
                  << "granularity=" << AIC_BATCH_COLS_HOST << ")"
                  << std::endl;
    }

    groupDim = num_row_blocks < CORENUM ? num_row_blocks : CORENUM;
    if (groupDim <= 0)
    {
        std::cerr << "Invalid groupDim = " << groupDim << std::endl;
        return -1;
    }

    b_pack_offsets.assign(num_row_blocks + 1, 0);
    for (int64_t i = 0; i < num_row_blocks; ++i)
    {
        int64_t cnt =
            static_cast<int64_t>(row_offsets[i + 1]) -
            static_cast<int64_t>(row_offsets[i]);
        int64_t row_k_round = ((cnt + 15) / 16) * 16;
        // AIV/AIC double-buffered GM workspace layout:
        //   per row_block: B_PACK_PIPE_DEPTH_HOST slots
        //   per slot     : up to B_PACK_N_BATCH_HOST n-blocks.
        // The n_batch index is mapped to slot = n_batch_idx % 2 in the kernel.
        b_pack_offsets[i + 1] =
            b_pack_offsets[i] +
            B_PACK_PIPE_DEPTH_HOST * row_k_round * B_PACK_BATCH_COLS_HOST;
    }
    total_b_pack_elems = b_pack_offsets.back();
    std::cout << "B-pack workspace: double-buffered ring, slots="
              << B_PACK_PIPE_DEPTH_HOST << ", n_batch=" << B_PACK_N_BATCH_HOST
              << ", aic_n_batch=" << selected_tiling.aic_n_batch
              << ", sparse_k_tile=" << selected_tiling.sparse_k_tile
              << ", total_b_pack_elems=" << total_b_pack_elems << std::endl;
    }

    {
        ScopedTimer timer(timing.host_malloc_init);
        CALL_RT(aclrtMallocHost((void **)(&h_B), B_size_kernel));
        std::memset(h_B, 0, B_size_kernel);
        if (needVerify)
        {
            CALL_RT(aclrtMallocHost((void **)(&h_C_ref), C_size_ref));
            CALL_RT(aclrtMallocHost((void **)(&h_C_out), C_size_kernel));
            std::memset(h_C_ref, 0, C_size_ref);
            std::memset(h_C_out, 0, C_size_kernel);
        }
    }

    {
        ScopedTimer timer(timing.read_b_dense);
        if (!n_padded)
        {
            ReadFloat32ToFp16("../../data/B_dense.bin", h_B, B_count_input);
        }
        else
        {
            __fp16 *h_B_input = nullptr;
            CALL_RT(aclrtMallocHost((void **)(&h_B_input), B_size_input));
            ReadFloat32ToFp16("../../data/B_dense.bin", h_B_input, B_count_input);

            // Input B_dense.bin is K x N_original row-major.  The selected
            // kernel is launched with N_kernel = ceil_to(N_original,
            // selected_aic_n_batch * 16), so B must be materialized as
            // K x N_kernel row-major with zero padding in the extra columns.
            for (int32_t k_row = 0; k_row < K; ++k_row)
            {
                std::memcpy(
                    h_B + static_cast<int64_t>(k_row) * N_kernel,
                    h_B_input + static_cast<int64_t>(k_row) * N_original,
                    static_cast<size_t>(N_original) * sizeof(__fp16));
            }

            CALL_RT(aclrtFreeHost(h_B_input));
        }
    }
    if (needVerify)
    {
        ScopedTimer timer(timing.read_c_reference);
        ReadFloat32("../../data/C_dense.bin", h_C_ref, C_count_ref);
    }

    /*
     * 与新版 spmm_kernel.cce 中的 DBG_SLOTS 保持一致。
     *
     * 0   ~ 31  : 标量调试字段
     * 32  ~ 47  : row_offsets dump
     * 48  ~ 63  : col_indices dump
     * 64  ~ 127 : A_src GM raw dump, 128 fp16
     * 128 ~ 191 : B workspace GM memory dump, 128 fp16
     * 192 ~ 255 : B_selected logical Kx16 dump, 128 fp16
     * 256 ~ 319 : A_selected logical Kx16 dump, 128 fp16
     * 320 ~ 323 : scalar reference C[0,0], C[1,0], C[0,1], C[1,1]
     */
    const int DEBUG_SLOTS = 384;
    const int64_t DEBUG_SLOTS_PER_CORE = DEBUG_SLOTS;
    size_t debug_size = static_cast<size_t>(groupDim) *
                        static_cast<size_t>(DEBUG_SLOTS_PER_CORE) *
                        sizeof(int32_t);

    if (debug_enable)
    {
        ScopedTimer timer(timing.debug_host_alloc_init);
        CALL_RT(aclrtMallocHost((void **)(&debug_out_host), debug_size));
        std::memset(debug_out_host, 0, debug_size);
    }

    size_t values_size = values.size() * sizeof(__fp16);
    size_t col_indices_size = col_indices.size() * sizeof(int32_t);
    size_t row_offsets_size = row_offsets.size() * sizeof(int32_t);
    size_t b_pack_offsets_size = b_pack_offsets.size() * sizeof(int64_t);
    size_t b_pack_workspace_size =
        static_cast<size_t>(total_b_pack_elems) * sizeof(__fp16);
    b_pack_workspace_size = ((b_pack_workspace_size + 511) / 512) * 512;
    if (b_pack_workspace_size == 0)
    {
        b_pack_workspace_size = 512;
    }

    __fp16 *d_values = nullptr;
    int32_t *d_col_indices = nullptr;
    int32_t *d_row_offsets = nullptr;
    int64_t *d_b_pack_offsets = nullptr;
    __fp16 *d_b_pack_workspace = nullptr;
    __fp16 *d_B = nullptr;
    float *d_C = nullptr;

    {
        ScopedTimer timer(timing.device_malloc);
        CALL_RT(aclrtMalloc((void **)(&d_values), values_size, ACL_MEM_MALLOC_HUGE_FIRST));
        CALL_RT(aclrtMalloc((void **)(&d_col_indices), col_indices_size, ACL_MEM_MALLOC_HUGE_FIRST));
        CALL_RT(aclrtMalloc((void **)(&d_row_offsets), row_offsets_size, ACL_MEM_MALLOC_HUGE_FIRST));
        CALL_RT(aclrtMalloc((void **)(&d_b_pack_offsets), b_pack_offsets_size, ACL_MEM_MALLOC_HUGE_FIRST));
        CALL_RT(aclrtMalloc((void **)(&d_b_pack_workspace), b_pack_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
        CALL_RT(aclrtMalloc((void **)(&d_B), B_size_kernel, ACL_MEM_MALLOC_HUGE_FIRST));
        CALL_RT(aclrtMalloc((void **)(&d_C), C_size_kernel, ACL_MEM_MALLOC_HUGE_FIRST));
        if (debug_enable)
        {
            CALL_RT(aclrtMalloc((void **)(&debug_out_dev), debug_size, ACL_MEM_MALLOC_HUGE_FIRST));
        }
    }

    {
        ScopedTimer timer(timing.h2d_memcpy);
        CALL_RT(aclrtMemcpy(d_values, values_size, values.data(), values_size, ACL_MEMCPY_HOST_TO_DEVICE));
        CALL_RT(aclrtMemcpy(d_col_indices, col_indices_size, col_indices.data(), col_indices_size, ACL_MEMCPY_HOST_TO_DEVICE));
        CALL_RT(aclrtMemcpy(d_row_offsets, row_offsets_size, row_offsets.data(), row_offsets_size, ACL_MEMCPY_HOST_TO_DEVICE));
        CALL_RT(aclrtMemcpy(d_b_pack_offsets, b_pack_offsets_size, b_pack_offsets.data(), b_pack_offsets_size, ACL_MEMCPY_HOST_TO_DEVICE));
        CALL_RT(aclrtMemcpy(d_B, B_size_kernel, h_B, B_size_kernel, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // 初始化输出；debug buffer 只在验证模式下使用，避免污染 prof 模式性能数据。
    {
        ScopedTimer timer(timing.device_memset);
        CALL_RT(aclrtMemset(d_C, C_size_kernel, 0, C_size_kernel));
#if SPMM_PROFILE_AIC_ONLY
        CALL_RT(aclrtMemset(d_b_pack_workspace, b_pack_workspace_size, 0, b_pack_workspace_size));
#endif
        if (debug_enable)
        {
            CALL_RT(aclrtMemset(debug_out_dev, debug_size, 0, debug_size));
        }
    }

    std::cout << "Kernel start!" << std::endl;

    if (needDebugDump)
    {
    ScopedTimer timer(timing.debug_dump_printing);
    printf("\n===== Host Side Sanity Check =====\n");
    printf("M=%d, N_original=%d, N_kernel=%d, K=%d, vec_length=%d, aic_batch_cols=%lld\n",
           M, N_original, N_kernel, K, vec_length,
           static_cast<long long>(AIC_BATCH_COLS_HOST));
    printf("B_count_input=%zu, B_count_kernel=%zu, C_count_ref=%zu, C_count_kernel=%zu\n",
           B_count_input, B_count_kernel, C_count_ref, C_count_kernel);
    printf("values.size()=%zu, col_indices.size()=%zu, row_offsets.size()=%zu\n",
           values.size(), col_indices.size(), row_offsets.size());
    printf("A_num_vectors=%d, A_d=%d\n", A_num_vectors, A_d);
    printf("num_row_blocks=%lld, num_n_blocks=%lld, groupDim=%lld\n",
           static_cast<long long>(num_row_blocks),
           static_cast<long long>(num_n_blocks),
           static_cast<long long>(groupDim));
    printf("max_vectors_per_row=%lld\n", static_cast<long long>(max_vectors_per_row));

    if (!row_offsets.empty())
    {
        printf("row_offsets[0]=%d\n", row_offsets[0]);
    }
    if (row_offsets.size() > 1)
    {
        printf("row_offsets[1]=%d\n", row_offsets[1]);
    }
    if (static_cast<int64_t>(row_offsets.size()) > num_row_blocks)
    {
        printf("row_offsets[num_row_blocks]=%d\n", row_offsets[num_row_blocks]);
    }
    if (!col_indices.empty())
    {
        printf("col_indices[0]=%d\n", col_indices[0]);
    }
    if (col_indices.size() > 1)
    {
        printf("col_indices[1]=%d\n", col_indices[1]);
    }

    printf("\nrow_offsets dump on host:\n");
    int row_dump_count = static_cast<int>(num_row_blocks + 1);
    if (row_dump_count > 32)
    {
        row_dump_count = 32;
    }
    for (int i = 0; i < row_dump_count; ++i)
    {
        printf("%d ", row_offsets[i]);
        if ((i + 1) % 16 == 0)
            printf("\n");
    }
    printf("\n");

    printf("\ncol_indices dump on host:\n");
    int col_dump_count = static_cast<int>(col_indices.size());
    if (col_dump_count > 32)
    {
        col_dump_count = 32;
    }
    for (int i = 0; i < col_dump_count; ++i)
    {
        printf("%d ", col_indices[i]);
        if ((i + 1) % 16 == 0)
            printf("\n");
    }
    printf("\n");
    }

    launch_count = (verifyLevel == 0) ? 21 : 1;
    if (verifyLevel == 0)
    {
        std::cout << "Prof mode: launch spmm kernel " << launch_count
                  << " times for stable msprof statistics." << std::endl;
    }

    {
        ScopedTimer timer(timing.kernel_loop_sync);
        for (int launch_idx = 0; launch_idx < launch_count; ++launch_idx)
        {
            CALL_RT(ascblasSpmm(
                handle,
                M,
                N_kernel,
                K,
                vec_length,
                d_values,
                max_vectors_per_row,
                d_col_indices,
                d_row_offsets,
                d_B,
                d_C,
                d_b_pack_offsets,
                total_b_pack_elems,
                d_b_pack_workspace,
                debug_out_dev,
                debug_enable));
        }

        CALL_RT(aclrtSynchronizeStream(stream));
    }

    if (needVerify)
    {
        ScopedTimer timer(timing.d2h_c_memcpy);
        CALL_RT(aclrtMemcpy(h_C_out, C_size_kernel, d_C, C_size_kernel, ACL_MEMCPY_DEVICE_TO_HOST));
    }
    if (debug_enable && debug_out_host != nullptr && debug_out_dev != nullptr)
    {
        ScopedTimer timer(timing.d2h_debug_memcpy);
        CALL_RT(aclrtMemcpy(debug_out_host, debug_size, debug_out_dev, debug_size, ACL_MEMCPY_DEVICE_TO_HOST));
    }

    if (needDebugDump)
    {
        ScopedTimer timer(timing.debug_dump_printing);
        printf("\nB matrix first 16 values (fp16 as float):\n");
        for (size_t i = 0; i < 16 && i < B_count_kernel; ++i)
        {
            printf("%f ", static_cast<float>(h_B[i]));
            if ((i + 1) % 8 == 0)
                printf("\n");
        }
        printf("\n");

        printf("C_ref row-major first row, first 16 cols:\n");
        for (int j = 0; j < 16 && j < N; ++j)
        {
            printf("%f ", h_C_ref[j]);
            if ((j + 1) % 8 == 0)
                printf("\n");
        }
        printf("\n");

        printf("C_out column-major first column, first 16 rows:\n");
        for (int i = 0; i < 16 && i < M; ++i)
        {
            printf("%f ", h_C_out[i]);
            if ((i + 1) % 8 == 0)
                printf("\n");
        }
        printf("\n");

        printf("C_out row 0, first 16 cols, read from column-major output:\n");
        for (int j = 0; j < 16 && j < N; ++j)
        {
            printf("%f ", h_C_out[static_cast<int64_t>(j) * M + 0]);
            if ((j + 1) % 8 == 0)
                printf("\n");
        }
        printf("\n");
    }

    bool verification_passed = true;
    auto verify_column_major_output_against_row_major_ref =
        [&](const float *out_col_major,
            const float *ref_row_major,
            int32_t rows,
            int32_t cols,
            int max_print) -> bool {
            double max_abs_err = 0.0;
            double max_rel_err_at_max_abs = 0.0;
            double max_rel_err = 0.0;
            double sum_abs_err = 0.0;

            int64_t max_abs_i = 0;
            int64_t max_abs_j = 0;
            int64_t max_rel_i = 0;
            int64_t max_rel_j = 0;
            int64_t mismatch_count = 0;
            int64_t total = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);

            const char *abs_tol_env = std::getenv("SPMM_VERIFY_ABS_TOL");
            const char *rel_tol_env = std::getenv("SPMM_VERIFY_REL_TOL");
            const double abs_tol = abs_tol_env ? std::atof(abs_tol_env) : 2.0e-1;
            const double rel_tol = rel_tol_env ? std::atof(rel_tol_env) : 2.0e-2;

            for (int32_t i = 0; i < rows; ++i)
            {
                for (int32_t j = 0; j < cols; ++j)
                {
                    float got = out_col_major[static_cast<int64_t>(j) * rows + i];
                    float ref = ref_row_major[static_cast<int64_t>(i) * cols + j];

                    double abs_err = std::fabs(static_cast<double>(got) - static_cast<double>(ref));
                    double rel_err = abs_err / std::max(1.0, std::fabs(static_cast<double>(ref)));

                    sum_abs_err += abs_err;

                    if (abs_err > max_abs_err)
                    {
                        max_abs_err = abs_err;
                        max_rel_err_at_max_abs = rel_err;
                        max_abs_i = i;
                        max_abs_j = j;
                    }

                    if (rel_err > max_rel_err)
                    {
                        max_rel_err = rel_err;
                        max_rel_i = i;
                        max_rel_j = j;
                    }

                    if (abs_err > abs_tol && rel_err > rel_tol)
                    {
                        if (mismatch_count < max_print)
                        {
                            printf("[Mismatch %lld] C[%d,%d]: got=%f, ref=%f, abs=%e, rel=%e\n",
                                   static_cast<long long>(mismatch_count),
                                   i,
                                   j,
                                   got,
                                   ref,
                                   abs_err,
                                   rel_err);
                        }

                        ++mismatch_count;
                    }
                }
            }

            printf("\n===== Layout-aware Verification =====\n");
            printf("Output layout : column-major, out[j * M + i]\n");
            printf("Ref layout    : row-major,    ref[i * N + j]\n");
            printf("Abs tol       : %.3e\n", abs_tol);
            printf("Rel tol       : %.3e\n", rel_tol);
            printf("Total elems   : %lld\n", static_cast<long long>(total));
            printf("Mismatch cnt  : %lld\n", static_cast<long long>(mismatch_count));
            printf("Max abs err   : %.9e at C[%lld,%lld]\n",
                   max_abs_err,
                   static_cast<long long>(max_abs_i),
                   static_cast<long long>(max_abs_j));
            printf("Rel at max abs: %.9e\n", max_rel_err_at_max_abs);
            printf("Max rel err   : %.9e at C[%lld,%lld]\n",
                   max_rel_err,
                   static_cast<long long>(max_rel_i),
                   static_cast<long long>(max_rel_j));
            printf("Mean abs err  : %.9e\n", sum_abs_err / static_cast<double>(total));

            float got00 = out_col_major[0];
            float got10 = rows > 1 ? out_col_major[1] : 0.0f;
            float got01 = cols > 1 ? out_col_major[rows] : 0.0f;

            float ref00 = ref_row_major[0];
            float ref10 = rows > 1 ? ref_row_major[cols] : 0.0f;
            float ref01 = cols > 1 ? ref_row_major[1] : 0.0f;

            printf("\nLayout probe:\n");
            printf("  got C[0,0] = out[0] = %.6f, ref[0,0] = ref[0] = %.6f\n", got00, ref00);
            printf("  got C[1,0] = out[1] = %.6f, ref[1,0] = ref[N] = %.6f\n", got10, ref10);
            printf("  got C[0,1] = out[M] = %.6f, ref[0,1] = ref[1] = %.6f\n", got01, ref01);

            if (mismatch_count == 0)
            {
                printf("✅ Layout-aware verification PASSED.\n");
            }
            else
            {
                printf("❌ Layout-aware verification FAILED.\n");
            }

            printf("\n");
            return mismatch_count == 0;
        };

    if (needVerify)
    {
        ScopedTimer timer(timing.verification);
        verification_passed = verify_column_major_output_against_row_major_ref(
            h_C_out,
            h_C_ref,
            M,
            N_original,
            verifyLevel >= 2 ? 64 : 16);
    }

    if (debug_enable && debug_out_host != nullptr && !needDebugDump)
    {
        ScopedTimer timer(timing.debug_summary_printing);
        const int DBG_B_TILE_TOTAL = 324;
        const int DBG_B_TILE_CONTIG = 325;
        const int DBG_B_TILE_NEAR = 326;
        const int DBG_B_TILE_MAX_STEP = 327;
        const int DBG_B_TILE_SUM_STEP = 328;
        const int DBG_B_TILE_VALID = 329;

        int64_t all_b_tile_total = 0;
        int64_t all_b_tile_valid = 0;
        int64_t all_b_tile_contig = 0;
        int64_t all_b_tile_near = 0;
        int64_t all_b_tile_sum_step = 0;
        int32_t all_b_tile_max_step = 0;

        for (int64_t core = 0; core < groupDim; ++core)
        {
            int32_t *dbg = debug_out_host + core * DEBUG_SLOTS;

            all_b_tile_total += dbg[DBG_B_TILE_TOTAL];
            all_b_tile_valid += dbg[DBG_B_TILE_VALID];
            all_b_tile_contig += dbg[DBG_B_TILE_CONTIG];
            all_b_tile_near += dbg[DBG_B_TILE_NEAR];
            all_b_tile_sum_step += dbg[DBG_B_TILE_SUM_STEP];
            if (dbg[DBG_B_TILE_MAX_STEP] > all_b_tile_max_step)
            {
                all_b_tile_max_step = dbg[DBG_B_TILE_MAX_STEP];
            }
        }

        if (all_b_tile_total > 0)
        {
            double total = static_cast<double>(all_b_tile_total);
            printf("\n===== B Tile Index Locality Summary =====\n");
            printf("  total_tiles      : %lld\n", static_cast<long long>(all_b_tile_total));
            printf("  valid_tiles      : %lld (%.2f%%)\n",
                   static_cast<long long>(all_b_tile_valid),
                   100.0 * static_cast<double>(all_b_tile_valid) / total);
            printf("  contiguous_tiles : %lld (%.2f%%)\n",
                   static_cast<long long>(all_b_tile_contig),
                   100.0 * static_cast<double>(all_b_tile_contig) / total);
            printf("  near_tiles       : %lld (%.2f%%, step<=4)\n",
                   static_cast<long long>(all_b_tile_near),
                   100.0 * static_cast<double>(all_b_tile_near) / total);
            printf("  max_abs_step     : %d\n", all_b_tile_max_step);
            printf("  avg_step_sum/tile: %.2f\n",
                   static_cast<double>(all_b_tile_sum_step) / total);
        }
    }

    if (needDebugDump)
    {
    ScopedTimer timer(timing.debug_summary_printing);

    auto print_fp16_dump = [](const char *name, __fp16 *p, int count) {
        printf("\n--- %s, first %d fp16 ---\n", name, count);
        for (int i = 0; i < count; ++i)
        {
            printf("%.4f ", static_cast<float>(p[i]));
            if ((i + 1) % 16 == 0)
                printf("\n");
        }
    };

    const int DBG_CORE_ID = 0;
    const int DBG_CORE_CNT = 1;
    const int DBG_M = 2;
    const int DBG_N = 3;
    const int DBG_K = 4;
    const int DBG_NUM_ROW_BLOCKS = 5;
    const int DBG_ROW_LOOP = 6;
    const int DBG_START_VEC = 7;
    const int DBG_END_VEC = 8;
    const int DBG_NUM_VECTORS = 9;
    const int DBG_MAX_VEC_PER_ROW = 10;
    const int DBG_N_BLOCK = 11;
    const int DBG_N_ACTUAL = 12;
    const int DBG_FIRST_COL = 13;
    const int DBG_SECOND_COL = 14;
    const int DBG_STATUS = 15;
    const int DBG_A_RAW_0 = 16;
    const int DBG_A_RAW_1 = 17;
    const int DBG_B_RAW_0 = 18;
    const int DBG_B_RAW_1 = 19;
    const int DBG_BC_RAW_0 = 20;
    const int DBG_BC_RAW_1 = 21;
    const int DBG_MAD_M = 22;
    const int DBG_MAD_K = 23;
    const int DBG_MAD_N = 24;
    const int DBG_C_OFFSET = 25;
    const int DBG_K_FRACS = 26;
    const int DBG_L1_LOAD_REPEAT = 27;
    const int DBG_C_RAW_0 = 28;
    const int DBG_ERR = 29;
    const int DBG_PHASE = 30;
    const int DBG_RESERVED = 31;

    const int DBG_ROW_OFF_START = 32;
    const int DBG_COL_START = 48;

    const int DBG_A_DUMP_START = 64;
    const int DBG_BC_DUMP_START = 128;
    const int DBG_L1A_DUMP_START = 192;
    const int DBG_L1B_DUMP_START = 256;

    const int DBG_REF_C00 = 320;
    const int DBG_REF_C10 = 321;
    const int DBG_REF_C01 = 322;
    const int DBG_REF_C11 = 323;
    const int DBG_B_TILE_TOTAL = 324;
    const int DBG_B_TILE_CONTIG = 325;
    const int DBG_B_TILE_NEAR = 326;
    const int DBG_B_TILE_MAX_STEP = 327;
    const int DBG_B_TILE_SUM_STEP = 328;
    const int DBG_B_TILE_VALID = 329;

    const int ST_ENTER = 0x00000001;
    const int ST_ROW_OK = 0x00000002;
    const int ST_A_GM_OK = 0x00000004;
    const int ST_A_L1_OK = 0x00000008;
    const int ST_B_GATHER_OK = 0x00000010;
    const int ST_B_L1_OK = 0x00000020;
    const int ST_L0_OK = 0x00000040;
    const int ST_MAD_OK = 0x00000080;
    const int ST_C_GM_OK = 0x00000100;
    const int ST_DONE = 0x00000200;
    const int ST_ALL_OK = ST_ENTER | ST_ROW_OK | ST_A_GM_OK | ST_A_L1_OK |
                          ST_B_GATHER_OK | ST_B_L1_OK | ST_L0_OK |
                          ST_MAD_OK | ST_C_GM_OK | ST_DONE;

    int32_t *dbg0 = debug_out_host;

    printf("\n===== Core 0 Detailed Debug =====\n");
    printf("  core_id          : %d\n", dbg0[DBG_CORE_ID]);
    printf("  core_cnt         : %d\n", dbg0[DBG_CORE_CNT]);
    printf("  M / N / K        : %d / %d / %d\n", dbg0[DBG_M], dbg0[DBG_N], dbg0[DBG_K]);
    printf("  num_row_blocks   : %d\n", dbg0[DBG_NUM_ROW_BLOCKS]);
    printf("  row_loop         : %d\n", dbg0[DBG_ROW_LOOP]);
    printf("  start_vec        : %d\n", dbg0[DBG_START_VEC]);
    printf("  end_vec          : %d\n", dbg0[DBG_END_VEC]);
    printf("  num_vectors      : %d\n", dbg0[DBG_NUM_VECTORS]);
    printf("  max_vec_per_row  : %d\n", dbg0[DBG_MAX_VEC_PER_ROW]);
    printf("  n_block          : %d\n", dbg0[DBG_N_BLOCK]);
    printf("  n_actual         : %d\n", dbg0[DBG_N_ACTUAL]);
    printf("  first_col        : %d\n", dbg0[DBG_FIRST_COL]);
    printf("  second_col       : %d\n", dbg0[DBG_SECOND_COL]);
    printf("  status           : 0x%X\n", dbg0[DBG_STATUS]);

    if ((dbg0[DBG_STATUS] & ST_ALL_OK) != ST_ALL_OK)
    {
        printf("  status_warning   : core0 did NOT finish all expected stages. missing_mask=0x%X\n",
               ST_ALL_OK & (~dbg0[DBG_STATUS]));
    }

    printf("  A_raw[0]         : %d, %.6f\n", dbg0[DBG_A_RAW_0], dbg0[DBG_A_RAW_0] / 1000.0f);
    printf("  A_raw[1]         : %d, %.6f\n", dbg0[DBG_A_RAW_1], dbg0[DBG_A_RAW_1] / 1000.0f);
    printf("  B_raw[0]         : %d, %.6f\n", dbg0[DBG_B_RAW_0], dbg0[DBG_B_RAW_0] / 1000.0f);
    printf("  B_raw[1]         : %d, %.6f\n", dbg0[DBG_B_RAW_1], dbg0[DBG_B_RAW_1] / 1000.0f);
    printf("  B_gather[0]      : %d, %.6f\n", dbg0[DBG_BC_RAW_0], dbg0[DBG_BC_RAW_0] / 1000.0f);
    printf("  B_gather[1]      : %d, %.6f\n", dbg0[DBG_BC_RAW_1], dbg0[DBG_BC_RAW_1] / 1000.0f);
    printf("  mad_m / k / n    : %d / %d / %d\n", dbg0[DBG_MAD_M], dbg0[DBG_MAD_K], dbg0[DBG_MAD_N]);
    printf("  c_offset         : %d\n", dbg0[DBG_C_OFFSET]);
    printf("  k_fracs          : %d\n", dbg0[DBG_K_FRACS]);
    printf("  l1_load_repeat   : %d\n", dbg0[DBG_L1_LOAD_REPEAT]);
    printf("  sparse_k_tile    : %d\n", dbg0[DBG_RESERVED]);
    if (dbg0[DBG_B_TILE_TOTAL] > 0)
    {
        double total = static_cast<double>(dbg0[DBG_B_TILE_TOTAL]);
        double valid_rate = 100.0 * static_cast<double>(dbg0[DBG_B_TILE_VALID]) / total;
        double contig_rate = 100.0 * static_cast<double>(dbg0[DBG_B_TILE_CONTIG]) / total;
        double near_rate = 100.0 * static_cast<double>(dbg0[DBG_B_TILE_NEAR]) / total;
        double avg_step_sum = static_cast<double>(dbg0[DBG_B_TILE_SUM_STEP]) / total;

        printf("  B tile total     : %d\n", dbg0[DBG_B_TILE_TOTAL]);
        printf("  B tile valid     : %d (%.2f%%)\n", dbg0[DBG_B_TILE_VALID], valid_rate);
        printf("  B tile contig    : %d (%.2f%%)\n", dbg0[DBG_B_TILE_CONTIG], contig_rate);
        printf("  B tile near      : %d (%.2f%%, step<=4)\n", dbg0[DBG_B_TILE_NEAR], near_rate);
        printf("  B tile max_step  : %d\n", dbg0[DBG_B_TILE_MAX_STEP]);
        printf("  B tile avg_step_sum_per_tile: %.2f\n", avg_step_sum);
    }

    {
        int64_t c_offset0 = static_cast<int64_t>(dbg0[DBG_C_OFFSET]);
        float c_raw_host = 0.0f;

        if (c_offset0 >= 0 && c_offset0 < static_cast<int64_t>(C_count_kernel))
        {
            c_raw_host = h_C_out[c_offset0];
        }

        printf("  C_raw[0]         : %.6f  // from host h_C_out[c_offset=%lld]\n",
               c_raw_host,
               static_cast<long long>(c_offset0));

        printf("  C_raw_dbg_slot   : %d, %.6f  // kernel-side best-effort GM readback\n",
               dbg0[DBG_C_RAW_0],
               dbg0[DBG_C_RAW_0] / 1000.0f);
    }

    if (M > 1 && N > 1)
    {
        printf("  layout_probe     : out_col_major h_C_out[0]=%.6f, h_C_out[1]=%.6f, h_C_out[M]=%.6f\n",
               h_C_out[0],
               h_C_out[1],
               h_C_out[M]);

        printf("                     ref_row_major h_C_ref[0]=%.6f, h_C_ref[N]=%.6f, h_C_ref[1]=%.6f\n",
               h_C_ref[0],
               h_C_ref[N],
               h_C_ref[1]);

        printf("                     debug_ref     ref_C[0,0]=%.6f, ref_C[1,0]=%.6f, ref_C[0,1]=%.6f\n",
               dbg0[DBG_REF_C00] / 1000.0f,
               dbg0[DBG_REF_C10] / 1000.0f,
               dbg0[DBG_REF_C01] / 1000.0f);
    }

    printf("  ref_C[0,0]       : %d, %.6f\n", dbg0[DBG_REF_C00], dbg0[DBG_REF_C00] / 1000.0f);
    printf("  ref_C[1,0]       : %d, %.6f\n", dbg0[DBG_REF_C10], dbg0[DBG_REF_C10] / 1000.0f);
    printf("  ref_C[0,1]       : %d, %.6f\n", dbg0[DBG_REF_C01], dbg0[DBG_REF_C01] / 1000.0f);
    printf("  ref_C[1,1]       : %d, %.6f\n", dbg0[DBG_REF_C11], dbg0[DBG_REF_C11] / 1000.0f);
    printf("  err              : 0x%X\n", dbg0[DBG_ERR]);
    printf("  phase            : %d\n", dbg0[DBG_PHASE]);

    printf("\nCore 0 row_offsets dump from kernel debug_out:\n");
    for (int i = 0; i < 16; ++i)
    {
        printf("%d ", dbg0[DBG_ROW_OFF_START + i]);
        if ((i + 1) % 16 == 0)
            printf("\n");
    }

    printf("\nCore 0 col_indices dump from kernel debug_out:\n");
    for (int i = 0; i < 16; ++i)
    {
        printf("%d ", dbg0[DBG_COL_START + i]);
        if ((i + 1) % 16 == 0)
            printf("\n");
    }

    print_fp16_dump("A_src GM raw dump",
                    reinterpret_cast<__fp16 *>(dbg0 + DBG_A_DUMP_START),
                    128);

    print_fp16_dump("B_workspace GM memory dump: column-major KxN / row-major NxK",
                    reinterpret_cast<__fp16 *>(dbg0 + DBG_BC_DUMP_START),
                    128);

    print_fp16_dump("B_selected logical Kx16 dump",
                    reinterpret_cast<__fp16 *>(dbg0 + DBG_L1A_DUMP_START),
                    128);

    print_fp16_dump("A_selected logical Kx16 dump",
                    reinterpret_cast<__fp16 *>(dbg0 + DBG_L1B_DUMP_START),
                    128);

    printf("\n===== Other Cores Debug Summary =====\n");
    int64_t all_b_tile_total = dbg0[DBG_B_TILE_TOTAL];
    int64_t all_b_tile_valid = dbg0[DBG_B_TILE_VALID];
    int64_t all_b_tile_contig = dbg0[DBG_B_TILE_CONTIG];
    int64_t all_b_tile_near = dbg0[DBG_B_TILE_NEAR];
    int64_t all_b_tile_sum_step = dbg0[DBG_B_TILE_SUM_STEP];
    int32_t all_b_tile_max_step = dbg0[DBG_B_TILE_MAX_STEP];

    for (int64_t core = 1; core < groupDim; ++core)
    {
        int32_t *dbg = debug_out_host + core * DEBUG_SLOTS;

        if (dbg[DBG_CORE_ID] == static_cast<int32_t>(core) || dbg[DBG_STATUS] != 0)
        {
            bool done = ((dbg[DBG_STATUS] & ST_ALL_OK) == ST_ALL_OK);

            printf("Core %lld: row=%d, start=%d, end=%d, vec=%d, n_block=%d, phase=%d, err=0x%X, status=0x%X%s\n",
                   static_cast<long long>(core),
                   dbg[DBG_ROW_LOOP],
                   dbg[DBG_START_VEC],
                   dbg[DBG_END_VEC],
                   dbg[DBG_NUM_VECTORS],
                   dbg[DBG_N_BLOCK],
                   dbg[DBG_PHASE],
                   dbg[DBG_ERR],
                   dbg[DBG_STATUS],
                   done ? "" : "  <-- NOT_DONE");

            all_b_tile_total += dbg[DBG_B_TILE_TOTAL];
            all_b_tile_valid += dbg[DBG_B_TILE_VALID];
            all_b_tile_contig += dbg[DBG_B_TILE_CONTIG];
            all_b_tile_near += dbg[DBG_B_TILE_NEAR];
            all_b_tile_sum_step += dbg[DBG_B_TILE_SUM_STEP];
            if (dbg[DBG_B_TILE_MAX_STEP] > all_b_tile_max_step)
            {
                all_b_tile_max_step = dbg[DBG_B_TILE_MAX_STEP];
            }
        }
    }

    if (all_b_tile_total > 0)
    {
        double total = static_cast<double>(all_b_tile_total);
        printf("\n===== B Tile Index Locality Summary =====\n");
        printf("  total_tiles      : %lld\n", static_cast<long long>(all_b_tile_total));
        printf("  valid_tiles      : %lld (%.2f%%)\n",
               static_cast<long long>(all_b_tile_valid),
               100.0 * static_cast<double>(all_b_tile_valid) / total);
        printf("  contiguous_tiles : %lld (%.2f%%)\n",
               static_cast<long long>(all_b_tile_contig),
               100.0 * static_cast<double>(all_b_tile_contig) / total);
        printf("  near_tiles       : %lld (%.2f%%, step<=4)\n",
               static_cast<long long>(all_b_tile_near),
               100.0 * static_cast<double>(all_b_tile_near) / total);
        printf("  max_abs_step     : %d\n", all_b_tile_max_step);
        printf("  avg_step_sum/tile: %.2f\n",
               static_cast<double>(all_b_tile_sum_step) / total);
    }

    }

    {
        ScopedTimer timer(timing.cleanup);
        CALL_RT(aclrtFree(d_values));
        CALL_RT(aclrtFree(d_col_indices));
        CALL_RT(aclrtFree(d_row_offsets));
        CALL_RT(aclrtFree(d_b_pack_offsets));
        CALL_RT(aclrtFree(d_b_pack_workspace));
        CALL_RT(aclrtFree(d_B));
        CALL_RT(aclrtFree(d_C));
        if (debug_out_dev != nullptr)
        {
            CALL_RT(aclrtFree(debug_out_dev));
        }

        CALL_RT(aclrtFreeHost(h_B));
        if (h_C_ref != nullptr)
            CALL_RT(aclrtFreeHost(h_C_ref));
        if (h_C_out != nullptr)
            CALL_RT(aclrtFreeHost(h_C_out));
        if (debug_out_host != nullptr)
            CALL_RT(aclrtFreeHost(debug_out_host));

        CALL_RT(aclrtResetDevice(deviceId));
        CALL_RT(aclFinalize());
    }

    double total_inside_main_ms = elapsed_ms(main_start, HostClock::now());
    print_host_wall_time_breakdown(timing, launch_count, total_inside_main_ms);

    std::cout << "Kernel end!" << std::endl;
    return verification_passed ? 0 : -1;
}

