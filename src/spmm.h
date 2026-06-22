#include <fstream>
#include <string>
#include <limits>
#include "ascblas.h"

struct SpmmTilingConfig {
    int64_t max_vectors_limit;
    int32_t aic_n_batch;
    int32_t sparse_k_tile;
    const char *kernel_name;
    const char *bin_name;
};

static inline SpmmTilingConfig SelectSpmmTilingConfig(int64_t max_vectors_per_row)
{
    // Keep thresholds in sync with tools/select_spmm_tiling.py.
    if (max_vectors_per_row <= 256) {
        return {256, 8, 256, "spmm_kernel_aic8_ktile256", "spmm_kernel_aic8_ktile256.o"};
    }
    if (max_vectors_per_row <= 512) {
        return {512, 4, 512, "spmm_kernel_aic4_ktile512", "spmm_kernel_aic4_ktile512.o"};
    }
    if (max_vectors_per_row <= 1024) {
        return {1024, 2, 1024, "spmm_kernel_aic2_ktile1024", "spmm_kernel_aic2_ktile1024.o"};
    }
    return {std::numeric_limits<int64_t>::max(), 1, 2048, "spmm_kernel_aic1_ktile2048", "spmm_kernel_aic1_ktile2048.o"};
}

aclError ascblasSpmm(
    ascblasHandle_t handle,
    int M,
    // N is the kernel-visible column count. The host pads it to
    // selected_aic_n_batch * 16 before launch.
    int N,
    int K,
    int vec_length,
    __fp16* values,
    int64_t max_vectors_per_row,
    int* col_indices,
    int* row_offsets,
    __fp16 *B,
    float *C,
    int64_t *b_pack_offsets,
    int64_t total_b_pack_elems,
    __fp16 *b_pack_workspace,
    int32_t *debug_out,
    int32_t debug_enable
)
{
    aclError error;
    aclrtStream stream;
    ascblasGetStream(handle, &stream);

    const SpmmTilingConfig tiling = SelectSpmmTilingConfig(max_vectors_per_row);
    std::string kernel_name = tiling.kernel_name;
    std::string bin_name = tiling.bin_name;

    // One process may exercise multiple shapes, so keep one registration flag
    // per dispatch variant instead of a single global flag.
    static bool registered_aic8_k256 = false;
    static bool registered_aic4_k512 = false;
    static bool registered_aic2_k1024 = false;
    static bool registered_aic1_k2048 = false;
    bool *registered = &registered_aic1_k2048;
    if (tiling.aic_n_batch == 8) {
        registered = &registered_aic8_k256;
    } else if (tiling.aic_n_batch == 4) {
        registered = &registered_aic4_k512;
    } else if (tiling.aic_n_batch == 2) {
        registered = &registered_aic2_k1024;
    }
    if (!(*registered)) {
        RegisterBinaryKernel(kernel_name.c_str(), bin_name.c_str());
        *registered = true;
    }

    void* ffts_addr = nullptr;
    uint32_t ffts_len = 0;
    error = rtGetC2cCtrlAddr((uint64_t*)(&ffts_addr), &ffts_len);

    int64_t num_row_blocks = (M + vec_length - 1) / vec_length;
    constexpr int64_t BLOCK_SIZE = 16;
    int64_t num_n_blocks   = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int64_t groupDim = num_row_blocks < CORENUM ? num_row_blocks : CORENUM;

    typedef struct {
        int M;
        int N;
        int K;
        int vec_length;
        __fp16 *values;
        int *col_indices;
        int *row_offsets;
        __fp16 *B;
        float *C;
        void *ffts_addr;
        __fp16 *workspace1;
        int64_t *b_pack_offsets;
        int64_t total_b_pack_elems;
        int64_t max_vectors_per_row;
        int64_t num_n_blocks;
        int64_t ldc;
        int32_t *debug_out;
        int32_t debug_enable;
    } KernelArgs;

    KernelArgs kernel_args;
    kernel_args.M = M;
    kernel_args.N = N;
    kernel_args.K = K;
    kernel_args.vec_length = vec_length;
    kernel_args.values = values;
    kernel_args.col_indices = col_indices;
    kernel_args.row_offsets = row_offsets;
    kernel_args.B = B;
    kernel_args.C = C;
    kernel_args.ffts_addr = ffts_addr;
    kernel_args.workspace1 = b_pack_workspace;
    kernel_args.b_pack_offsets = b_pack_offsets;
    kernel_args.total_b_pack_elems = total_b_pack_elems;
    kernel_args.max_vectors_per_row = max_vectors_per_row;
    kernel_args.num_n_blocks = num_n_blocks;
    kernel_args.ldc = M;
    kernel_args.debug_out = debug_out;
    kernel_args.debug_enable = debug_enable;

    error = rtKernelLaunch((void *)kernel_name.c_str(), groupDim, &kernel_args,
                           sizeof(kernel_args), NULL, stream);
    aclrtSynchronizeStream(stream);

    return error;
}

