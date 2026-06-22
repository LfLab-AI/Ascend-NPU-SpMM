#ifndef ASCBLAS_FP16_UTILS_H
#define ASCBLAS_FP16_UTILS_H

namespace fp16 {
    constexpr int64_t L0AB_PINGPONG_BUFFER_LEN = 32 * 1024 / sizeof(__fp16);   // 32KB
    constexpr int64_t L0C_PINGPONG_BUFFER_LEN = 64 * 1024 / sizeof(float);    // 64KB
    constexpr int64_t NUM_ELE_PERBLOCK = 32 / sizeof(__fp16); // 每个block能放多少个__fp16
    // CUBE的最小基块 （M，N，K）= 16 * 16 * 16
    constexpr int64_t CUBE_M0 = 16;
    constexpr int64_t CUBE_N0 = 16;
    constexpr int64_t CUBE_K0 = 32 / sizeof(__fp16);
    constexpr int64_t CUBE_MATRIX_SIZE = CUBE_K0 * CUBE_N0;                  // 16 * 16
    constexpr int64_t L1_PINGPONG_BUFFER_LEN = 64 * 1024 / sizeof(__fp16);    // 256KB
    constexpr int64_t UINT16_STRIDE_LIMIT = 65536;

    /**
     * @brief AIC函数：将列优先的nD矩阵转换为nZ格式
     * @param [in] __cbuf__ __fp16 *dst：L1 目的地址
     * @param [in] __gm__ __fp16 *src: GM 源地址
     * @param [in] int64_t CBUF_M0：在L1中矩阵的行数（至少要32B对齐）
     * @param [in] int64_t CBUF_N0：在L1中矩阵的列数（至少要32B对齐）
     * @param [in] int64_t m_actual：在GM中矩阵的行数（不需要对齐）
     * @param [in] int64_t n_actual：在GM中矩阵的列数（不需要对齐）
     * @param [in] int64_t stride：在GM中矩阵中两列之间的距离（不需要对齐）
     */
    __aicore__ __inline__ void ascblas_matrix_gm2cbuf_ND2nZ(
        __cbuf__ __fp16 *dst, 
        __gm__ __fp16 * src,
        int64_t CBUF_M0,
        int64_t CBUF_N0, 
        int64_t m_actual, 
        int64_t n_actual, 
        size_t stride
    )
    {
        if(stride < UINT16_STRIDE_LIMIT) {
            copy_gm_to_cbuf_multi_nd2nz_b16(
                dst,
                src,
                static_cast<uint8_t>(0),
                static_cast<uint16_t>(1),
                static_cast<uint16_t>(n_actual),
                static_cast<uint16_t>(m_actual),
                static_cast<uint16_t>(0),
                static_cast<uint16_t>(stride),
                static_cast<uint16_t>(CBUF_N0),
                static_cast<uint16_t>(1),
                static_cast<uint16_t>(0)
            );
        } else {
            for(int i = 0; i < n_actual; i++) {
                copy_gm_to_cbuf_multi_nd2nz_b16(
                    dst + i * CUBE_K0,
                    src + i * stride,
                    static_cast<uint8_t>(0),
                    static_cast<uint16_t>(1),
                    static_cast<uint16_t>(1),
                    static_cast<uint16_t>(m_actual),
                    static_cast<uint16_t>(0),
                    static_cast<uint16_t>(0),
                    static_cast<uint16_t>(CBUF_N0),
                    static_cast<uint16_t>(0),
                    static_cast<uint16_t>(0)
                );    
            }
        }
    }

    /**
     * @brief AIC函数：将列优先的nD矩阵转换为nN格式
     * @param [in] __cbuf__ __fp16 *dst：L1 目的地址
     * @param [in] __gm__ __fp16 *src: GM 源地址
     * @param [in] int64_t CBUF_M0：在L1中矩阵的行数（至少要32B对齐）
     * @param [in] int64_t CBUF_N0：在L1中矩阵的列数（至少要32B对齐）
     * @param [in] int64_t m_actual：在GM中矩阵的行数（不需要对齐）
     * @param [in] int64_t n_actual：在GM中矩阵的列数（不需要对齐）
     * @param [in] int64_t stride：在GM中矩阵两列之间的距离（不需要对齐）
     */
    __aicore__ __inline__ void ascblas_matrix_gm2cbuf_ND2nN(
        __cbuf__ __fp16 *dst, 
        __gm__ __fp16 * src,
        int64_t CBUF_M0, 
        int64_t CBUF_N0, 
        int64_t m_actual, 
        int64_t n_actual, 
        size_t stride
    )
    {
        int64_t srcNdStride = CUBE_N0 * stride;
        int64_t srcNStride = stride;
        if(srcNdStride < UINT16_STRIDE_LIMIT) {
            int ndNum = n_actual / CUBE_N0;
            int remains = n_actual % CUBE_N0;
            if(ndNum > 0) {
                copy_gm_to_cbuf_multi_nd2nz_b16(
                    dst,
                    src,
                    static_cast<uint8_t>(0),            // sid
                    static_cast<uint16_t>(ndNum),       // ndNum
                    static_cast<uint16_t>(CUBE_N0),          // nValue
                    static_cast<uint16_t>(m_actual),   // dValue
                    static_cast<uint16_t>(srcNdStride), // srcNdMatrixStride
                    static_cast<uint16_t>(srcNStride),  // srcDValue
                    static_cast<uint16_t>(CUBE_N0),          // dstNzC0Stride
                    static_cast<uint16_t>(1),           // dstNzNStride
                    static_cast<uint16_t>(CUBE_N0 * CBUF_M0)      // dstNzMatrixStride
                );
            }
            if(remains > 0){
                copy_gm_to_cbuf_multi_nd2nz_b16(
                    dst + ndNum * CUBE_N0 * CBUF_M0,
                    src + ndNum * CUBE_N0 * stride,
                    static_cast<uint8_t>(0),            // sid
                    static_cast<uint16_t>(1),           // ndNum
                    static_cast<uint16_t>(remains),     // nValue
                    static_cast<uint16_t>(m_actual),   // dValue
                    static_cast<uint16_t>(0),           // srcNdMatrixStride
                    static_cast<uint16_t>(srcNStride),  // srcDValue
                    static_cast<uint16_t>(CUBE_N0),          // dstNzC0Stride
                    static_cast<uint16_t>(1),           // dstNzNStride
                    static_cast<uint16_t>(0)           // dstNzMatrixStride
                );
            }
        } else if (srcNStride < UINT16_STRIDE_LIMIT) {
            int ndNum = n_actual / CUBE_N0;
            int remains = n_actual % CUBE_N0;
            for (int i = 0; i < ndNum; i++) {
                copy_gm_to_cbuf_multi_nd2nz_b16(
                    dst + i * CUBE_N0 * CBUF_M0,
                    src + i * CUBE_N0 * stride,
                    static_cast<uint8_t>(0),            // sid
                    static_cast<uint16_t>(1),           // ndNum
                    static_cast<uint16_t>(CUBE_N0),          // nValue
                    static_cast<uint16_t>(m_actual),   // dValue
                    static_cast<uint16_t>(0),           // srcNdMatrixStride
                    static_cast<uint16_t>(srcNStride),  // srcDValue
                    static_cast<uint16_t>(CUBE_N0),          // dstNzC0Stride
                    static_cast<uint16_t>(1),           // dstNzNStride
                    static_cast<uint16_t>(0)           // dstNzMatrixStride
                );
            }
            if(remains > 0) {
                copy_gm_to_cbuf_multi_nd2nz_b16(
                    dst + ndNum * CUBE_N0 * CBUF_M0,
                    src + ndNum * CUBE_N0 * stride,
                    static_cast<uint8_t>(0),            // sid
                    static_cast<uint16_t>(1),           // ndNum
                    static_cast<uint16_t>(remains),     // nValue
                    static_cast<uint16_t>(m_actual),   // dValue
                    static_cast<uint16_t>(0),           // srcNdMatrixStride
                    static_cast<uint16_t>(srcNStride),  // srcDValue
                    static_cast<uint16_t>(CUBE_N0),          // dstNzC0Stride
                    static_cast<uint16_t>(1),           // dstNzNStride
                    static_cast<uint16_t>(0)           // dstNzMatrixStride
                );
            }
        } else {
            for(int i = 0; i < n_actual; i++) {
                int idxR0 = i / CUBE_N0;
                int idxInR0 = i % CUBE_N0;
                copy_gm_to_cbuf_multi_nd2nz_b16(
                    dst + idxR0 * CUBE_N0 * CBUF_M0 + idxInR0 * CUBE_K0,
                    src + i * stride,
                    static_cast<uint8_t>(0),            // sid
                    static_cast<uint16_t>(1),           // ndNum
                    static_cast<uint16_t>(1),           // nValue
                    static_cast<uint16_t>(m_actual),   // dValue
                    static_cast<uint16_t>(0),           // srcNdMatrixStride, unused
                    static_cast<uint16_t>(0),           // srcDValue, unused
                    static_cast<uint16_t>(CUBE_N0),          // dstNzC0Stride
                    static_cast<uint16_t>(0),           // dstNzNStride, unused
                    static_cast<uint16_t>(0)           // dstNzMatrixStride, unused
                );
            }
        }
    }

    /**
     * @brief AIV函数：将列优先的nD矩阵GM读取到UB
     * @param [in] __cbuf__ __fp16 *dst：UB 目的地址
     * @param [in] __gm__ __fp16 *src: GM 源地址
     * @param [in] int64_t m_actual：在GM中矩阵的行数（不需要对齐）
     * @param [in] int64_t n_actual：在GM中矩阵的列数（不需要对齐）
     * @param [in] int64_t srcStride：在GM中矩阵两列之间的距离（不需要对齐）
     * @param [in] int64_t dstStride：在UB中矩阵两列之间的距离（需要32B对齐）
     */
    __aicore__ __inline__ void ascblas_matrix_gm2ubuf(
        __ubuf__ __fp16 * dst, 
        __gm__ __fp16 *src,
        int64_t m_actual,
        int64_t n_actual,
        size_t srcStride,
        size_t dstStride
    )
    {
        int64_t m_round = ROUND(m_actual, NUM_ELE_PERBLOCK);
        if (m_actual % NUM_ELE_PERBLOCK == 0 && srcStride % NUM_ELE_PERBLOCK == 0 && srcStride < UINT16_STRIDE_LIMIT) {
            copy_gm_to_ubuf(
                dst,
                src,
                0,
                n_actual,
                m_round / NUM_ELE_PERBLOCK,
                (srcStride - m_round) / NUM_ELE_PERBLOCK,
                (dstStride - m_round) / NUM_ELE_PERBLOCK
            );
        } else if (m_actual % NUM_ELE_PERBLOCK == 0 && srcStride * NUM_ELE_PERBLOCK < UINT16_STRIDE_LIMIT) {
            int C0_SIZE_loop = n_actual / NUM_ELE_PERBLOCK;
            int C0_SIZE_remain = n_actual % NUM_ELE_PERBLOCK;
            if (C0_SIZE_loop > 0) {
                for (int i = 0; i < NUM_ELE_PERBLOCK; i++) {
                    copy_gm_to_ubuf(
                        dst + i * dstStride,
                        src + i * srcStride,
                        0,
                        C0_SIZE_loop,
                        m_round / NUM_ELE_PERBLOCK,
                        (srcStride * NUM_ELE_PERBLOCK - m_round) / NUM_ELE_PERBLOCK,
                        (dstStride * NUM_ELE_PERBLOCK - m_round) / NUM_ELE_PERBLOCK
                    );
                }
            }
            for (int i = 0; i < C0_SIZE_remain; i++) {
                copy_gm_to_ubuf(
                    dst + C0_SIZE_loop * NUM_ELE_PERBLOCK * dstStride + i * dstStride,
                    src + C0_SIZE_loop * NUM_ELE_PERBLOCK * srcStride + i * srcStride,
                    0,
                    1,
                    m_round / NUM_ELE_PERBLOCK,
                    0,
                    0
                );
            }
        } else {
            for (int i = 0; i < n_actual; i++) {
                copy_gm_to_ubuf_align_b16(
                    dst + i * dstStride,
                    src + i * srcStride,
                    0,
                    1,
                    m_actual * sizeof(__fp16),
                    0,
                    0,
                    0,
                    0
                );
            }
        }
    }

    /**
     * @brief AIV函数：将列优先的nD矩阵从UB读取到GM
     * @param [in] __cbuf__ __fp16 *dst：GM 目的地址
     * @param [in] __gm__ __fp16 *src: UB 源地址
     * @param [in] int64_t m_actual：在UB中矩阵的行数（不需要对齐）
     * @param [in] int64_t n_actual：在UB中矩阵的列数（不需要对齐）
     * @param [in] int64_t srcStride：在UB中矩阵两列之间的距离（需要32B对齐）
     * @param [in] int64_t dstStride：在GM中矩阵两列之间的距离（不需要对齐）
     */
    __aicore__ __inline__ void ascblas_matrix_ubuf2gm(
        __gm__ __fp16 * dst, 
        __ubuf__ __fp16 *src,
        int64_t m_actual,
        int64_t n_actual,
        size_t srcStride,
        size_t dstStride
    ) 
    {
        int64_t m_round = ROUND(m_actual, NUM_ELE_PERBLOCK);
        if (m_actual % NUM_ELE_PERBLOCK == 0 && dstStride % NUM_ELE_PERBLOCK == 0 && dstStride < UINT16_STRIDE_LIMIT) {
            copy_ubuf_to_gm(
                dst,
                src,
                0,
                n_actual,
                m_round / NUM_ELE_PERBLOCK,
                (srcStride - m_round) / NUM_ELE_PERBLOCK,
                (dstStride - m_round) / NUM_ELE_PERBLOCK
            );
        } else if (m_actual % NUM_ELE_PERBLOCK == 0 && dstStride * NUM_ELE_PERBLOCK < UINT16_STRIDE_LIMIT) {
            int C0_SIZE_loop = n_actual / NUM_ELE_PERBLOCK;
            int C0_SIZE_remain = n_actual % NUM_ELE_PERBLOCK;
            if (C0_SIZE_loop > 0) {
                for (int i = 0; i < NUM_ELE_PERBLOCK; i++) {
                    copy_ubuf_to_gm(
                        dst + i * dstStride,
                        src + i * srcStride,
                        0,
                        C0_SIZE_loop,
                        m_actual / NUM_ELE_PERBLOCK,
                        (srcStride * NUM_ELE_PERBLOCK - m_actual) / NUM_ELE_PERBLOCK,
                        (dstStride * NUM_ELE_PERBLOCK - m_actual) / NUM_ELE_PERBLOCK
                    );
                }
            }
            for (int i = 0; i < C0_SIZE_remain; i++) {
                copy_ubuf_to_gm(
                    dst + C0_SIZE_loop * NUM_ELE_PERBLOCK * dstStride + i * dstStride,
                    src + C0_SIZE_loop * NUM_ELE_PERBLOCK * srcStride + i * srcStride,
                    0,
                    1,
                    m_actual / NUM_ELE_PERBLOCK,
                    0,
                    0
                );
            }
        } else {
            for (int i = 0; i < n_actual; i++) {
                copy_ubuf_to_gm_align_b16(
                    dst + i * dstStride,
                    src + i * srcStride,
                    0,
                    1,
                    m_actual * sizeof(__fp16),
                    0,
                    0,
                    0,
                    0
                );
            }
        }
    }
}
#endif