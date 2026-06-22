#ifndef ASCBLAS_KERNEL_UTILS_H
#define ASCBLAS_KERNEL_UTILS_H

#include "ascblas_type.h"

/**
 * @brief 用于构建AIC和AIV同步的config
 * @param [in] int64_t mode：同步模式
 * @param [in] int64_t flagId：区分不同的同步
 * @return int64_t config：ffts_cross_core_sync第二个参数
 */
__aicore__ inline int64_t GET_FFST_MSG(int64_t mode, int64_t flagId)
{
    return 1 | (mode << 4) | (flagId << 8);
}

/**
 * @brief num 向上按照 padding_num 的倍数取整
 * @param [in] int64_t num：需要取整的数
 * @param [in] int64_t padding_num：需要向上取整的最小粒度
 * @return int64_t 向上取整的值
 */
__aicore__ inline int64_t ROUND(int64_t num, int64_t padding_num)
{
    return ((num + padding_num - 1) / padding_num * padding_num);
}

#include "ascblas_fp32_utils.h"
#include "ascblas_fp16_utils.h"


#endif