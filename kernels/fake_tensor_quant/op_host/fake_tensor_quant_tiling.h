/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#ifndef SIGMOID_FOCAL_WEIGHT_TILING_H
#define SIGMOID_FOCAL_WEIGHT_TILING_H
#include "register/tilingdata_base.h"

#define CHECK_ON_SUCCESS(status)        \
    if ((status) == ge::GRAPH_FAILED) { \
        return ge::GRAPH_FAILED;        \
    }

#define OP_LOGE(OP_NAME, LOG, ...)                                \
    do {                                                          \
        printf("[ERROR]  %s: " LOG "\n", OP_NAME, ##__VA_ARGS__); \
    } while (0)

#define OPS_LOG_E_IF(OP_NAME, COND, EXPR, LOG, ...)                                                         \
    static_assert(std::is_same<bool, std::decay<decltype(COND)>::type>::value, "condition should be bool"); \
    do {                                                                                                    \
        if (__builtin_expect((COND), 0)) {                                                                  \
            OP_LOGE(OP_NAME, LOG, ##__VA_ARGS__);                                                           \
            EXPR;                                                                                           \
        }                                                                                                   \
    } while (0)

#define OPS_LOG_E_IF_NULL(OP_NAME, PTR, EXPR)                  \
    if (__builtin_expect((PTR) == nullptr, 0)) {               \
        printf("[ERROR] %s: %s is nullptr!\n", OP_NAME, #PTR); \
        EXPR;                                                  \
    }

namespace optiling {
BEGIN_TILING_DATA_DEF(FakeTensorQuantTilingData)
TILING_DATA_FIELD_DEF(uint32_t, headCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, normalCopyElemNum);
TILING_DATA_FIELD_DEF(uint32_t, eachHeadCoreElemNum);
TILING_DATA_FIELD_DEF(uint32_t, eachTailCoreElemNum);
TILING_DATA_FIELD_DEF(uint32_t, headCoreCopyNum);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreCopyNum);
TILING_DATA_FIELD_DEF(uint32_t, headCoreLastCopyElemNum);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreLastCopyElemNum);
TILING_DATA_FIELD_DEF(int32_t, maxBound);
TILING_DATA_FIELD_DEF(int32_t, minBound);
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(FakeTensorQuant, FakeTensorQuantTilingData)
} // namespace optiling
#endif // SIGMOID_FOCAL_WEIGHT_TILING_H
