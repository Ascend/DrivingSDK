/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef COMMON_H
#define COMMON_H

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"

inline uint32_t ceil_multiple(uint32_t num, uint32_t block)
{
    if (block == 0) {
        return 0;
    }
    return (num + block - 1) / block;
}

inline uint32_t ceil_value(uint32_t num, uint32_t block)
{
    if (block == 0) {
        return 0;
    }
    return ((num + block - 1) / block) * block;
}

#endif // COMMON_H
