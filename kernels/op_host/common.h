/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
 */
#ifndef COMMON_H
#define COMMON_H

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"

inline std::map<ge::DataType, uint64_t> kDataSizeMap = {
    {ge::DT_FLOAT, sizeof(float)},
    {ge::DT_INT32, sizeof(int32_t)},
    {ge::DT_INT64, sizeof(int64_t)}
};

template<typename T>
inline T ceil_multiple(T num, T block)
{
    if (block == 0) {
        return 0;
    }
    return (num + block - 1) / block;
}

template<typename T>
inline T ceil_value(T num, T block)
{
    if (block == 0) {
        return 0;
    }
    return ((num + block - 1) / block) * block;
}

#endif // COMMON_H