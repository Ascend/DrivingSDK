/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#include "knn_small_n.h"
#include "knn_big_n.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void knn(
    GM_ADDR xyz,
    GM_ADDR center_xyz,
    GM_ADDR idx,
    GM_ADDR dist2,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    TPipe tmpPipe;
    knnTilingArgs tmpTiling;
    GET_TILING_DATA(tiling_data, tiling);
    tmpTiling.batch                 = tiling_data.batch;
    tmpTiling.npoint                = tiling_data.npoint;
    tmpTiling.nsample               = tiling_data.nsample;
    tmpTiling.nsample_aligned       = tiling_data.nsample_aligned;
    tmpTiling.nsource               = tiling_data.nsource;
    tmpTiling.nsource_aligned       = tiling_data.nsource_aligned;
    tmpTiling.nsource_aligned2      = tiling_data.nsource_aligned2;
    tmpTiling.nsource_aligned_size  = tiling_data.nsource_aligned_size;
    tmpTiling.nsource_aligned_size2 = tiling_data.nsource_aligned_size2;
    tmpTiling.is_from_knn           = tiling_data.is_from_knn;
    tmpTiling.inner                 = tiling_data.inner;
    tmpTiling.inner2                = tiling_data.inner2;
    tmpTiling.topkmax               = tiling_data.topkmax;
    tmpTiling.topkmax2              = tiling_data.topkmax2;
    tmpTiling.loop_times            = tiling_data.loop_times;
    tmpTiling.b_times_m             = tiling_data.b_times_m;
    tmpTiling.big_core_num          = tiling_data.big_core_num;
    tmpTiling.small_core_num        = tiling_data.small_core_num;
    tmpTiling.big_core_len          = tiling_data.big_core_len;
    tmpTiling.small_core_len        = tiling_data.small_core_len;
    tmpTiling.aligned_big_len       = tiling_data.aligned_big_len;
    tmpTiling.aligned_small_len     = tiling_data.aligned_small_len;

    tmpTiling.topkTilingData        = tiling_data.topkTilingData;
    tmpTiling.topkTilingData2       = tiling_data.topkTilingData2;
    tmpTiling.topkInfo.outter       = 1;
    tmpTiling.topkInfo.inner        = tiling_data.inner;
    tmpTiling.topkInfo2.outter      = 1;
    tmpTiling.topkInfo2.inner       = tiling_data.inner2;
    tmpTiling.topkInfo2.n           = tiling_data.nsample * 2;

    if (TILING_KEY_IS(100)) {
        KnnCase1<float, int32_t> op;
        op.target_x_num = 8;
        op.target_num = 24;
        op.Init(xyz, center_xyz, idx, dist2, &tmpTiling, &tmpPipe);
        op.Process();
    } else if (TILING_KEY_IS(101)) {
        KnnCase1<half, int32_t> op;
        op.target_x_num = 16;
        op.target_num = 48;
        op.Init(xyz, center_xyz, idx, dist2, &tmpTiling, &tmpPipe);
        op.Process();
    } else if (TILING_KEY_IS(102)) {
        KnnCase2<float, int32_t> op;
        op.target_x_num = 8;
        op.target_num = 24;
        op.Init(xyz, center_xyz, idx, dist2, &tmpTiling, &tmpPipe);
        op.Process();
    } else if (TILING_KEY_IS(103)) {
        KnnCase2<half, int32_t> op;
        op.target_x_num = 16;
        op.target_num = 48;
        op.Init(xyz, center_xyz, idx, dist2, &tmpTiling, &tmpPipe);
        op.Process();
    }
}