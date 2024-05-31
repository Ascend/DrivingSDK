/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef KNN_TILING_H
#define KNN_TILING_H

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"

enum CalcDataSizeFuncReturn {
    knnError,
    knnSpaceEnough,
    knnSpaceOverflow
};

namespace optiling {
/****************TilingData definition*****************/
BEGIN_TILING_DATA_DEF(KnnTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, npoint);
    TILING_DATA_FIELD_DEF(uint32_t, nsample);
    TILING_DATA_FIELD_DEF(uint32_t, nsample_aligned);
    TILING_DATA_FIELD_DEF(uint32_t, nsource);
    TILING_DATA_FIELD_DEF(uint32_t, nsource_aligned);
    TILING_DATA_FIELD_DEF(uint32_t, nsource_aligned2);
    TILING_DATA_FIELD_DEF(uint32_t, nsource_aligned_size);
    TILING_DATA_FIELD_DEF(uint32_t, nsource_aligned_size2);
    TILING_DATA_FIELD_DEF(bool, is_from_knn);
    TILING_DATA_FIELD_DEF(uint32_t, inner);
    TILING_DATA_FIELD_DEF(uint32_t, inner2);
    TILING_DATA_FIELD_DEF(uint32_t, topkmax);
    TILING_DATA_FIELD_DEF(uint32_t, topkmax2);
    TILING_DATA_FIELD_DEF(uint32_t, loop_times);
    TILING_DATA_FIELD_DEF(uint32_t, b_times_m);
    TILING_DATA_FIELD_DEF(uint32_t, big_core_num);
    TILING_DATA_FIELD_DEF(uint32_t, small_core_num);
    TILING_DATA_FIELD_DEF(uint32_t, big_core_len);
    TILING_DATA_FIELD_DEF(uint32_t, small_core_len);
    TILING_DATA_FIELD_DEF(uint32_t, aligned_big_len);
    TILING_DATA_FIELD_DEF(uint32_t, aligned_small_len);
    TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTilingData);
    TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTilingData2);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Knn, KnnTilingData)

/****************class definition*****************/
class KnnTiling {
public:
    explicit KnnTiling(gert::TilingContext *context) : TilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunTiling();
private:
    CalcDataSizeFuncReturn CalcDataSize();
    CalcDataSizeFuncReturn CalcDataSizeUbNotEnough();
private:
    KnnTilingData TilingData;

    gert::TilingContext* TilingContext = nullptr;
    uint64_t ub_size;
    uint32_t core_num;
    uint32_t dtype_size_;

    uint32_t batch = 0;
    uint32_t npoint = 0;
    uint32_t nsample = 0;
    uint32_t nsample_aligned = 0;
    uint32_t nsource = 0;
    uint32_t nsource_aligned = 0;
    uint32_t nsource_aligned2 = 0;
    uint32_t nsource_aligned_size = 0;
    uint32_t nsource_aligned_size2 = 0;
    bool is_from_knn = false;
    uint32_t inner = 0;
    uint32_t inner2 = 0;
    uint32_t topkmax = 0;
    uint32_t topkmin = 0;
    uint32_t topkmax2 = 0;
    uint32_t topkmin2 = 0;
    uint32_t loop_times = 0;
    uint32_t b_times_m = 0;
    uint32_t big_core_num = 0;
    uint32_t small_core_num = 0;
    uint32_t big_core_len = 0;
    uint32_t small_core_len = 0;
    uint32_t aligned_big_len = 0;
    uint32_t aligned_small_len = 0;
    uint32_t target_size = 96; // the size of target/center_xyz we need to move into UB
    uint32_t source_size = 0; // the size of source/xyz we need to move into UB
    uint32_t source_backup_size = 0; // the size for backup of source/xyz in UB, equal to source_size
    uint32_t dist_size = 0; // the size of UB need for caching dist calculated by source/xyz
};
}

#endif // KNN_TILING_H
