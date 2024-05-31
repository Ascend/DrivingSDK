/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */

#include "knn_tiling.h"
#include "common.h"

/****************constexpr definition*****************/
constexpr int64_t KNNDIM = 3;
constexpr int64_t KNN_BLOCK_SIZE = 32;
constexpr int64_t KNN_UBRESERVE = 2;
constexpr int64_t KNN_FP32_SIZE = 4;
constexpr int64_t KNN_FP16_SIZE = 2;

namespace optiling {
/****************class impl*****************/
ge::graphStatus KnnTiling::Init()
{
    if (TilingContext == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape *xyz_shape = TilingContext->GetInputShape(0);
    const gert::StorageShape *center_xyz_shape = TilingContext->GetInputShape(1);
    const gert::RuntimeAttrs *attr = TilingContext->GetAttrs();
    auto platformInfoPtr = TilingContext->GetPlatformInfo();
    ge::DataType dtype_;
    bool is_float = true;
    this->dtype_size_ = KNN_FP32_SIZE;
    if ((xyz_shape == nullptr) || (center_xyz_shape == nullptr) || (attr == nullptr) || (platformInfoPtr == nullptr) ||
        (TilingContext->GetInputDesc(0) == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    if ((attr->GetAttrPointer<uint32_t>(0) == nullptr) || (attr->GetAttrPointer<bool>(1) == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    dtype_ = TilingContext->GetInputDesc(0)->GetDataType();
    if (dtype_ == ge::DT_FLOAT) {
        is_float = true;
        this->dtype_size_ = KNN_FP32_SIZE;
    } else if (dtype_ == ge::DT_FLOAT16) {
        is_float = false;
        this->dtype_size_ = KNN_FP16_SIZE;
    } else {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    this->batch = center_xyz_shape->GetStorageShape().GetDim(0);
    this->npoint = center_xyz_shape->GetStorageShape().GetDim(1);
    this->nsample = *attr->GetAttrPointer<uint32_t>(0);
    this->nsource = xyz_shape->GetStorageShape().GetDim(2);
    this->is_from_knn = *attr->GetAttrPointer<bool>(1);
    this->core_num = platformInfo.GetCoreNumAiv();
    if (this->core_num == 0) {
        return ge::GRAPH_FAILED;
    }
    // divide the core by (batch x npoint)
    this->nsample_aligned = ceil_value(this->nsample, KNN_BLOCK_SIZE / this->dtype_size_);
    this->nsource_aligned = ceil_value(this->nsource, KNN_BLOCK_SIZE / this->dtype_size_);
    this->nsource_aligned_size = this->nsource_aligned * this->dtype_size_;
    this->b_times_m = this->batch * this->npoint;
    this->small_core_len = this->b_times_m / this->core_num;
    this->big_core_len = this->small_core_len + 1;
    this->big_core_num = this->b_times_m - this->small_core_len * this->core_num; // 910B1range:0-39
    this->small_core_num = this->core_num - this->big_core_num; // 910B1range:1-40
    this->aligned_big_len = ceil_value(this->big_core_len * 3, KNN_BLOCK_SIZE / this->dtype_size_);
    this->aligned_small_len = ceil_value(this->small_core_len * 3, KNN_BLOCK_SIZE / this->dtype_size_);
    this->inner = ceil_value(this->nsource, 32) * 32;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, this->ub_size);

    // setting tiling key...
    enum CalcDataSizeFuncReturn calcDataSizeFuncReturn;
    uint32_t data_size;
    uint32_t aligned_size_idx = ceil_value(this->nsample * sizeof(int32_t), KNN_BLOCK_SIZE) * KNN_BLOCK_SIZE;
    uint32_t aligned_size_dist2 = ceil_value(this->nsample * this->dtype_size_, KNN_BLOCK_SIZE) * KNN_BLOCK_SIZE;
    bool res = true;
    // case1
    res = AscendC::GetTopKMaxMinTmpSize(platformInfo, this->inner, 1, false, false, AscendC::TopKMode::TOPK_NORMAL,
        true, this->dtype_size_, this->topkmax, this->topkmin);
    if (!res) {
        return ge::GRAPH_FAILED;
    }
    // size = target + dist + source + source_backup + topkmax + idx + dist2
    this->source_size = this->nsource_aligned_size * 3;
    this->source_backup_size = this->source_size;
    data_size = this->target_size + this->inner * this->dtype_size_ + this->source_size + this->source_backup_size +
        this->topkmax + aligned_size_idx + aligned_size_dist2;
    if (data_size <= (this->ub_size - KNN_UBRESERVE)) {
         // in this case, we can move complete source in the UB
        if (is_float) {
            TilingContext->SetTilingKey(100);
        } else {
            TilingContext->SetTilingKey(101);
        }
        res = AscendC::TopKTilingFunc(platformInfo, this->inner, 1, this->nsample, this->dtype_size_,
            false, AscendC::TopKMode::TOPK_NORMAL, true, TilingData.topkTilingData);
        if (!res) {
            return ge::GRAPH_FAILED;
        }
    } else {
        // in this case, we need to split source/xyz
        if (is_float) {
            TilingContext->SetTilingKey(102);
        } else {
            TilingContext->SetTilingKey(103);
        }
        uint32_t size_tmp;
        uint32_t new_n;
        // calc the size of topkmax2
        this->inner2 = ceil_value(this->nsample_aligned * 2, 32);
        res = AscendC::GetTopKMaxMinTmpSize(platformInfo, this->inner2, 1, false, true, AscendC::TopKMode::TOPK_NORMAL,
            true, this->dtype_size_, this->topkmax2, this->topkmin2);
        if (!res) {
            return ge::GRAPH_FAILED;
        }
        // size = target + idx + dist2
        data_size = this->target_size + aligned_size_idx * 2 + this->inner2 * this->dtype_size_;
        // minimum size of source and topkmax is based on 32 data
        res = AscendC::GetTopKMaxMinTmpSize(platformInfo, 32, 1, false, false,
            AscendC::TopKMode::TOPK_NORMAL, true, this->dtype_size_, this->topkmax, this->topkmin);
        if (!res) {
            return ge::GRAPH_FAILED;
        }
        this->source_size = 96 * this->dtype_size_;
        this->source_backup_size = this->source_size;
        this->dist_size = this->source_size;
        // size_tmp = target + idx + dist2 + source + source_backup + max(dist + topkmax, topkmax2)
        size_tmp = data_size + this->source_size + this->source_backup_size +
            std::max(this->dist_size + this->topkmax, this->topkmax2);
        if (size_tmp > (this->ub_size - KNN_UBRESERVE)) {
            // 超出UB能处理的最大数据范围， 怎么办？
            return ge::GRAPH_FAILED;
        }
        // then at least the UB space is capable for 32 data
        this->loop_times = 1;
        new_n = this->nsource;
        do {
            this->loop_times *= 2;
            new_n = ceil_multiple(this->nsource, this->loop_times);
            this->nsource_aligned2 = ceil_value(new_n, KNN_BLOCK_SIZE / this->dtype_size_);
            this->nsource_aligned_size2 = this->nsource_aligned2 * this->dtype_size_;
            this->inner = ceil_value(new_n, 32);
            res = AscendC::GetTopKMaxMinTmpSize(platformInfo, this->inner, 1, false, false,
                AscendC::TopKMode::TOPK_NORMAL, true, this->dtype_size_, this->topkmax, this->topkmin);
            if (!res) {
                return ge::GRAPH_FAILED;
            }
            // size = target + idx + dist2 + source + source_backup + max(dist + topkmax, topkmax2)
            this->source_size = this->nsource_aligned_size2 * 3;
            this->source_backup_size = this->source_size;
            this->dist_size = this->inner * this->dtype_size_;
            data_size += this->source_size + this->source_backup_size +
                std::max(this->dist_size + this->topkmax, this->topkmax2);
        } while ((data_size > (this->ub_size - KNN_UBRESERVE)) && (new_n >= 32));
        res = AscendC::TopKTilingFunc(platformInfo, this->inner, 1, this->nsample, this->dtype_size_,
            false, AscendC::TopKMode::TOPK_NORMAL, true, TilingData.topkTilingData);
        if (!res) {
            return ge::GRAPH_FAILED;
        }
        res = AscendC::TopKTilingFunc(platformInfo, this->inner2, 1, this->nsample, this->dtype_size_,
            true, AscendC::TopKMode::TOPK_NORMAL, true, TilingData.topkTilingData2);
        if (!res) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus KnnTiling::RunTiling()
{
    size_t sysWorkspaceSize = 16 * 1024 * 1024; // Alloc 16M workspace
    size_t *currentWorkSpace = TilingContext->GetWorkspaceSizes(1);
    if (currentWorkSpace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkSpace[0] = sysWorkspaceSize;

    TilingData.set_batch(this->batch);
    TilingData.set_npoint(this->npoint);
    TilingData.set_nsample(this->nsample);
    TilingData.set_nsample_aligned(this->nsample_aligned);
    TilingData.set_nsource(this->nsource);
    TilingData.set_nsource_aligned(this->nsource_aligned);
    TilingData.set_nsource_aligned2(this->nsource_aligned2);
    TilingData.set_nsource_aligned_size(this->nsource_aligned_size);
    TilingData.set_nsource_aligned_size2(this->nsource_aligned_size2);
    TilingData.set_is_from_knn(this->is_from_knn);
    TilingData.set_inner(this->inner);
    TilingData.set_inner2(this->inner2);
    TilingData.set_topkmax(this->topkmax);
    TilingData.set_topkmax2(this->topkmax2);
    TilingData.set_loop_times(this->loop_times);
    TilingData.set_b_times_m(this->b_times_m);
    TilingData.set_big_core_num(this->big_core_num);
    TilingData.set_small_core_num(this->small_core_num);
    TilingData.set_big_core_len(this->big_core_len);
    TilingData.set_small_core_len(this->small_core_len);
    TilingData.set_aligned_big_len(this->aligned_big_len);
    TilingData.set_aligned_small_len(this->aligned_small_len);
    if ((this->b_times_m) < this->core_num) {
        TilingContext->SetBlockDim(this->b_times_m);
    } else {
        TilingContext->SetBlockDim(this->core_num);
    }
    TilingData.SaveToBuffer(TilingContext->GetRawTilingData()->GetData(), TilingContext->GetRawTilingData()->GetCapacity());
    TilingContext->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus TilingForKnn(gert::TilingContext *context)
{
    KnnTiling tilingObj(context);
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (tilingObj.Init() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (tilingObj.RunTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InfershapeForKnn(gert::InferShapeContext *context)
{
    const gert::Shape *xyz_shape = context->GetInputShape(0);
    const gert::Shape *center_xyz_shape = context->GetInputShape(1);
    gert::Shape *idx_shape = context->GetOutputShape(0);
    gert::Shape *dist2_shape = context->GetOutputShape(1);
    const gert::RuntimeAttrs *attr = context->GetAttrs();
    uint32_t batch;
    uint32_t npoint;
    if ((xyz_shape == nullptr) || (center_xyz_shape == nullptr) || (idx_shape == nullptr) || (dist2_shape == nullptr) ||
        (attr == nullptr)) {
            return ge::GRAPH_FAILED;
    }
    const int32_t *nsample = attr->GetAttrPointer<int32_t>(0);
    const bool *is_from_knn = attr->GetAttrPointer<bool>(1);
    if ((nsample == nullptr) || (is_from_knn == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    if ((xyz_shape->GetDimNum() != KNNDIM) || ((center_xyz_shape->GetDimNum() != KNNDIM))) {
        return ge::GRAPH_FAILED;
    }
    batch = center_xyz_shape->GetDim(0);
    npoint = center_xyz_shape->GetDim(1);
    idx_shape->SetDimNum(3);
    idx_shape->SetDim(0, batch);
    idx_shape->SetDim(1, npoint);
    idx_shape->SetDim(2, *nsample);
    dist2_shape->SetDimNum(3);
    dist2_shape->SetDim(0, batch);
    dist2_shape->SetDim(1, npoint);
    dist2_shape->SetDim(2, *nsample);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Knn : public OpDef {
public:
    explicit Knn(const char* name) : OpDef(name)
    {
        this->Input("xyz")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("center_xyz")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("nsample")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("is_from_knn")
            .AttrType(REQUIRED)
            .Bool();
        this->Output("idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("dist2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InfershapeForKnn);
        this->AICore().SetTiling(optiling::TilingForKnn);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910c", aicore_config);
    }
};

OP_ADD(Knn);
}