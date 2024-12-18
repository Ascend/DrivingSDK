/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#include <graph/types.h>
#include <log/log.h>
#include <register/op_def.h>

#include <cstdint>

#include "bev_pool_v3_tiling.h"
#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t INPUT_FEAT = 1;
constexpr size_t INPUT_FEAT_GRAD = 2;
constexpr size_t INPUT_RANKS_BEV = 4;
constexpr size_t INPUT_RANKS_BEV_GRAD = 5;
constexpr uint64_t RANK_NUM_PER_TASK = 1024;
constexpr int32_t ONE_BLK_SIZE = 8;
constexpr int32_t RESERVE_UB = 10 * 1024; // 10 KB
constexpr int32_t CHANNEL_IDX = 1;
constexpr int32_t CHANNEL_IDX_WITH_DEPTH = 4;
} // namespace

namespace optiling {
template<bool is_grad>
static ge::graphStatus TilingForBEVPoolV3(gert::TilingContext* context)
{
    CHECK_NULLPTR(context);
    BEVPoolV3TilingData tiling;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = platform.GetCoreNum();
    auto featShape = context->GetInputShape(is_grad ? INPUT_FEAT_GRAD : INPUT_FEAT);
    auto ranksBevShape = context->GetInputShape(is_grad ? INPUT_RANKS_BEV_GRAD : INPUT_RANKS_BEV);
    auto attrsPtr = context->GetAttrs();
    CHECK_NULLPTR(attrsPtr);

    auto withDepthPtr = attrsPtr->GetBool(0);
    CHECK_NULLPTR(withDepthPtr);

    bool withDepth = *withDepthPtr;
    context->SetTilingKey(withDepth);

    auto channel = featShape->GetOriginShape().GetDim(withDepth ? CHANNEL_IDX_WITH_DEPTH : CHANNEL_IDX);
    uint64_t ranks = ranksBevShape->GetOriginShape().GetDim(0);
    uint64_t avgRankNum = withDepth ?
                         RANK_NUM_PER_TASK :
                         (ubSize - RESERVE_UB) / (sizeof(float) * (channel + 1) * 2) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    avgRankNum = std::min(avgRankNum, ranks);
    if (avgRankNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto totalTaskNum = (ranks + avgRankNum - 1) / avgRankNum;
    uint64_t usedCoreNum = std::min(static_cast<uint64_t>(coreNum), totalTaskNum);
    if (usedCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(usedCoreNum);

    auto avgTaskNum = totalTaskNum / usedCoreNum;
    auto tailTaskNum = totalTaskNum % usedCoreNum;
    auto tailRankNum = ranks - (totalTaskNum - 1) * avgRankNum;
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_totalTaskNum(totalTaskNum);
    tiling.set_avgTaskNum(avgTaskNum);
    tiling.set_tailTaskNum(tailTaskNum);
    tiling.set_avgRankNum(avgRankNum);
    tiling.set_tailRankNum(tailRankNum);
    tiling.set_channel(channel);
    MX_DRIVING_LOGI("BEVPoolV3 tiling: usedCoreNum=%d, totalTaskNum=%d, avgTaskNum=%d, tailTaskNum=%d, avgRankNum=%d, "
                    "tailRankNum=%d, channel=%d",
        usedCoreNum, totalTaskNum, avgTaskNum, tailTaskNum, avgRankNum, tailRankNum, channel);

    ADD_TILING_DATA(context, tiling);

    uint32_t sysWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class BEVPoolV3 : public OpDef {
public:
    explicit BEVPoolV3(const char* name) : OpDef(name)
    {
        this->Input("depth")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous()
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous()
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ranks_depth")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous()
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ranks_feat")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous()
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ranks_bev")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous()
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("with_depth").Bool();

        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingForBEVPoolV3<false>);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

/**
 * @brief: BEVPoolGrad, the backward of bev_pool
 * @par Inputs:
 * grad_out: input grad, 5D tensor(b, d, h, w, c), dtype: float32, format:
 * NDHWC, ND geom_feat: input coords, 2D tensor(n, 4), dtype: int32, format: ND
 * interval_starts: starting position for pooled point, 1D tensor(n_interval),
 * dtype: int32, format: ND interval_lengths: the number of points in each
 * interval, 1D tensor(n_interval), dtype: int32, format: ND
 * @par Outputs:
 * grad_feat: output grad, 2D tensor(n, c), dtype: float32, format: ND
 * @par Attributes:
 **/
class BEVPoolV3Grad : public OpDef {
public:
    explicit BEVPoolV3Grad(const char* name) : OpDef(name)
    {
        this->Input("grad_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous()
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("depth")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous()
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous()
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ranks_depth")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous()
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ranks_feat")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous()
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ranks_bev")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous()
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("with_depth").Bool();

        this->Output("grad_depth")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingForBEVPoolV3<true>);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(BEVPoolV3);
OP_ADD(BEVPoolV3Grad);
} // namespace ops
