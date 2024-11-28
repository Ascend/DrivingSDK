/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#include <graph/types.h>
#include <log/log.h>
#include <register/op_def.h>

#include <cstdint>

#include "bev_pool_v3_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t INPUT_FEAT = 1;
constexpr size_t INPUT_FEAT_GRAD = 2;
constexpr size_t INPUT_RANKS_DEPTH = 2;
constexpr size_t INPUT_RANKS_DEPTH_GRAD = 3;
constexpr int32_t RANK_NUM_PER_TASK = 1024;
} // namespace

namespace optiling {
template<bool is_grad>
static ge::graphStatus TilingForBEVPoolV3(gert::TilingContext* context)
{
    BEVPoolV3TilingData tiling;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = platform.GetCoreNum();
    auto featShape = context->GetInputShape(is_grad ? INPUT_FEAT_GRAD : INPUT_FEAT);
    auto channel = featShape->GetOriginShape().GetDim(4);
    auto ranksDepthShape = context->GetInputShape(is_grad ? INPUT_RANKS_DEPTH_GRAD : INPUT_RANKS_DEPTH);
    int32_t ranks = ranksDepthShape->GetOriginShape().GetDim(0);

    int32_t avgRankNum = std::min(RANK_NUM_PER_TASK, ranks);
    if (avgRankNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto totalTaskNum = (ranks + avgRankNum - 1) / avgRankNum;
    int32_t usedCoreNum = std::min(static_cast<int32_t>(coreNum), totalTaskNum);
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

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    uint32_t sysWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
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
            .ParamType(REQUIRED)
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
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous()
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ranks_feat")
            .ParamType(REQUIRED)
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
            .ParamType(REQUIRED)
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
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous()
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ranks_feat")
            .ParamType(REQUIRED)
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

        this->Output("grad_depth")
            .ParamType(REQUIRED)
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
