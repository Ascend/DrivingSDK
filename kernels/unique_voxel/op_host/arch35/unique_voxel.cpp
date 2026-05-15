/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include <graph/ge_error_codes.h>
#include <graph/types.h>
#include <register/op_def.h>

#include <algorithm>
#include <cmath>

#include "common/op_host/common.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "unique_voxel/op_host/unique_voxel_tiling.h"
namespace {
constexpr size_t POINT_IDX = 0;
constexpr int32_t RESERVE_UB = 10 * 1024; // 10 KB
constexpr int32_t COEF = 17; // 4[vox1, vox2, idx, arg] * 4[float size] + 1[temp] = 17
constexpr int32_t ONE_REPEAT_FLOAT_SIZE = 64;

ge::graphStatus TaskSchedule(gert::TilingContext *context, optiling::UniqueVoxelTilingData &tilingData) {
    auto platformInfo = context->GetPlatformInfo();
    if (!platformInfo) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    int32_t core_num = ascendcPlatform.GetCoreNumAiv();

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    int32_t avgPts =
        FloorAlign<int32_t>((ubSize - RESERVE_UB) / COEF, ONE_REPEAT_FLOAT_SIZE); // avgPts must be multiple of 64
    auto pointShape = context->GetInputShape(POINT_IDX);
    if (!pointShape) {
        return ge::GRAPH_FAILED;
    }
    int32_t totalPts = pointShape->GetStorageShape().GetDim(0);
    avgPts = std::min(avgPts, CeilAlign<int32_t>(totalPts, ONE_REPEAT_FLOAT_SIZE));
    if (avgPts == 0) {
        return ge::GRAPH_FAILED;
    }
    int32_t tailPts = totalPts % avgPts;
    int32_t totalTasks = totalPts / avgPts + (tailPts > 0 ? 1 : 0);
    tailPts = tailPts == 0 ? avgPts : tailPts;
    int32_t usedBlkNum = std::min(core_num, totalTasks);
    if (usedBlkNum == 0) {
        return ge::GRAPH_FAILED;
    }

    int32_t avgTasks = totalTasks / usedBlkNum;
    int32_t tailTasks = totalTasks % usedBlkNum;

    tilingData.set_usedBlkNum(usedBlkNum);
    tilingData.set_avgTasks(avgTasks);
    tilingData.set_tailTasks(tailTasks);
    tilingData.set_totalTasks(totalTasks);
    tilingData.set_avgPts(avgPts);
    tilingData.set_tailPts(tailPts);
    tilingData.set_totalPts(totalPts);

    context->SetBlockDim(usedBlkNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddWorkspace(gert::TilingContext *context, optiling::UniqueVoxelTilingData &tilingData) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // outputIdx0, count0, outputIdx1, ...
    uint64_t alignSize = 256;
    uint64_t totalPts = tilingData.get_totalPts();
    uint64_t usrWorkspaceSize = CeilAlign<int64_t>(tilingData.get_usedBlkNum() * sizeof(int32_t) * 2, alignSize);
    uint64_t spaceNum = 3; // tmpVox, tmpArg, tmpIdx
    usrWorkspaceSize += CeilAlign<int64_t>((totalPts + 1) * sizeof(int32_t), alignSize) * spaceNum;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = sysWorkspaceSize + usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingForUniqueVoxel(gert::TilingContext *context) {
    if (!context) {
        return ge::GRAPH_FAILED;
    }

    UniqueVoxelTilingData tilingData;

    if (TaskSchedule(context, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (AddWorkspace(context, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForUniqueVoxel(gert::InferShapeContext *context) {
    if (!context) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *pointShape = context->GetInputShape(POINT_IDX);
    gert::Shape *uniVoxShape = context->GetOutputShape(0);
    gert::Shape *uniIdxShape = context->GetOutputShape(1);
    gert::Shape *uniArgsortIdxShape = context->GetOutputShape(2);
    gert::Shape *voxNumShape = context->GetOutputShape(3);
    if (!pointShape || !uniVoxShape || !uniArgsortIdxShape || !uniIdxShape || !voxNumShape) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputShape(1) == nullptr || context->GetInputShape(2) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *uniVoxShape = *pointShape;
    *uniIdxShape = *pointShape;
    *uniArgsortIdxShape = *pointShape;
    *voxNumShape = {1};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForUniqueVoxel(gert::InferDataTypeContext *context) {
    if (!context) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(2, ge::DT_INT32);
    context->SetOutputDataType(3, ge::DT_INT32);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class UniqueVoxel : public OpDef {
  public:
    explicit UniqueVoxel(const char *name) : OpDef(name) {
        this->Input("voxels")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("argsort_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("uni_voxels")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("uni_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("uni_argsort_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("voxel_num")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->SetInferShape(ge::InferShapeForUniqueVoxel).SetInferDataType(ge::InferDataTypeForUniqueVoxel);
        this->AICore().SetTiling(optiling::TilingForUniqueVoxel);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(UniqueVoxel);
} // namespace ops
