/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include <graph/ge_error_codes.h>
#include <graph/types.h>
#include <register/op_def.h>

#include <algorithm>
#include <cmath>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "unique_voxel_tiling.h"

namespace {
constexpr size_t POINT_IDX = 0;
constexpr int32_t RESERVE_UB = 10 * 1024; // 10 KB
// 2[double buffer] * （3[vox1, vox2, idx] + 3[uniVox, uniIdx, uinArgsortIdx]）* 4[float size] + 1[temp] = 49
constexpr int32_t COEF = 49;
constexpr int32_t ONE_REPEAT_FLOAT_SIZE = 64;

int32_t AlignDown(int32_t a, int32_t b)
{
    if (b == 0) {
        return a;
    }
    return a / b * b;
}

int32_t AlignUp(int32_t a, int32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

ge::graphStatus TaskSchedule(gert::TilingContext* context, optiling::UniqueVoxelTilingData& tilingData)
{
    auto platformInfo = context->GetPlatformInfo();
    if (!platformInfo) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    int32_t core_num = ascendcPlatform.GetCoreNumAiv();

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    int32_t avgPts = AlignDown((ubSize - RESERVE_UB) / COEF, ONE_REPEAT_FLOAT_SIZE); // avgPts must be multiple of 64
    auto pointShape = context->GetInputShape(POINT_IDX);
    if (!pointShape) {
        return ge::GRAPH_FAILED;
    }
    int32_t totalPts = pointShape->GetStorageShape().GetDim(0);
    avgPts = std::min(avgPts, AlignUp(totalPts, ONE_REPEAT_FLOAT_SIZE));
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

ge::graphStatus AddWorkspace(gert::TilingContext* context, optiling::UniqueVoxelTilingData& tilingData)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // outputIdx0, count0, outputIdx1, ...
    uint32_t usrWorkspaceSize = tilingData.get_usedBlkNum() * sizeof(int32_t) * 2;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize + usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
} // namespace


namespace optiling {
static ge::graphStatus TilingForUniqueVoxel(gert::TilingContext* context)
{
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
static ge::graphStatus InferShapeForUniqueVoxel(gert::InferShapeContext* context)
{
    const gert::Shape* pointShape = context->GetInputShape(POINT_IDX);
    gert::Shape* uniVoxShape = context->GetOutputShape(0);
    gert::Shape* uniIdxShape = context->GetOutputShape(1);
    gert::Shape* uniArgsortIdxShape = context->GetOutputShape(2);
    gert::Shape* voxNumShape = context->GetOutputShape(3);
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

static ge::graphStatus InferDataTypeForUniqueVoxel(gert::InferDataTypeContext* context)
{
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
    explicit UniqueVoxel(const char* name) : OpDef(name)
    {
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
            .DataType({ge::DT_FLOAT})
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
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(UniqueVoxel);
} // namespace ops
