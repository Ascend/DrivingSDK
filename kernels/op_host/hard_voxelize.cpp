/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include <exe_graph/runtime/runtime_attrs.h>
#include <graph/ge_error_codes.h>
#include <graph/types.h>
#include <register/op_def.h>

#include <algorithm>
#include <cmath>
#include <iostream>

#include "hard_voxelize_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace {
constexpr size_t POINT_IDX = 0;
constexpr size_t UNI_INDICIES_IDX = 4;
constexpr int32_t RESERVE_UB = 10 * 1024; // 10 KB
constexpr int32_t DIFF_COEF = 24;
constexpr int32_t ONE_REPEAT_FLOAT_SIZE = 64;
constexpr int32_t ONE_BLK_SIZE = 32;
constexpr int32_t B32_BYTES = 4;
constexpr int32_t FREE_NUM = 1024;
constexpr int32_t ONE_BLK_FLOAT_NUM = 8;
constexpr int64_t ALIGN_TILING_KEY = 1;
constexpr int64_t NOT_ALIGN_TILING_KEY = 0;

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
int32_t GetRealVoxelNum(const gert::RuntimeAttrs* attrs)
{
    auto getAttr = [attrs](size_t idx) -> int32_t {
        auto ptr = attrs->GetInt(idx);
        if (!ptr) {
            return -1;
        }
        return static_cast<int32_t>(*ptr);
    };

    int32_t numVoxels = getAttr(0);
    int32_t maxVoxels = getAttr(1);
    return std::min(numVoxels, maxVoxels);
}
ge::graphStatus TaskScheduleForDiff(gert::TilingContext* context, optiling::HardVoxelizeTilingData& tilingData)
{
    auto platformInfo = context->GetPlatformInfo();
    if (!platformInfo) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    int32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    int32_t avgPts =
        AlignDown((ubSize - RESERVE_UB) / DIFF_COEF, ONE_REPEAT_FLOAT_SIZE); // avgPts must be multiple of 64
    auto uniIndicesShape = context->GetInputShape(UNI_INDICIES_IDX);
    if (!uniIndicesShape) {
        return ge::GRAPH_FAILED;
    }
    int32_t totalPts = uniIndicesShape->GetStorageShape().GetDim(0);
    avgPts = std::min(avgPts, AlignUp(totalPts, ONE_REPEAT_FLOAT_SIZE));
    if (avgPts == 0) {
        return ge::GRAPH_FAILED;
    }
    int32_t tailPts = totalPts % avgPts;
    int32_t totalDiffTasks = totalPts / avgPts + (tailPts > 0 ? 1 : 0);
    tailPts = tailPts == 0 ? avgPts : tailPts;
    int32_t usedDiffBlkNum = std::min(coreNum, totalDiffTasks);
    if (usedDiffBlkNum == 0) {
        return ge::GRAPH_FAILED;
    }

    int32_t avgDiffTasks = totalDiffTasks / usedDiffBlkNum;
    int32_t tailDiffTasks = totalDiffTasks % usedDiffBlkNum;

    auto attrs = context->GetAttrs();
    int32_t realVoxelNum = GetRealVoxelNum(attrs);
    auto numPtsPtr = attrs->GetInt(3);
    if (!numPtsPtr) {
        return ge::GRAPH_FAILED;
    }

    tilingData.set_avgPts(avgPts);
    tilingData.set_tailPts(tailPts);
    tilingData.set_totalPts(totalPts);
    tilingData.set_numPts(*numPtsPtr);
    tilingData.set_avgDiffTasks(avgDiffTasks);
    tilingData.set_tailDiffTasks(tailDiffTasks);
    tilingData.set_totalDiffTasks(totalDiffTasks);
    tilingData.set_usedDiffBlkNum(usedDiffBlkNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TaskScheduleForCopy(gert::TilingContext* context, optiling::HardVoxelizeTilingData& tilingData)
{
    auto platformInfo = context->GetPlatformInfo();
    if (!platformInfo) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    int32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    auto attrs = context->GetAttrs();
    int32_t realVoxelNum = GetRealVoxelNum(attrs);
    auto maxPointsPtr = attrs->GetInt(2);
    if (!maxPointsPtr) {
        return ge::GRAPH_FAILED;
    }
    int32_t maxPoints = *maxPointsPtr;

    int32_t usedCopyBlkNum = std::min(coreNum, realVoxelNum);
    if (usedCopyBlkNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto pointShape = context->GetInputShape(POINT_IDX);
    if (!pointShape) {
        return ge::GRAPH_FAILED;
    }
    int32_t featNum = pointShape->GetStorageShape().GetDim(1);
    int32_t avgVoxs = (ubSize - RESERVE_UB - 2 * AlignUp(featNum, ONE_BLK_SIZE) * sizeof(float) - 3 * ONE_BLK_SIZE -
                          FREE_NUM * sizeof(int32_t)) /
                      B32_BYTES;
    avgVoxs = std::min(avgVoxs, (realVoxelNum + usedCopyBlkNum - 1) / usedCopyBlkNum);
    avgVoxs = AlignUp(avgVoxs, ONE_BLK_FLOAT_NUM);
    if (avgVoxs == 0) {
        return ge::GRAPH_FAILED;
    }

    int32_t tailVoxs = realVoxelNum % avgVoxs;
    int32_t totalCopyTasks = realVoxelNum / avgVoxs + (tailVoxs > 0 ? 1 : 0);
    tailVoxs = tailVoxs == 0 ? avgVoxs : tailVoxs;

    usedCopyBlkNum = std::min(coreNum, totalCopyTasks);
    int32_t avgCopyTasks = totalCopyTasks / usedCopyBlkNum;
    int32_t tailCopyTasks = totalCopyTasks % usedCopyBlkNum;

    tilingData.set_usedCopyBlkNum(usedCopyBlkNum);
    tilingData.set_avgVoxs(avgVoxs);
    tilingData.set_tailVoxs(tailVoxs);
    tilingData.set_totalVoxs(realVoxelNum);
    tilingData.set_avgCopyTasks(avgCopyTasks);
    tilingData.set_tailCopyTasks(tailCopyTasks);
    tilingData.set_totalCopyTasks(totalCopyTasks);
    tilingData.set_featNum(featNum);
    tilingData.set_freeNum(FREE_NUM);
    tilingData.set_maxPoints(maxPoints);

    return ge::GRAPH_SUCCESS;
}
ge::graphStatus AddWorkspace(gert::TilingContext* context, optiling::HardVoxelizeTilingData& tilingData)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint32_t usrWorkspaceSize = tilingData.get_totalPts() * sizeof(int32_t); // uniLens
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize + usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
} // namespace


namespace optiling {
static ge::graphStatus TilingForHardVoxelize(gert::TilingContext* context)
{
    if (!context) {
        return ge::GRAPH_FAILED;
    }

    HardVoxelizeTilingData tilingData;

    if (TaskScheduleForDiff(context, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (TaskScheduleForCopy(context, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (AddWorkspace(context, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tilingData.get_featNum() % ONE_BLK_FLOAT_NUM == 0 ? ALIGN_TILING_KEY : NOT_ALIGN_TILING_KEY);
    context->SetBlockDim(std::max(tilingData.get_usedDiffBlkNum(), tilingData.get_usedCopyBlkNum()));
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


namespace ge {
static ge::graphStatus InferShapeForHardVoxelize(gert::InferShapeContext* context)
{
    if (!context) {
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    if (!attrs) {
        return ge::GRAPH_FAILED;
    }

    auto maxPointsPtr = attrs->GetInt(2);
    if (!maxPointsPtr) {
        return ge::GRAPH_FAILED;
    }
    int32_t maxPoints = *maxPointsPtr;

    int32_t realVoxels = GetRealVoxelNum(attrs);

    const gert::Shape* pointShape = context->GetInputShape(POINT_IDX);
    if (!pointShape) {
        return ge::GRAPH_FAILED;
    }
    int32_t featNum = pointShape->GetDim(1);
    gert::Shape* voxelShape = context->GetOutputShape(0);
    gert::Shape* numPointsPerVoxel = context->GetOutputShape(1);
    gert::Shape* sortedUniVoxels = context->GetOutputShape(2);
    if (!voxelShape || !numPointsPerVoxel || !sortedUniVoxels) {
        return ge::GRAPH_FAILED;
    }
    *voxelShape = {realVoxels, maxPoints, featNum};
    *numPointsPerVoxel = {realVoxels};
    *sortedUniVoxels = {realVoxels};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForHardVoxelize(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class HardVoxelize : public OpDef {
public:
    explicit HardVoxelize(const char* name) : OpDef(name)
    {
        this->Input("points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("uni_voxels")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("argsort_voxel_idices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("uni_argsort_idices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("uni_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("num_voxels").AttrType(REQUIRED).Int();
        this->Attr("max_voxels").AttrType(REQUIRED).Int();
        this->Attr("max_points").AttrType(REQUIRED).Int();
        this->Attr("num_points").AttrType(REQUIRED).Int();

        this->Output("voxels")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("num_points_per_voxel")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("sorted_uni_voxels")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->SetInferShape(ge::InferShapeForHardVoxelize).SetInferDataType(ge::InferDataTypeForHardVoxelize);
        this->AICore().SetTiling(optiling::TilingForHardVoxelize);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(HardVoxelize);
} // namespace ops
