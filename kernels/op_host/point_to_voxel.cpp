/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include <graph/ge_error_codes.h>
#include <graph/types.h>
#include <register/op_def.h>
#include <register/tilingdata_base.h>
#include <strings.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "point_to_voxel_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t POINT_IDX = 0;
constexpr uint64_t RAW_TILING_KEY = 0;
constexpr uint64_t COOR_TILING_KEY = 1;
constexpr uint64_t XYZ_TILING_KEY = 0;
constexpr uint64_t ZYX_TILING_KEY = 1;
constexpr size_t VOXEL_SIZES_IDX = 0;
constexpr size_t COOR_MINS_IDX = 1;
constexpr uint64_t LAYOUT_IDX = 2;
constexpr int32_t RESERVE_UB = 10 * 1024; // 10 KB
// 2[double buffer] * （3[coorx, coor_y, coor_z] + 1[output]）* 4[float size] + 1[temp] = 33
constexpr int32_t COEF = 33;
constexpr int32_t ONE_REPEAT_FLOAT_SIZE = 64;
constexpr int DEFAULT_GRID_X = 2048;
constexpr int DEFAULT_GRID_Y = 2048;
constexpr int DEFAULT_GRID_Z = 256;

template<typename T>
ge::graphStatus GetElementInListAttr(const gert::RuntimeAttrs* attrs, size_t index, size_t offset, T& value)
{
    if (!attrs) {
        return ge::GRAPH_FAILED;
    }
    const gert::TypedContinuousVector<T>* list = attrs->GetAttrPointer<gert::TypedContinuousVector<T>>(index);
    if (!list) {
        return ge::GRAPH_FAILED;
    }

    const T* data = list->GetData();
    if (!data) {
        return ge::GRAPH_FAILED;
    }
    if (list->GetSize() <= offset) {
        return ge::GRAPH_FAILED;
    }
    value = data[offset];
    return ge::GRAPH_SUCCESS;
}

template<bool forward>
ge::graphStatus SetTilingKey(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    if (!attrs) {
        return ge::GRAPH_FAILED;
    }
    auto layout = attrs->GetStr(LAYOUT_IDX);
    if (layout == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (!forward) {
        return context->SetTilingKey(strcmp(layout, "XYZ") != 0 ? ZYX_TILING_KEY : XYZ_TILING_KEY);
    }

    float voxelSizeX;
    if (GetElementInListAttr(attrs, VOXEL_SIZES_IDX, 0, voxelSizeX) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return context->SetTilingKey(((voxelSizeX > 0 ? RAW_TILING_KEY : COOR_TILING_KEY) << 1) |
                                 (strcmp(layout, "XYZ") != 0 ? ZYX_TILING_KEY : XYZ_TILING_KEY));
}

ge::graphStatus SetTilingDataFromAttr(const gert::TilingContext* context, optiling::PointToVoxelTilingData& tilingData)
{
    auto attrs = context->GetAttrs();
    if (!attrs) {
        return ge::GRAPH_FAILED;
    }
    int32_t gridX(DEFAULT_GRID_X), gridY(DEFAULT_GRID_Y), gridZ(DEFAULT_GRID_Z);
    float voxelSizeX, voxelSizeY, voxelSizeZ, coorXMax, coorYMax, coorZMax, coorXMin, coorYMin, coorZMin;
    if (GetElementInListAttr(attrs, VOXEL_SIZES_IDX, 0, voxelSizeX) != ge::GRAPH_SUCCESS ||
        GetElementInListAttr(attrs, VOXEL_SIZES_IDX, 1, voxelSizeY) != ge::GRAPH_SUCCESS ||
        GetElementInListAttr(attrs, VOXEL_SIZES_IDX, 2, voxelSizeZ) != ge::GRAPH_SUCCESS ||
        GetElementInListAttr(attrs, COOR_MINS_IDX, 0, coorXMin) != ge::GRAPH_SUCCESS ||
        GetElementInListAttr(attrs, COOR_MINS_IDX, 1, coorYMin) != ge::GRAPH_SUCCESS ||
        GetElementInListAttr(attrs, COOR_MINS_IDX, 2, coorZMin) != ge::GRAPH_SUCCESS ||
        GetElementInListAttr(attrs, COOR_MINS_IDX, 3, coorXMax) != ge::GRAPH_SUCCESS ||
        GetElementInListAttr(attrs, COOR_MINS_IDX, 4, coorYMax) != ge::GRAPH_SUCCESS ||
        GetElementInListAttr(attrs, COOR_MINS_IDX, 5, coorZMax) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (voxelSizeX > 0 && voxelSizeY > 0 && voxelSizeZ > 0) {
        gridX = std::round((coorXMax - coorXMin) / voxelSizeX);
        gridY = std::round((coorYMax - coorYMin) / voxelSizeY);
        gridZ = std::round((coorZMax - coorZMin) / voxelSizeZ);
    }

    tilingData.set_gridX(gridX);
    tilingData.set_gridY(gridY);
    tilingData.set_gridZ(gridZ);
    tilingData.set_voxelSizeX(voxelSizeX);
    tilingData.set_voxelSizeY(voxelSizeY);
    tilingData.set_voxelSizeZ(voxelSizeZ);
    tilingData.set_coorXMin(coorXMin);
    tilingData.set_coorYMin(coorYMin);
    tilingData.set_coorZMin(coorZMin);
    return ge::GRAPH_SUCCESS;
}
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

template<bool forward>
ge::graphStatus TaskSchedule(gert::TilingContext* context, optiling::PointToVoxelTilingData& tilingData)
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
    int32_t totalPts = forward ? pointShape->GetStorageShape().GetDim(1) : pointShape->GetStorageShape().GetDim(0);
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

} // namespace


namespace optiling {
template<bool forward>
static ge::graphStatus TilingForPointToVoxel(gert::TilingContext* context)
{
    if (!context) {
        return ge::GRAPH_FAILED;
    }
    PointToVoxelTilingData tilingData;

    if (SetTilingKey<forward>(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (SetTilingDataFromAttr(context, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (TaskSchedule<forward>(context, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


namespace ge {
template<bool forward>
static graphStatus InferShapeForPointToVoxel(gert::InferShapeContext* context)
{
    const gert::Shape* pointShape = context->GetInputShape(POINT_IDX);
    gert::Shape* outShape = context->GetOutputShape(0);
    if (!pointShape || !outShape) {
        return ge::GRAPH_FAILED;
    }

    if (forward) {
        *outShape = {pointShape->GetDim(1)};
    } else {
        *outShape = {3, pointShape->GetDim(0)};
    }

    return GRAPH_SUCCESS;
}


template<bool forward>
static graphStatus InferDataTypeForPointToVoxel(gert::InferDataTypeContext* context)
{
    if (forward) {
        context->SetOutputDataType(0, ge::DT_FLOAT);
    } else {
        context->SetOutputDataType(0, ge::DT_INT32);
    }
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class PointToVoxel : public OpDef {
public:
    explicit PointToVoxel(const char* name) : OpDef(name)
    {
        this->Input("points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("voxels")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("voxel_sizes").AttrType(REQUIRED).ListFloat();
        this->Attr("coor_ranges").AttrType(REQUIRED).ListFloat();
        this->Attr("layout").AttrType(REQUIRED).String("XYZ");

        this->SetInferShape(ge::InferShapeForPointToVoxel<true>)
            .SetInferDataType(ge::InferDataTypeForPointToVoxel<true>);
        this->AICore().SetTiling(optiling::TilingForPointToVoxel<true>);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

class VoxelToPoint : public OpDef {
public:
    explicit VoxelToPoint(const char* name) : OpDef(name)
    {
        this->Input("voxels")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("voxel_sizes").AttrType(REQUIRED).ListFloat();
        this->Attr("coor_ranges").AttrType(REQUIRED).ListFloat();
        this->Attr("layout").AttrType(REQUIRED).String("XYZ");

        this->SetInferShape(ge::InferShapeForPointToVoxel<false>)
            .SetInferDataType(ge::InferDataTypeForPointToVoxel<false>);
        this->AICore().SetTiling(optiling::TilingForPointToVoxel<false>);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(PointToVoxel);
OP_ADD(VoxelToPoint);
} // namespace ops
