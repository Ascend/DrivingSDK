/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "points_in_box_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace ge;
using namespace std;
using namespace AscendC;
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static int32_t GetCeilInt(int32_t value1, int32_t value2)
{
    if (value2 == 0) {
        return value1;
    }
    return static_cast<int32_t>((value1 + value2 - 1) / value2);
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    PointsInBoxTilingData tiling;
    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    auto core_number = ascendplatformInfo.GetCoreNumAiv();
    if (context->GetInputTensor(0) == nullptr || context->GetInputTensor(1) ==nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t totalresult = context->GetInputTensor(1)->GetShapeSize() / 3;
    auto boxes_shape = context->GetInputTensor(0)->GetStorageShape();
    auto points_shape = context->GetInputTensor(1)->GetStorageShape();
    int32_t core_data;
    int32_t core_used;
    int32_t core_last;
    core_data = GetCeilInt(totalresult, core_number);
    core_data = GetCeilInt(core_data, 64) * 64;
    core_used = GetCeilInt(totalresult, core_data);
    core_last = core_data;
    if (core_data == 0) {
        return ge::GRAPH_FAILED;
    }
    if (totalresult % core_data != 0) {
        core_last = totalresult % core_data;
    }
    uint64_t available_ub_size;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, available_ub_size);
    available_ub_size = (available_ub_size - 20*1024) / 50 / 4;
    available_ub_size = GetCeilInt(available_ub_size, 32) * 32;
    if (available_ub_size == 0) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(core_used);
    tiling.set_core_data(core_data);
    tiling.set_core_used(core_used);
    tiling.set_copy_loop(core_data / available_ub_size);
    tiling.set_copy_tail(core_data % available_ub_size);
    tiling.set_last_copy_loop(core_last / available_ub_size);
    tiling.set_last_copy_tail(core_last % available_ub_size);
    tiling.set_batch(boxes_shape.GetDim(0));
    tiling.set_npoints(points_shape.GetDim(1));
    tiling.set_box_number(boxes_shape.GetDim(2));
    tiling.set_available_ub_size(available_ub_size);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* pts_shape = context->GetInputShape(1);
    if (pts_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (y_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    y_shape->SetDimNum(0);
    y_shape->AppendDim(pts_shape->GetDim(0));
    y_shape->AppendDim(pts_shape->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus PointsInBoxInferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class PointsInBox : public OpDef {
public:
    explicit PointsInBox(const char* name) : OpDef(name)
    {
        this->Input("boxes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("pts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("boxes_idx_of_points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::PointsInBoxInferDataType);
        
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(PointsInBox);
}
