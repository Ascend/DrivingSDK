/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "furthest_point_sampling_with_dist_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingForFurthestPointSamplingWithDist(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    FurthestPointSamplingWithDistTilingData tiling;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static uint32_t core_num = ascendcPlatform.GetCoreNumAiv();
    uint64_t UB_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, UB_size);
    
    auto dist_shape_ptr = context->GetInputShape(0);
    if (dist_shape_ptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dist_shape = dist_shape_ptr->GetStorageShape();
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (core_num == 0) {
        return ge::GRAPH_FAILED;
    }
    
    uint32_t points_num = *(attrs->GetAttrPointer<int32_t>(0));
    uint32_t b = dist_shape.GetDim(0);
    uint32_t n = dist_shape.GetDim(1);

    uint32_t dtype_bytes = 4;
    uint32_t ele_per_block = 8;

    uint32_t task_num = (b - 1) / core_num + 1;
    if (task_num == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t used_core_num = (b - 1) / task_num + 1;
    uint32_t task_num_tail = b % task_num;
    if (task_num_tail == 0) {
        task_num_tail = task_num;
    }

    uint32_t batch_dist_offset = n * n;
    uint32_t batch_idx_offset = points_num;

    uint32_t part = 5 * dtype_bytes + 1;
    uint32_t part_ub = (UB_size - 20 * 1024) / part / 32 * 32;
    if (part_ub == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t move_n_times = (n - 1) / part_ub + 1;
    uint32_t n_tail = n % part_ub;
    if (n_tail == 0) {
        n_tail = part_ub;
    }
    uint32_t id_move_len = 1024;
    uint32_t repeat_id_times = (points_num - 1) / id_move_len + 1;
    uint32_t id_tail = points_num % id_move_len;
    if (id_tail == 0) {
        id_tail = id_move_len;
    }
    uint32_t work_size = 1024 * 2;
    context->SetBlockDim(used_core_num);

    tiling.set_used_core_num(used_core_num);
    tiling.set_points_num(points_num);
    tiling.set_task_num(task_num);
    tiling.set_task_num_tail(task_num_tail);
    tiling.set_n(n);
    tiling.set_batch_dist_offset(batch_dist_offset);
    tiling.set_batch_idx_offset(batch_idx_offset);
    tiling.set_part_ub(part_ub);
    tiling.set_move_n_times(move_n_times);
    tiling.set_n_tail(n_tail);
    tiling.set_id_move_len(id_move_len);
    tiling.set_repeat_id_times(repeat_id_times);
    tiling.set_id_tail(id_tail);
    tiling.set_work_size(work_size);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* points_dist_shape = context->GetInputShape(0);
    if (points_dist_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t points_num = *(attrs->GetAttrPointer<int32_t>(0));

    gert::Shape* idx_shape = context->GetOutputShape(0);
    if (idx_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    idx_shape->AppendDim(points_dist_shape->GetDim(0));
    idx_shape->AppendDim(points_num);
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class FurthestPointSamplingWithDist : public OpDef {
public:
    explicit FurthestPointSamplingWithDist(const char* name) : OpDef(name)
    {
        this->Input("points_dist")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("nearest_temp")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("num_points").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingForFurthestPointSamplingWithDist);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(FurthestPointSamplingWithDist);
}
