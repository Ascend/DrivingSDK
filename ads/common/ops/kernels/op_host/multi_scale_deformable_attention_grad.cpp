/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#include "multi_scale_deformable_attention_grad.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace std;
using namespace AscendC;

namespace optiling
{
    uint64_t static RESERVE_SAPCE = 1024;

    static ge::graphStatus TilingFuncForMultiScaleDeformableAttentionGrad(gert::TilingContext *context)
    {
        MultiScaleDeformableAttentionGradTilingData tiling;
        auto platform_info = context->GetPlatformInfo();
        auto ascendc_platform = platform_ascendc::PlatformAscendC(platform_info);
        static uint32_t core_num = ascendc_platform.GetCoreNumAiv();
        uint64_t total_ub_size;
        ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, total_ub_size);
        auto value_shape = context->GetInputShape(0)->GetStorageShape();
        auto sampling_loc_shape = context->GetInputShape(3)->GetStorageShape();
        auto ub_size = total_ub_size - RESERVE_SAPCE;
        auto batch_size = value_shape.GetDim(0);
        auto spatial_size = value_shape.GetDim(1);
        auto num_heads = value_shape.GetDim(2);
        auto channels = value_shape.GetDim(3);
        auto num_query = sampling_loc_shape.GetDim(1);
        auto num_levels = sampling_loc_shape.GetDim(3);
        auto num_point = sampling_loc_shape.GetDim(5);
        auto task_per_core = (batch_size * num_query - 1) / core_num + 1;
        auto core_used = (batch_size * num_query - 1) / task_per_core + 1;
        auto task_tail_core = batch_size * num_query - (core_used - 1) * task_per_core;
        tiling.set_batch_size(batch_size);
        tiling.set_spatial_size(spatial_size);
        tiling.set_num_heads(num_heads);
        tiling.set_channels(channels);
        tiling.set_num_levels(num_levels);
        tiling.set_num_query(num_query);
        tiling.set_num_point(num_point);
        tiling.set_task_per_core(task_per_core);
        tiling.set_task_tail_core(task_tail_core);
        tiling.set_core_used(core_used);
        tiling.set_ub_size(ub_size);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        context->SetBlockDim(core_used);
        size_t *current_workspace = context->GetWorkspaceSizes(1);
        current_workspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge
{
    static ge::graphStatus InferShapeForMultiScaleDeformableAttentionGrad(gert::InferShapeContext *context)
    {
        const gert::Shape *value_shape = context->GetInputShape(0);
        const gert::Shape *sampling_locations_shape = context->GetInputShape(3);

        gert::Shape *grad_value_shape = context->GetOutputShape(0);
        gert::Shape *grad_sample_loc_shape = context->GetOutputShape(1);
        gert::Shape *grad_attn_weight_shape = context->GetOutputShape(2);

        grad_value_shape->AppendDim(value_shape->GetDim(0));
        grad_value_shape->AppendDim(value_shape->GetDim(1));
        grad_value_shape->AppendDim(value_shape->GetDim(2));
        grad_value_shape->AppendDim(value_shape->GetDim(3));
        grad_sample_loc_shape->AppendDim(sampling_locations_shape->GetDim(0));
        grad_sample_loc_shape->AppendDim(sampling_locations_shape->GetDim(1));
        grad_sample_loc_shape->AppendDim(sampling_locations_shape->GetDim(2));
        grad_sample_loc_shape->AppendDim(sampling_locations_shape->GetDim(3));
        grad_sample_loc_shape->AppendDim(sampling_locations_shape->GetDim(4));
        grad_sample_loc_shape->AppendDim(sampling_locations_shape->GetDim(5));
        grad_attn_weight_shape->AppendDim(sampling_locations_shape->GetDim(0));
        grad_attn_weight_shape->AppendDim(sampling_locations_shape->GetDim(1));
        grad_attn_weight_shape->AppendDim(sampling_locations_shape->GetDim(2));
        grad_attn_weight_shape->AppendDim(sampling_locations_shape->GetDim(3));
        grad_attn_weight_shape->AppendDim(sampling_locations_shape->GetDim(5));
        return GRAPH_SUCCESS;
    }
}

namespace ops
{
    class MultiScaleDeformableAttentionGrad : public OpDef
    {
    public:
        explicit MultiScaleDeformableAttentionGrad(const char *name) : OpDef(name)
        {
            this->Input("value")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("spatial_shapes")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("level_start_index")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("sampling_loc")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("attn_weight")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("grad_output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("grad_value")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("grad_sampling_loc")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("grad_attn_weight")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->SetInferShape(ge::InferShapeForMultiScaleDeformableAttentionGrad);

            this->AICore()
                .SetTiling(optiling::TilingFuncForMultiScaleDeformableAttentionGrad);

            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(MultiScaleDeformableAttentionGrad);
}