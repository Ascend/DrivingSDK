/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "gather_nms3d_mask_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint32_t BLOCK_DIM = 1;
static ge::graphStatus GatherNms3dMaskTiling(gert::TilingContext *context)
{
    GatherNms3dMaskTilingData tiling;
    auto const maskShape = context->GetInputShape(0);
    if (maskShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto const maskShapeVal = maskShape->GetStorageShape();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_box_num(maskShapeVal.GetDim(0));
    tiling.set_mask_num(maskShapeVal.GetDim(1));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus GatherNms3dMaskInferShape(gert::InferShapeContext* context)
{
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForGatherNms3dMask(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT16);
    context->SetOutputDataType(1, ge::DT_INT16);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class GatherNms3dMask : public OpDef {
public:
    explicit GatherNms3dMask(const char *name) : OpDef(name)
    {
        this->Input("mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("keep")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("num_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::GatherNms3dMaskInferShape)
            .SetInferDataType(ge::InferDataTypeForGatherNms3dMask);

        this->AICore()
            .SetTiling(optiling::GatherNms3dMaskTiling);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GatherNms3dMask);
}
