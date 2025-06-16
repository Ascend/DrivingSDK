/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 *
 */
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "unique_tiling.h"
constexpr size_t SYS_RSVD_WS_SIZE = 16 * 1024 * 1024;
constexpr size_t BYTE_PER_BLK = 32;
constexpr size_t EVENTID_MAX = 8;

namespace optiling {
static ge::graphStatus UniqueTilingFunc(gert::TilingContext* context)
{
    if (!context) {
        return ge::GRAPH_FAILED;
    }
    UniqueTilingData tiling;

    constexpr uint16_t tileLength = 8192;
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    if (!inputShape) {
        return ge::GRAPH_FAILED;
    }
    const uint8_t dimNum = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint32_t totalLength = 1;
    for (int i = 0; i < dimNum; i++) {
        totalLength *= inputShape->GetStorageShape().GetDim(i);
    }
    const uint32_t tileNum = (totalLength + tileLength - 1) / tileLength;
    const uint16_t tailLength = totalLength % tileLength;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    const uint8_t blockNum = tileNum >= aivNum ? aivNum : tileNum;
    const uint32_t shortBlockTileNum = tileNum / blockNum;
    const uint8_t longBlockNum = tileNum % blockNum;
    const uint8_t shortBlockNum = blockNum - longBlockNum;

    tiling.set_totalLength(totalLength);
    tiling.set_shortBlockTileNum(shortBlockTileNum);
    tiling.set_tileLength(tileLength);
    tiling.set_tailLength(tailLength);
    tiling.set_aivNum(aivNum);
    tiling.set_blockNum(blockNum);
    tiling.set_shortBlockNum(shortBlockNum);

    context->SetBlockDim(blockNum);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    // Workspace for IBSet/IBWait up to 8 times, and 2 times full data.
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    auto&& currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    size_t usrSize = (blockNum * BYTE_PER_BLK * EVENTID_MAX + aivNum * BYTE_PER_BLK + BYTE_PER_BLK) +
                     (blockNum + BYTE_PER_BLK - 1) / BYTE_PER_BLK * BYTE_PER_BLK +
                     (tileNum * tileLength) * 2 * sizeof(float) * 2;
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


namespace ge {
static ge::graphStatus UniqueInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (!x1_shape || !y_shape) {
        return GRAPH_FAILED;
    }
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus UniqueInferDtype(gert::InferDataTypeContext* context)
{
    auto inputDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDtype);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge


namespace ops {
class Unique : public OpDef {
public:
    explicit Unique(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("uniqueCnt")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::UniqueInferShape);
        this->SetInferDataType(ge::UniqueInferDtype);

        this->AICore().SetTiling(optiling::UniqueTilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(Unique);
} // namespace ops