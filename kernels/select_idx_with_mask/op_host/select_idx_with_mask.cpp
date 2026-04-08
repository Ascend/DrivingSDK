/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#include "select_idx_with_mask_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "csrc/utils.h"

namespace {
const uint32_t POS_INPUT_POLY_LINE = 0;
const uint32_t POS_INPUT_MIN_IDX = 1;
const uint32_t POS_OUTPUT_MIN_IDX = 0;

const uint32_t POLY_LINE_BATCH_SIZE_DIM = 0;
const uint32_t POLY_LINE_NUM_POINT_DIM = 1;
const uint32_t MIN_IDX_NUM_IDX_DIM = 1;

const uint64_t UB_RESERVE_BYTES = 16 * 1024;
const uint32_t FLOAT32_BYTES = 4;
const uint32_t INT16_BYTES = 2;
const uint32_t BLOCK_BYTES = 32;
} // namespace

namespace optiling {
static ge::graphStatus TilingForSelectIdxWithMask(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto coreNum = ascendPlatformInfo.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(coreNum);

    auto polyLineTensorPtr = context->GetInputTensor(POS_INPUT_POLY_LINE);
    auto minIdxTensorPtr = context->GetInputTensor(POS_INPUT_MIN_IDX);
    if (polyLineTensorPtr == nullptr || minIdxTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto polyLineShape = polyLineTensorPtr->GetStorageShape();
    auto minIdxShape = minIdxTensorPtr->GetStorageShape();

    uint32_t batchSize = polyLineShape.GetDim(POLY_LINE_BATCH_SIZE_DIM);
    uint32_t numPoint = polyLineShape.GetDim(POLY_LINE_NUM_POINT_DIM);
    uint32_t numIdx = minIdxShape.GetDim(MIN_IDX_NUM_IDX_DIM);

    // 以batch分核，计算大核数量
    uint32_t totalTaskNum = batchSize;
    uint32_t numTaskPerCore = totalTaskNum / coreNum;
    uint32_t numTaskTail = totalTaskNum % coreNum;

    uint32_t numItemsPerBlock = BLOCK_BYTES / FLOAT32_BYTES;
    uint32_t numIdxAligned = AlignUp(numIdx, numItemsPerBlock);
    uint32_t numPointAligned = AlignUp(numPoint, numItemsPerBlock);
    uint32_t numIdxRound64 = AlignUp(numIdx, 64);

    // 计算ub中一次可以放下多少个batch
    uint64_t ubBytesTotal;
    ascendPlatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytesTotal);
    uint64_t ubBytes = ubBytesTotal - UB_RESERVE_BYTES;
    uint32_t ubBytesPerBatch = numPointAligned * 2 * FLOAT32_BYTES + numIdxAligned * 12 * FLOAT32_BYTES
                               + numIdxRound64 * 5 * FLOAT32_BYTES + numIdxRound64 * 3 * INT16_BYTES + numIdxRound64 * 2;
    uint32_t compBatchNum = (ubBytes - BLOCK_BYTES) / ubBytesPerBatch;

    SelectIdxWithMaskTilingData tilingData;
    tilingData.set_batchSize(batchSize);
    tilingData.set_numPoint(numPoint);
    tilingData.set_numIdx(numIdx);
    tilingData.set_compBatchNum(compBatchNum);
    tilingData.set_numTaskPerCore(numTaskPerCore);
    tilingData.set_numTaskTail(numTaskTail);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShapeForSelectIdxWithMask(gert::InferShapeContext* context)
{
    const gert::Shape* minIdxShape = context->GetInputShape(POS_INPUT_MIN_IDX);
    gert::Shape* outMinIdxShape = context->GetOutputShape(POS_OUTPUT_MIN_IDX);

    if (minIdxShape == nullptr || outMinIdxShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outMinIdxShape = *minIdxShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSelectIdxWithMask(gert::InferDataTypeContext* context)
{
    const auto outMinIdxDtype = context->GetInputDataType(POS_INPUT_MIN_IDX);
    context->SetOutputDataType(POS_OUTPUT_MIN_IDX, outMinIdxDtype);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class SelectIdxWithMask : public OpDef {
public:
    explicit SelectIdxWithMask(const char* name) : OpDef(name)
    {
        this->Input("poly_line")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("min_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("pt")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
         this->Input("back_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("out_min_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->SetInferShape(ge::InferShapeForSelectIdxWithMask)
            .SetInferDataType(ge::InferDataTypeForSelectIdxWithMask);
        this->AICore().SetTiling(optiling::TilingForSelectIdxWithMask);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SelectIdxWithMask);
}