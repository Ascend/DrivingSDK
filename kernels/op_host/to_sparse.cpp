#include "to_sparse_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace ge;

namespace optiling {
static uint32_t AlignUp(uint32_t x, uint32_t y)
{
    if (y == 0) {
        return x;
    }
    return (x - 1 + y) / y;
}

static ge::graphStatus TilingForToSparse(gert::TilingContext* context)
{
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    if (context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto indices_offset_shape = context->GetInputShape(0)->GetStorageShape();
    auto value_shape = context->GetInputShape(1)->GetStorageShape();

    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    uint32_t actualNum = indices_offset_shape.GetDim(0) - 1;
    uint32_t outChannels = value_shape.GetDim(1);
    uint32_t coreTask = AlignUp(actualNum, coreNum);
    uint32_t usedCoreNum = AlignUp(actualNum, coreTask);
    uint32_t lastCoreTask = 0;
    if (coreTask != 0) {
        lastCoreTask = actualNum % coreTask;
    }
    if (lastCoreTask == 0) lastCoreTask = coreTask;

    uint64_t availableUbSize;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbSize);

    uint32_t moveLen = (uint32_t)((availableUbSize - 20 * 1024) / 4 / (outChannels + 8 + 1));
    if (moveLen > coreTask) moveLen = coreTask;
    uint32_t repeatTimes = AlignUp(coreTask, moveLen);
    uint32_t lastRepeatTimes = AlignUp(lastCoreTask, moveLen);
    uint32_t moveTail = 0;
    uint32_t lastMoveTail = 0;
    if (moveLen != 0) {
        moveTail = coreTask % moveLen;
        lastMoveTail = lastCoreTask % moveLen;
    }
    if (moveTail == 0) moveTail = moveLen;
    if (lastMoveTail == 0) lastMoveTail = moveLen;

    ToSparseTilingData tiling;
    context->SetBlockDim(usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_coreTask(coreTask);
    tiling.set_lastCoreTask(lastCoreTask);
    tiling.set_moveLen(moveLen);
    tiling.set_repeatTimes(repeatTimes);
    tiling.set_moveTail(moveTail);
    tiling.set_lastRepeatTimes(lastRepeatTimes);
    tiling.set_lastMoveTail(lastMoveTail);
    tiling.set_outChannels(outChannels);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForToSparse(gert::InferShapeContext* context)
{
    auto indicesOffsetShape = context->GetInputShape(0);
    auto valueShape = context->GetInputShape(1);
    if (indicesOffsetShape == nullptr || valueShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* sparseValueShape = context->GetOutputShape(0);
    gert::Shape* sparseIndicesShape = context->GetOutputShape(1);
    if (sparseValueShape == nullptr || sparseIndicesShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint64_t actualNum = indicesOffsetShape->GetDim(0) - 1;
    *sparseValueShape = {actualNum, valueShape->GetDim(1)};
    *sparseIndicesShape = {actualNum, 8};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeForToSparse(gert::InferDataTypeContext* context)
{
    const ge::DataType indices_dtype = context->GetInputDataType(0);
    const ge::DataType feature_dtype = context->GetInputDataType(1);
    context->SetOutputDataType(0, feature_dtype);
    context->SetOutputDataType(1, indices_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ToSparse : public OpDef {
public:
    explicit ToSparse(const char* name) : OpDef(name)
    {
        this->Input("indices_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("former_sorted_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("sparse_value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("sparse_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForToSparse).SetInferDataType(ge::InferDtypeForToSparse);

        this->AICore().SetTiling(optiling::TilingForToSparse);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(ToSparse);
}