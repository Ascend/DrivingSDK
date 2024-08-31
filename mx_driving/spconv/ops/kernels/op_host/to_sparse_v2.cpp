
#include "to_sparse_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace ge;
using namespace std;
namespace optiling {
static uint32_t AlignUp(uint32_t x, uint32_t y)
{
    if (y == 0) {
        return x;
    }
    return (x - 1 + y) / y;
}

static ge::graphStatus TilingForToSparseV2(gert::TilingContext* context)
{
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto indices_offset_shape = context->GetInputShape(2)->GetStorageShape();
    auto weight_shape = context->GetInputShape(1)->GetStorageShape();

    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    uint32_t actualNum = indices_offset_shape.GetDim(0) - 1;
    uint32_t kernelD = weight_shape.GetDim(0);
    uint32_t kernelH = weight_shape.GetDim(1);
    uint32_t kernelW = weight_shape.GetDim(2);
    uint32_t kernelOC = weight_shape.GetDim(3);
    uint32_t kernelIC = weight_shape.GetDim(4);
    uint32_t kernelICAlign =  AlignUp(kernelIC, 8) * 8;
    uint32_t kernelSize = kernelD * kernelH * kernelW;
    uint32_t coreTask = AlignUp(actualNum, coreNum);
    uint32_t usedCoreNum = AlignUp(actualNum, coreTask);
    uint32_t lastCoreTask = 0;
    if (coreTask != 0) {
        lastCoreTask = actualNum % coreTask;
    }
    if (lastCoreTask == 0) lastCoreTask = coreTask;

    uint64_t availableUbSize;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbSize);
    uint32_t weightUbSize = 128 * 128 * sizeof(float);
    uint32_t kernelOneLen = 128 * 128 / kernelICAlign / kernelOC;
    uint32_t kernelRepeateTimes = AlignUp(kernelSize, kernelOneLen);
    uint32_t kernelLastLen = kernelSize % kernelOneLen;
    if (kernelLastLen == 0) kernelLastLen = kernelOneLen;
    uint32_t moveLen = (uint32_t)((availableUbSize - 8 * 1024 - weightUbSize * 2) / 4 / (kernelOC + 8 + 1 + AlignUp(kernelSize, 8) * 8));
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

    ToSparseV2TilingData tiling;
    context->SetBlockDim(usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_coreTask(coreTask);
    tiling.set_lastCoreTask(lastCoreTask);
    tiling.set_moveLen(moveLen);
    tiling.set_repeatTimes(repeatTimes);
    tiling.set_moveTail(moveTail);
    tiling.set_lastRepeatTimes(lastRepeatTimes);
    tiling.set_lastMoveTail(lastMoveTail);
    tiling.set_kernelIC(kernelIC);
    tiling.set_kernelOC(kernelOC);
    tiling.set_kernelSize(kernelSize);
    tiling.set_kernelOneLen(kernelOneLen);
    tiling.set_kernelRepeateTimes(kernelRepeateTimes);
    tiling.set_kernelLastLen(kernelLastLen);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForToSparseV2(gert::InferShapeContext* context)
{
    auto weightShape = context->GetInputShape(1);
    auto indicesOffsetShape = context->GetInputShape(2);
    if (indicesOffsetShape == nullptr || weightShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* sparseValueShape = context->GetOutputShape(0);
    gert::Shape* sparseIndicesShape = context->GetOutputShape(1);
    if (sparseValueShape == nullptr || sparseIndicesShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint64_t actualNum = indicesOffsetShape->GetDim(0) - 1;
    *sparseValueShape = {actualNum, weightShape->GetDim(3)};
    *sparseIndicesShape = {actualNum, 8};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeForToSparseV2(gert::InferDataTypeContext* context)
{
    const ge::DataType feature_dtype = context->GetInputDataType(0);
    const ge::DataType indices_dtype = context->GetInputDataType(2);
    context->SetOutputDataType(0, feature_dtype);
    context->SetOutputDataType(1, indices_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ToSparseV2 : public OpDef {
public:
    explicit ToSparseV2(const char* name) : OpDef(name)
    {
        this->Input("features")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indices_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
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

        this->SetInferShape(ge::InferShapeForToSparseV2).SetInferDataType(ge::InferDtypeForToSparseV2);

        this->AICore().SetTiling(optiling::TilingForToSparseV2);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910c");
    }
};

OP_ADD(ToSparseV2);
}
