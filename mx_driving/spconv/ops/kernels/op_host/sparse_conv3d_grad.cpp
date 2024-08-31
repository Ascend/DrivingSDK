#include "sparse_conv3d_grad_tiling.h"
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

static ge::graphStatus TilingForSparseConv3dGrad(gert::TilingContext* context)
{
    SparseConv3dGradTilingData tiling;
    auto indices_offset_shape = context->GetInputShape(0)->GetStorageShape();
    auto weight_shape = context->GetInputShape(3)->GetStorageShape();
    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    uint32_t coreNum = ascendplatformInfo.GetCoreNumAiv();
    uint32_t actualNum = indices_offset_shape.GetDim(0) - 1;
    uint32_t kernelSize = weight_shape.GetDim(0) * weight_shape.GetDim(1) * weight_shape.GetDim(2);
    uint32_t kernelIC = weight_shape.GetDim(3);
    uint32_t kernelOC = weight_shape.GetDim(4);
    uint32_t coreTask = AlignUp(actualNum, coreNum);
    uint32_t usedCoreNum = AlignUp(actualNum, coreTask);
    uint32_t lastCoreTask = 0;
    uint32_t kernelSizeAlign = AlignUp(kernelSize, 8) * 8;
    if (coreTask != 0) {
        lastCoreTask = actualNum % coreTask;
    }
    if (lastCoreTask == 0) lastCoreTask = coreTask;

    uint64_t availableUbSize;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbSize);
    // indicesOffsetUb [moveLen]
    // SortindicesOffsetUb [moveLen, kernelSizeAlign]
    // gradUb [kernelOC]
    // featureUb and reduceSumTmpUB [kernelIC] * 2
    // WeightUB and WeightTmpUB [kernelIC, kernelOC] * 2
    uint32_t tmpUsedUbSize = (kernelIC * kernelOC * 2 + kernelIC * 2) * sizeof(float);
    uint32_t reserveUbSize = 8 * 1024;
    uint32_t moveLen = (uint32_t)((availableUbSize - tmpUsedUbSize - reserveUbSize) / 4 / (1 + kernelSizeAlign));
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

    context->SetBlockDim(usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_coreTask(coreTask);
    tiling.set_lastCoreTask(lastCoreTask);
    tiling.set_moveLen(moveLen);
    tiling.set_repeatTimes(repeatTimes);
    tiling.set_moveTail(moveTail);
    tiling.set_lastRepeatTimes(lastRepeatTimes);
    tiling.set_lastMoveTail(lastMoveTail);
    tiling.set_kernelSize(kernelSize);
    tiling.set_kernelIC(kernelIC);
    tiling.set_kernelOC(kernelOC);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());

    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForSparseConv3dGrad(gert::InferShapeContext* context)
{
    const gert::Shape* featureShape = context->GetInputShape(2);
    const gert::Shape* weightShape = context->GetInputShape(3);
    if (featureShape == nullptr || weightShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* featureGradShape = context->GetOutputShape(0);
    gert::Shape* weightGradShape = context->GetOutputShape(1);
    if (featureGradShape == nullptr || weightGradShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *featureGradShape = *featureShape;
    *weightGradShape = *weightShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeForSparseConv3dGrad(gert::InferDataTypeContext* context)
{
    const ge::DataType indices_dtype = context->GetInputDataType(1);
    const ge::DataType feature_dtype = context->GetInputDataType(2);
    context->SetOutputDataType(0, feature_dtype);
    context->SetOutputDataType(1, indices_dtype);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class SparseConv3dGrad : public OpDef {
public:
    explicit SparseConv3dGrad(const char* name) : OpDef(name)
    {
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
        this->Input("feature")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("feature_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("weight_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForSparseConv3dGrad).SetInferDataType(ge::InferDtypeForSparseConv3dGrad);
        this->AICore().SetTiling(optiling::TilingForSparseConv3dGrad);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910c");
    }
};

OP_ADD(SparseConv3dGrad);
}