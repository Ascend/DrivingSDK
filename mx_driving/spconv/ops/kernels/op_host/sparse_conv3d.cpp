
#include "sparse_conv3d_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace ge;

namespace optiling {
static uint32_t Ceil(uint32_t x, uint32_t y)
{
    if (y == 0) {
        return x;
    }
    return (x - 1 + y) / y;
}
static uint32_t AlignUp(uint32_t x, uint32_t y)
{
    if (y == 0) {
        return x;
    }
    return ((x - 1 + y) / y) * y;
}

static ge::graphStatus TilingForSparseConv3d(gert::TilingContext* context)
{
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    auto feature_shape = context->GetInputShape(0)->GetStorageShape();
    auto weight_shape = context->GetInputShape(2)->GetStorageShape();

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    uint32_t actualNum = feature_shape.GetDim(0);

    uint32_t coreTask = AlignUp(Ceil(actualNum, coreNum), 32);
    uint32_t usedCoreNum = Ceil(actualNum, coreTask);
    uint32_t lastCoreTask = 0;
    if (coreTask != 0) {
        lastCoreTask = actualNum % coreTask;
    }
    if (lastCoreTask == 0) lastCoreTask = coreTask;
    uint64_t availableUbSize;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbSize);

    uint32_t kernelD = weight_shape.GetDim(0);
    uint32_t kernelH = weight_shape.GetDim(1);
    uint32_t kernelW = weight_shape.GetDim(2);
    uint32_t kernelOC = weight_shape.GetDim(3);
    uint32_t kernelIC = weight_shape.GetDim(4);
    uint32_t kernelSize = kernelD * kernelH * kernelW;

    uint32_t reserveUbSize = 8 * 1024;
    uint32_t usedUbSize = 256 * 4 * sizeof(float);
    uint32_t weightUbSize = kernelOC * AlignUp(kernelIC, 8) * sizeof(float);
    // featureUb [moveLen, icAlignUp], indicesUb [moveLen, 4], outidxUb [moveLen, 27] outidxPairUb [moveLen, kernelSize, 4]
    // The max Value of ic and oc is 256 -> mulUb + sumUb + tmpUb + workUb = 256 * 4 * sizeof(int32 or float32)
    // weight shape is [..., oc, ic] ->  weightUb = oc * ic
    uint32_t moveLen = (availableUbSize - usedUbSize - weightUbSize - reserveUbSize) / 4 / (AlignUp(kernelIC, 8) + 4 + 5 * kernelSize);
    moveLen = moveLen / 32 * 32;
    if (moveLen > coreTask) moveLen = coreTask;

    uint32_t repeatTimes = Ceil(coreTask, moveLen);
    uint32_t lastRepeatTimes = Ceil(lastCoreTask, moveLen);
    uint32_t moveTail = 0;
    uint32_t lastMoveTail = 0;
    if (moveLen != 0) {
        moveTail = coreTask % moveLen;
        lastMoveTail = lastCoreTask % moveLen;
    }
    if (moveTail == 0) moveTail = moveLen;
    if (lastMoveTail == 0) lastMoveTail = moveLen;

    auto outSpatialShapePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(0);
    auto stridePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(1);
    auto paddingPtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(2);
    auto outSpatialShapeData = reinterpret_cast<const int64_t*>(outSpatialShapePtr->GetData());
    auto strideData = reinterpret_cast<const int64_t*>(stridePtr->GetData());
    auto paddingData = reinterpret_cast<const int64_t*>(paddingPtr->GetData());

    SparseConv3dTilingData tiling;
    context->SetBlockDim(usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_coreTask(coreTask);
    tiling.set_lastCoreTask(lastCoreTask);
    tiling.set_moveLen(moveLen);
    tiling.set_repeatTimes(repeatTimes);
    tiling.set_moveTail(moveTail);
    tiling.set_lastRepeatTimes(lastRepeatTimes);
    tiling.set_lastMoveTail(lastMoveTail);
    tiling.set_kernelD(kernelD);
    tiling.set_kernelH(kernelH);
    tiling.set_kernelW(kernelW);
    tiling.set_kernelIC(kernelIC);
    tiling.set_kernelOC(kernelOC);
    tiling.set_kernelSize(kernelSize);
    tiling.set_outfeatureB(outSpatialShapeData[0]);
    tiling.set_outputDepth(outSpatialShapeData[1]);
    tiling.set_outputHeight(outSpatialShapeData[2]);
    tiling.set_outputWidth(outSpatialShapeData[3]);
    tiling.set_strideDepth(strideData[0]);
    tiling.set_strideHeight(strideData[1]);
    tiling.set_strideWidth(strideData[2]);
    tiling.set_paddingDepth(paddingData[0]);
    tiling.set_paddingHeight(paddingData[1]);
    tiling.set_paddingWidth(paddingData[2]);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 16 * 1024 * 1024;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForSparseConv3d(gert::InferShapeContext* context)
{
    const gert::Shape* featureShape = context->GetInputShape(0);
    const gert::Shape* indicesShape = context->GetInputShape(1);
    const gert::Shape* weightShape = context->GetInputShape(2);
    if (featureShape == nullptr || indicesShape == nullptr || weightShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outShape = context->GetOutputShape(0);
    gert::Shape* indicesOutShape = context->GetOutputShape(1);
    gert::Shape* indicesPairShape = context->GetOutputShape(2);
    if (outShape == nullptr || indicesOutShape == nullptr || indicesPairShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // weightShape[kernelH, kernelW, kernelD, outChannels, inChannels]
    uint64_t kernelSize = 1;
    for (size_t i = 0; i < weightShape->GetDimNum() - 2; i++) {
        kernelSize *= weightShape->GetDim(i);
    }
    uint64_t kernelOC = weightShape->GetDim(3);
    uint64_t indicesSecondSize = indicesShape->GetDim(1);
    *outShape = {kernelSize, kernelOC};
    *indicesOutShape = {kernelSize};
    *indicesPairShape = {kernelSize, indicesSecondSize};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeForSparseConv3d(gert::InferDataTypeContext* context)
{
    const ge::DataType feature_dtype = context->GetInputDataType(0);
    const ge::DataType indices_dtype = context->GetInputDataType(1);
    context->SetOutputDataType(0, feature_dtype);
    context->SetOutputDataType(1, indices_dtype);
    context->SetOutputDataType(2, indices_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class SparseConv3d : public OpDef {
public:
    explicit SparseConv3d(const char* name) : OpDef(name)
    {
        this->Input("features")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("feature_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("indices_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("indices_pair")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("out_spatial_shape").ListInt();
        this->Attr("stride").ListInt();
        this->Attr("padding").ListInt();

        this->SetInferShape(ge::InferShapeForSparseConv3d).SetInferDataType(ge::InferDtypeForSparseConv3d);
        this->AICore().SetTiling(optiling::TilingForSparseConv3d);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910c");
    }
};

OP_ADD(SparseConv3d);
}