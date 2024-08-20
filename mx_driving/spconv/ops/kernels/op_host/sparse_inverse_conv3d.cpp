
#include "sparse_inverse_conv3d_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace ge;

namespace optiling {
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t DTYPE_FP32_BLOCK = 8;
constexpr uint32_t RESERVED_UB_SIZE = 16 * 1024;
constexpr uint32_t OTHER_UB_NUMBER = 256;

void SparseInverseConv3dTiling::CalUsedCoreNumAndCoreTask()
{
    coreTask = ((actualNum + coreNum - 1) / coreNum / BYTE_BLOCK) * BYTE_BLOCK;
    usedCoreNum = (actualNum + coreTask - 1) / coreTask;
    lastCoreTask = (actualNum % coreTask == 0) ? coreTask : actualNum % coreTask;
}

void SparseInverseConv3dTiling::CalAvailableUbTiling()
{
    // featureUb [moveLen, icAlignUp], indicesUb [moveLen, 4], outidxUb [moveLen, 27] outidxPairUb [moveLen, kernelSize, 4]
    // otherUB The max Value of ic and oc is 256 -> mulUb + sumUb + tmpUb + workUb = 256 * 4 * sizeof(int32 or float32)
    // weight shape is [..., oc, ic] ->  weightUb = oc * ic
    uint64_t availableUbSize = ubSizePlatForm - RESERVED_UB_SIZE;
    uint64_t ubAvailableNumber = availableUbSize / sizeof(float);
    ubAvailableNumber -= OTHER_UB_NUMBER * 4;
    ubAvailableNumber -= kernelIC * (kernelIC - DTYPE_FP32_BLOCK + 1) / DTYPE_FP32_BLOCK ;
    uint32_t partNum = (kernelIC - DTYPE_FP32_BLOCK + 1) / DTYPE_FP32_BLOCK + 4 + 5 * kernelSize;
    moveLen = ubAvailableNumber / partNum;
    moveLen = moveLen / BYTE_BLOCK * BYTE_BLOCK;
}

void SparseInverseConv3dTiling::GetIntArrayList()
{
    auto attrsPtr = tilingContext->GetAttrs();
    auto outSpatialShapePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(0);
    auto stridePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(1);
    auto paddingPtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(2);
    auto dilationPtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(3);
    auto outputPaddingPtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(4);
    auto outSpatialShapeData = reinterpret_cast<const int64_t*>(outSpatialShapePtr->GetData());
    auto strideData = reinterpret_cast<const int64_t*>(stridePtr->GetData());
    auto paddingData = reinterpret_cast<const int64_t*>(paddingPtr->GetData());
    auto dilationData = reinterpret_cast<const int64_t*>(paddingPtr->GetData());
    auto outputPaddingData = reinterpret_cast<const int64_t*>(paddingPtr->GetData());
    tilingData.set_outfeatureB(outSpatialShapeData[0]);
    tilingData.set_outputDepth(outSpatialShapeData[1]);
    tilingData.set_outputHeight(outSpatialShapeData[2]);
    tilingData.set_outputWidth(outSpatialShapeData[3]);
    tilingData.set_strideDepth(strideData[0]);
    tilingData.set_strideHeight(strideData[1]);
    tilingData.set_strideWidth(strideData[2]);
    tilingData.set_paddingDepth(paddingData[0]);
    tilingData.set_paddingHeight(paddingData[1]);
    tilingData.set_paddingWidth(paddingData[2]);
    tilingData.set_dilationDepth(dilationData[0]);
    tilingData.set_dilationHeight(dilationData[1]);
    tilingData.set_dilationWidth(dilationData[2]);
    tilingData.set_outputPaddingDepth(outputPaddingData[0]);
    tilingData.set_outputPaddingHeight(outputPaddingData[1]);
    tilingData.set_outputPaddingWidth(outputPaddingData[2]);
}

ge::graphStatus SparseInverseConv3dTiling::Init()
{
    auto platformInfo = tilingContext->GetPlatformInfo();
    auto attrsPtr = tilingContext->GetAttrs();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto feature_shape = tilingContext->GetInputShape(0)->GetStorageShape();
    auto weight_shape = tilingContext->GetInputShape(2)->GetStorageShape();
    actualNum = feature_shape.GetDim(0);

    CalUsedCoreNumAndCoreTask();

    kernelD = weight_shape.GetDim(0);
    kernelH = weight_shape.GetDim(1);
    kernelW = weight_shape.GetDim(2);
    kernelOC = weight_shape.GetDim(3);
    kernelIC = weight_shape.GetDim(4);
    kernelSize = kernelD * kernelH * kernelW;

    CalAvailableUbTiling();

    GetIntArrayList();
}

ge::graphStatus SparseInverseConv3dTiling::RunKernelTiling()
{
    tilingContext->SetBlockDim(usedCoreNum);
    tilingData.set_usedCoreNum(usedCoreNum);
    tilingData.set_coreTask(coreTask);
    tilingData.set_lastCoreTask(lastCoreTask);
    tilingData.set_moveLen(moveLen);
    tilingData.set_repeatTimes(repeatTimes);
    tilingData.set_moveTail(moveTail);
    tilingData.set_lastRepeatTimes(lastRepeatTimes);
    tilingData.set_lastMoveTail(lastMoveTail);
    tilingData.set_kernelD(kernelD);
    tilingData.set_kernelH(kernelH);
    tilingData.set_kernelW(kernelW);
    tilingData.set_kernelIC(kernelIC);
    tilingData.set_kernelOC(kernelOC);
    tilingData.set_kernelSize(kernelSize);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForSparseInverseConv3d(gert::TilingContext* context)
{
    SparseInverseConv3dTiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.RunKernelTiling();
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
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
}

namespace ops {
class SparseInverseConv3d : public OpDef {
public:
    explicit SparseInverseConv3d(const char* name) : OpDef(name)
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
        this->Attr("dilation").ListInt();
        this->Attr("output_padding").ListInt();

        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingForSparseInverseConv3d);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910c");
    }
};

OP_ADD(SparseInverseConv3d);
}