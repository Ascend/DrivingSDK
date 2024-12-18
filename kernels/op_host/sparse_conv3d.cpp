#include "ge/utils.h"
#include "sparse_conv3d_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace ge;
namespace optiling {
static ge::graphStatus TilingForSparseConv3d(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr || context->GetInputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto indices_shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    uint32_t actualNum = indices_shape.GetDim(0);

    uint32_t coreTask = AlignUp(Ceil(actualNum, coreNum), 32);
    uint32_t usedCoreNum = Ceil(actualNum, coreTask);
    uint32_t lastCoreTask = 0;
    if (coreTask != 0) {
        lastCoreTask = actualNum % coreTask;
    }
    if (lastCoreTask == 0) lastCoreTask = coreTask;
    uint64_t availableUbSize;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbSize);
    auto kernelSizePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(0);
    auto kernelSizeData = reinterpret_cast<const int64_t*>(kernelSizePtr->GetData());

    uint32_t kernelD = kernelSizeData[0];
    uint32_t kernelH = kernelSizeData[1];
    uint32_t kernelW = kernelSizeData[2];
    uint32_t kernelSize = kernelD * kernelH * kernelW;

    uint32_t reserveUbSize = 8 * 1024;
    uint32_t moveLen = (availableUbSize - reserveUbSize) / 4 / (kernelSize * 5 + 4);
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

    auto outSpatialShapePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(1);
    auto stridePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(2);
    auto paddingPtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(3);
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
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
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
    const gert::Shape* indicesShape = context->GetInputShape(0);

    if (indicesShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* indicesOutShape = context->GetOutputShape(0);
    gert::Shape* indicesPairShape = context->GetOutputShape(1);
    if (indicesOutShape == nullptr || indicesPairShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t kernelSize = 27;
    uint64_t indicesSecondSize = indicesShape->GetDim(1);

    *indicesOutShape = {indicesSecondSize};
    *indicesPairShape = {kernelSize, indicesSecondSize};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeForSparseConv3d(gert::InferDataTypeContext* context)
{
    const ge::DataType indices_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, indices_dtype);
    context->SetOutputDataType(1, indices_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class SparseConv3d : public OpDef {
public:
    explicit SparseConv3d(const char* name) : OpDef(name)
    {
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

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

        this->Attr("kernel_size").ListInt();
        this->Attr("out_spatial_shape").ListInt();
        this->Attr("stride").ListInt();
        this->Attr("padding").ListInt();

        this->SetInferShape(ge::InferShapeForSparseConv3d).SetInferDataType(ge::InferDtypeForSparseConv3d);
        this->AICore().SetTiling(optiling::TilingForSparseConv3d);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SparseConv3d);
}
