/*
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
*/
#include "pixel_group_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "ge/utils.h"

using namespace ge;
using namespace std;

namespace {
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t SIZE_OF_FP32 = sizeof(float);
constexpr uint32_t MEMORY_DIVIDED = 200;
} // namespace

namespace optiling {
static ge::graphStatus TilingFuncForPixelGroup(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    PixelGroupTilingData tiling;

    const gert::StorageShape *scoreShape = context->GetInputShape(0);
    const gert::StorageShape *embeddingShape = context->GetInputShape(2);
    if (scoreShape == nullptr || embeddingShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t totalPixels = context->GetInputTensor(0)->GetShapeSize();
    uint32_t height = scoreShape->GetStorageShape().GetDim(0);
    uint32_t width = scoreShape->GetStorageShape().GetDim(1);
    uint32_t embeddingDim = embeddingShape->GetStorageShape().GetDim(2);
    uint32_t dataAlign = BYTE_BLOCK / SIZE_OF_FP32;
    uint32_t dimAlign = Ceil(embeddingDim, dataAlign) * dataAlign;

    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    auto coreNum = ascendplatformInfo.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto kernelRegionNumPtr = attrs->GetAttrPointer<int32_t>(0);
    auto distanceThresholdPtr = attrs->GetAttrPointer<float>(1);
    if ((kernelRegionNumPtr == nullptr) || (distanceThresholdPtr == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    int32_t kernelRegionNum = *kernelRegionNumPtr;
    float distanceThreshold = *distanceThresholdPtr;
    uint32_t averagePixels = totalPixels / coreNum;
    uint32_t pixelLast = totalPixels % coreNum;
    uint64_t availableUbSize;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbSize);
    availableUbSize = (availableUbSize - 20*1024) / SIZE_OF_FP32 / MEMORY_DIVIDED;
    availableUbSize = Ceil(availableUbSize, dataAlign) * dataAlign;
    uint32_t usedCoreNum = coreNum;
    if (averagePixels == 0) {
        usedCoreNum = pixelLast;
    }
    context->SetBlockDim(usedCoreNum);

    tiling.set_core_used(usedCoreNum);
    tiling.set_total_pixels(totalPixels);
    tiling.set_average_pixels(averagePixels);
    tiling.set_pixel_last(pixelLast);
    tiling.set_embedding_dim(embeddingDim);
    tiling.set_dim_align(dimAlign);
    tiling.set_kernel_region_num(kernelRegionNum);
    tiling.set_distance_threshold(distanceThreshold);
    tiling.set_available_ub_size(availableUbSize);
    tiling.set_loop_time_front((averagePixels + 1) / availableUbSize);
    tiling.set_last_loop_front((averagePixels + 1) % availableUbSize);
    tiling.set_loop_time_rear(averagePixels / availableUbSize);
    tiling.set_last_loop_rear(averagePixels % availableUbSize);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t usrSize = (kernelRegionNum * embeddingDim * 2) * SIZE_OF_FP32;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendplatformInfo.GetLibApiWorkSpaceSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShapeForPixelGroup(gert::InferShapeContext *context)
{
    const gert::Shape *labelShape = context->GetInputShape(3);
    if (labelShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto kernelRegionNumPtr = attrs->GetAttrPointer<int32_t>(0);
    if (kernelRegionNumPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int32_t kernelRegionNum = *kernelRegionNumPtr;
    if (context->GetOutputShape(0) == nullptr || context->GetOutputShape(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape *pointVectorShape = context->GetOutputShape(0);
    gert::Shape *labelUpdatedShape = context->GetOutputShape(1);
    *pointVectorShape = {kernelRegionNum, 2};
    *labelUpdatedShape = *labelShape;

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForPixelGroup(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_INT32);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class PixelGroup : public OpDef {
public:
    explicit PixelGroup(const char *name) : OpDef(name)
    {
        this->Input("score")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).AutoContiguous()
        .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mask")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BOOL})
        .Format({ge::FORMAT_ND}).AutoContiguous()
        .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("embedding")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).AutoContiguous()
        .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("kernel_label")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND}).AutoContiguous()
        .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("kernel_contour")
        .ParamType(REQUIRED)
        .DataType({ge::DT_UINT8})
        .Format({ge::FORMAT_ND}).AutoContiguous()
        .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("point_vector")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("label_updated")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("kernel_region_num").Int();
        this->Attr("distance_threshold").Float();

        this->SetInferShape(ge::InferShapeForPixelGroup)
            .SetInferDataType(ge::InferDataTypeForPixelGroup);

        this->AICore()
            .SetTiling(optiling::TilingFuncForPixelGroup);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(PixelGroup);
} // namespace ops