/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "dynamic_scatter_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"


using namespace ge;
using namespace std;
using namespace AscendC;

namespace optiling {
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t SIZE_OF_B8 = 1;
constexpr uint32_t SIZE_OF_B16 = 2;
constexpr uint32_t SIZE_OF_B32 = 4;
constexpr uint32_t BIT_OF_B8 = 8;
constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;
constexpr uint32_t BYTES_PER_DATA = 20;
constexpr uint32_t TILING_KEY_COE = 100;
constexpr uint32_t RESERVED_UB_SIZE = 2 * 1024;
std::string DEFAULT_REDUCE_TYPE = "max";
static std::map<std::string, uint32_t> REDUCE_TYPE_MAP = {{"sum", 0}, {"mean", 1}, {"max", 2}};


void DynamicScatterGradTiling::CalUsedCoreNum(const uint32_t coreNumPlatform)
{
    voxelNumPerCore = (totalVoxelNum + coreNumPlatform - 1) / coreNumPlatform;
    usedCoreNum = (totalVoxelNum + voxelNumPerCore - 1) / voxelNumPerCore;
    voxelNumLastCore = totalVoxelNum - (voxelNumPerCore * (usedCoreNum - 1));
    eleNumPerCore = voxelNumPerCore * featDim;
    eleNumLastCore = voxelNumLastCore * featDim;
}

ge::graphStatus DynamicScatterGradTiling::CalTilingAligned()
{
    alignedNum = BYTE_BLOCK / SIZE_OF_B32;
    featDimAligned = (featDim + alignedNum - 1) / alignedNum * alignedNum;
    blockLen = featDimAligned / alignedNum;
    if (featDim == featDimAligned) {
        isFeatsAligned = true;
    } else {
        blockLenPad = featDim * SIZE_OF_B32;
    }
    uint32_t sizePerPoint = featDimAligned * SIZE_OF_B32;
    uint32_t availableUbSize = ubSizePlatForm - sizePerPoint - RESERVED_UB_SIZE;
    if (sizePerPoint == 0) {
        return ge::GRAPH_FAILED;
    }
    maxPointNum = availableUbSize / 2 / sizePerPoint;
    return ge::GRAPH_SUCCESS;
}

void DynamicScatterGradTiling::CalMaskTiling()
{
    uint32_t alignedMaskNum = BYTE_BLOCK / SIZE_OF_B8;
    maskDim = (featDim + BIT_OF_B8 - 1) / BIT_OF_B8;
    maskDimAligned = (maskDim + alignedMaskNum - 1) / alignedMaskNum * alignedMaskNum;
    maskNum = maskDim * totalPointNum;
    blockLenMask = maskDimAligned / alignedMaskNum;
}

ge::graphStatus DynamicScatterGradTiling::Init()
{
    if (tilingContext == nullptr || tilingContext->GetInputShape(0) == nullptr || tilingContext->GetOutputShape(0) == nullptr || tilingContext->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto voxelShape = tilingContext->GetInputShape(0)->GetStorageShape();
    totalVoxelNum = voxelShape.GetDim(DIM_INDEX0);
    featDim = voxelShape.GetDim(DIM_INDEX1);
    auto pointShape = tilingContext->GetOutputShape(0)->GetStorageShape();
    totalPointNum = pointShape.GetDim(DIM_INDEX0);
    pointGradNum = totalPointNum * featDim;

    auto platformInfo = tilingContext->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    CalUsedCoreNum(coreNum);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

    auto attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const char* reduceTypePtr = attrs->GetAttrPointer<char>(DIM_INDEX0);
    std::string reduceType(reduceTypePtr);
    if (reduceType != "sum" && reduceType != "mean" && reduceType != "max") {
        return ge::GRAPH_PARAM_INVALID;
    }
    tilingContext->SetTilingKey(TILING_KEY_COE + REDUCE_TYPE_MAP[reduceType]);

    ge::graphStatus flag = CalTilingAligned();
    if (flag == ge::GRAPH_FAILED) {
        // division by zero
        return ge::GRAPH_FAILED;
    }
    if (reduceType == "max") {
        CalMaskTiling();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicScatterGradTiling::RunKernelTiling()
{
    tilingContext->SetBlockDim(usedCoreNum);
    tilingData.set_totalPointNum(totalPointNum);
    tilingData.set_totalVoxelNum(totalVoxelNum);
    tilingData.set_featDim(featDim);
    tilingData.set_pointGradNum(pointGradNum);
    tilingData.set_alignedNum(alignedNum);
    tilingData.set_featDimAligned(featDimAligned);
    tilingData.set_voxelNumPerCore(voxelNumPerCore);
    tilingData.set_voxelNumLastCore(voxelNumLastCore);
    tilingData.set_eleNumPerCore(eleNumPerCore);
    tilingData.set_eleNumLastCore(eleNumLastCore);
    tilingData.set_maskNum(maskNum);
    tilingData.set_maskDim(maskDim);
    tilingData.set_maskDimAligned(maskDimAligned);
    tilingData.set_maxPointNum(maxPointNum);
    tilingData.set_blockLen(blockLen);
    tilingData.set_blockLenPad(blockLenPad);
    tilingData.set_blockLenMask(blockLenMask);
    tilingData.set_usedCoreNum(usedCoreNum);
    tilingData.set_isFeatsAligned(isFeatsAligned);
    if (tilingContext->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForDynamicScatterGrad(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    DynamicScatterGradTiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.RunKernelTiling();
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForDynamicScatterGrad(gert::InferShapeContext* context)
{
    const gert::Shape* featShape = context->GetInputShape(3);
    if (featShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outShape->SetDim(0, featShape->GetDim(0));
    outShape->SetDim(1, featShape->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForDynamicScatterGrad(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class DynamicScatterGrad : public OpDef {
public:
    explicit DynamicScatterGrad(const char* name) : OpDef(name)
    {
        this->Input("grad_voxel_feats")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("num_point_per_voxel")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("argsort_coor")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("compare_mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_feats")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("reduce_type").AttrType(REQUIRED).String("max");
        this->SetInferShape(ge::InferShapeForDynamicScatterGrad)
            .SetInferDataType(ge::InferDataTypeForDynamicScatterGrad);
        this->AICore().SetTiling(optiling::TilingForDynamicScatterGrad);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};

OP_ADD(DynamicScatterGrad);
} // namespace ops