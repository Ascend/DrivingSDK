/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#include "dynamic_scatter_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace ge;
using namespace std;
using namespace AscendC;

namespace optiling {
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t BYTE_REPEAT = 256;
constexpr uint32_t SIZE_OF_B8 = 1;
constexpr uint32_t SIZE_OF_B16 = 2;
constexpr uint32_t SIZE_OF_B32 = 4;
constexpr uint32_t BIT_OF_B8 = 8;
constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;
constexpr uint32_t TILING_KEY_COE = 100;
constexpr uint32_t RESERVED_UB_SIZE = 16 * 1024;
constexpr uint32_t RESERVED_ARGSORT_NUM = 1000;
static std::map<std::string, uint32_t> REDUCE_TYPE_MAP = {{"sum", 0}, {"mean", 1}, {"max", 2}};

void DynamicScatterTiling::CalUsedCoreNum()
{
    voxelNumPerCore = (totalVoxelNum + coreNum - 1) / coreNum;
    usedCoreNum = (totalVoxelNum + voxelNumPerCore - 1) / voxelNumPerCore;
    voxelNumLastCore = totalVoxelNum - (voxelNumPerCore * (usedCoreNum - 1));
    voxelFeatsNumPerCore = voxelNumPerCore * featsDim;
    voxelFeatsNumLastCore = voxelNumLastCore * featsDim;
}

ge::graphStatus DynamicScatterTiling::CalTilingAligned()
{
    alignedNum = BYTE_BLOCK / SIZE_OF_B32;
    featsDimAligned = (featsDim + alignedNum - 1) / alignedNum * alignedNum;
    blockLen = featsDimAligned / alignedNum;
    if (featsDimAligned == 0) {
        return ge::GRAPH_FAILED;
    }
    if (featsDim == featsDimAligned) {
        isFeatsAligned = true;
    } else {
        blockLenPad = featsDim * SIZE_OF_B32;
    }
    return ge::GRAPH_SUCCESS;
}

void DynamicScatterTiling::CalMaskTiling()
{
    uint32_t alignedMaskNum = BYTE_BLOCK / SIZE_OF_B8;
    maskDim = (featsDim + BIT_OF_B8 - 1) / BIT_OF_B8;
    maskDimAligned = (maskDim + alignedMaskNum - 1) / alignedMaskNum * alignedMaskNum;
    maskDimAlignedB16 = maskDimAligned / 2;
    maskNum = maskDim * totalPointNum;
    blockLenMask = maskDim * SIZE_OF_B8;
    uint32_t elePerRepeat = BYTE_REPEAT / SIZE_OF_B32;
    repeatTimes = (featsDim + elePerRepeat) / elePerRepeat;
}

void DynamicScatterTiling::CalAvailableUbTiling()
{
    uint64_t availableUbSize = ubSizePlatForm - RESERVED_UB_SIZE;
    availableUbSize -= RESERVED_ARGSORT_NUM * SIZE_OF_B32;
    availableUbSize -= maskDimAligned * SIZE_OF_B8;
    availableUbSize -= maskDimAlignedB16 * SIZE_OF_B16 * 2;
    availableUbSize -= featsDimAligned * SIZE_OF_B32;
    availablePointNum = availableUbSize / (featsDimAligned * SIZE_OF_B32);
}

ge::graphStatus DynamicScatterTiling::Init()
{
    if (tilingContext == nullptr || tilingContext->GetInputShape(0) == nullptr || tilingContext->GetOutputShape(0) == nullptr || tilingContext->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto pointFeatsShape = tilingContext->GetInputShape(0)->GetStorageShape();
    totalPointNum = pointFeatsShape.GetDim(DIM_INDEX0);
    featsDim = pointFeatsShape.GetDim(DIM_INDEX1);
    auto voxelFeatsShape = tilingContext->GetOutputShape(0)->GetStorageShape();
    totalVoxelNum = voxelFeatsShape.GetDim(DIM_INDEX0);
    pointFeatsNum = totalPointNum * featsDim;

    auto platformInfo = tilingContext->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    CalUsedCoreNum();

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
    CalAvailableUbTiling();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicScatterTiling::RunKernelTiling()
{
    tilingData.set_totalPointNum(totalPointNum);
    tilingData.set_totalVoxelNum(totalVoxelNum);
    tilingData.set_featsDim(featsDim);
    tilingData.set_pointFeatsNum(pointFeatsNum);
    tilingData.set_voxelNumPerCore(voxelNumPerCore);
    tilingData.set_voxelNumLastCore(voxelNumLastCore);
    tilingData.set_voxelFeatsNumPerCore(voxelFeatsNumPerCore);
    tilingData.set_voxelFeatsNumLastCore(voxelFeatsNumLastCore);
    tilingData.set_alignedNum(alignedNum);
    tilingData.set_featsDimAligned(featsDimAligned);
    tilingData.set_availablePointNum(availablePointNum);
    tilingData.set_maskNum(maskNum);
    tilingData.set_maskDim(maskDim);
    tilingData.set_maskDimAligned(maskDimAligned);
    tilingData.set_blockLen(blockLen);
    tilingData.set_blockLenPad(blockLenPad);
    tilingData.set_blockLenMask(blockLenMask);
    tilingData.set_repeatTimes(repeatTimes);
    tilingData.set_maskDimAlignedB16(maskDimAlignedB16);
    tilingData.set_isFeatsAligned(isFeatsAligned);
    tilingData.set_usedCoreNum(usedCoreNum);
    tilingContext->SetBlockDim(usedCoreNum);

    if (tilingContext == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    
    if (tilingContext->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForDynamicScatter(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    DynamicScatterTiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.RunKernelTiling();
}
} // namespace optiling

namespace ge {
constexpr uint32_t BIT_OF_B8 = 8;

static ge::graphStatus InferShapeForDynamicScatter(gert::InferShapeContext* context)
{
    const gert::Shape* featShape = context->GetInputShape(0);
    if (featShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* prefixSumShape = context->GetInputShape(1);
    if (prefixSumShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* voxelFeatsShape = context->GetOutputShape(0);
    if (voxelFeatsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* compareMaskShape = context->GetOutputShape(1);
    if (compareMaskShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto featsDim = featShape->GetDim(1);
    auto maskDim = (featsDim + BIT_OF_B8 - 1) / BIT_OF_B8;

    voxelFeatsShape->SetDim(0, prefixSumShape->GetDim(0));
    voxelFeatsShape->SetDim(1, featsDim);
    compareMaskShape->SetDim(0, featShape->GetDim(0));
    compareMaskShape->SetDim(1, maskDim);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForDynamicScatter(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_UINT8);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class DynamicScatter : public OpDef {
public:
    explicit DynamicScatter(const char* name) : OpDef(name)
    {
        this->Input("feats")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("prefix_sum_point_per_voxel")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("argsort_coor")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("voxel_feats")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("compare_mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("reduce_type").AttrType(REQUIRED).String("max");
        this->SetInferShape(ge::InferShapeForDynamicScatter)
            .SetInferDataType(ge::InferDataTypeForDynamicScatter);
        this->AICore().SetTiling(optiling::TilingForDynamicScatter);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};

OP_ADD(DynamicScatter);
} // namespace ops
