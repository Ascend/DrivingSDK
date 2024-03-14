/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
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
constexpr uint32_t SIZE_OF_FP16 = 2;
constexpr uint32_t SIZE_OF_FP32 = 4;
constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;
constexpr uint32_t BYTES_PER_DATA = 20;
constexpr int KEY_FP16 = 0;
constexpr int KEY_FP32 = 1;
static std::map<std::string, uint32_t> REDUCE_TYPE_MAP = {{"max", 0}, {"sum", 1}};
class DynamicScatterTiling {
public:
    explicit DynamicScatterTiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();

private:
    void SetTilingKeyMode(ge::DataType dType, uint32_t reduceTypeNum) const;
    uint32_t GetNeedCoreNum(const uint32_t coreNumPlatform) const;
    void CalTilingAligned(ge::DataType dType);

private:
    DynamicScatterTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    uint32_t pointNum;
    uint32_t featsNum;
    uint32_t coreNum;
    uint32_t totalLength = 1; // the length of input
    uint32_t formerNum; // deal more data core num
    uint32_t tailNum; // deal less data core num
    uint32_t formerLength; // deal more data length
    uint32_t tailLength; // deal less data length
    uint32_t alignNum; // data count per block
    uint32_t totalLengthAligned; // length to align 32B
    uint32_t outPointNum;
    uint32_t outPointNumAligned;
    uint32_t featsAligned;
    uint32_t formerInputNum;
    uint32_t tailInputNum;
    uint32_t tileLength;
    uint64_t ubSizePlatForm;
};

void DynamicScatterTiling::SetTilingKeyMode(ge::DataType dType, uint32_t reduceTypeNum) const
{
    switch (dType) {
        case ge::DT_FLOAT:
            tilingContext->SetTilingKey(KEY_FP32 * 100 + reduceTypeNum);
            break;
        case ge::DT_FLOAT16:
            tilingContext->SetTilingKey(KEY_FP16 * 100 + reduceTypeNum);
            break;
        default:
            tilingContext->SetTilingKey(100);
            break;
    }
}

uint32_t DynamicScatterTiling::GetNeedCoreNum(const uint32_t coreNumPlatform) const
{
    uint32_t tempCoreNum = pointNum;
    if (tempCoreNum == 0) {
        tempCoreNum = 1;
    }
    if (tempCoreNum < coreNumPlatform) {
        return tempCoreNum;
    } else {
        return coreNumPlatform;
    }
}

void DynamicScatterTiling::CalTilingAligned(ge::DataType dType)
{
    alignNum = BYTE_BLOCK / SIZE_OF_FP32;
    if (dType == ge::DT_FLOAT16) {
        alignNum = BYTE_BLOCK / SIZE_OF_FP16;
    }
    tileLength = ubSizePlatForm / BYTES_PER_DATA;
    tileLength = tileLength / (featsNum * alignNum) * (featsNum * alignNum);
    featsAligned = (featsNum + alignNum - 1) / alignNum * alignNum;
    tailInputNum = pointNum / coreNum;
    formerNum = pointNum % coreNum;
    tailNum = coreNum - formerNum;
    formerInputNum = formerNum > 0 ? tailInputNum + 1 : tailInputNum;
    outPointNumAligned = (outPointNum + alignNum - 1) / alignNum * alignNum;
    formerLength = formerInputNum * featsNum;
    tailLength = tailInputNum * featsNum;
    totalLengthAligned = 0;
}

ge::graphStatus DynamicScatterTiling::Init()
{
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t *currentWorkSpace = tilingContext->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;
    auto featsShape = tilingContext->GetInputShape(0)->GetStorageShape();
    pointNum = featsShape.GetDim(DIM_INDEX0);
    featsNum = featsShape.GetDim(DIM_INDEX1);
    totalLength = featsShape.GetShapeSize();
    auto reducedFeatsShape = tilingContext->GetOutputShape(0)->GetStorageShape();
    outPointNum = reducedFeatsShape.GetDim(DIM_INDEX0);

    auto platformInfo = tilingContext->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    coreNum = GetNeedCoreNum(coreNum);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    
    auto attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const char* reduceTypePtr = attrs->GetAttrPointer<char>(DIM_INDEX0);
    std::string reduceType(reduceTypePtr);
    if (reduceType != "max" && reduceType != "sum") {
        return ge::GRAPH_PARAM_INVALID;
    }
    auto dType = tilingContext->GetInputDesc(0)->GetDataType();
    SetTilingKeyMode(dType, REDUCE_TYPE_MAP[reduceType]);
    CalTilingAligned(dType);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicScatterTiling::RunKernelTiling()
{
    tilingContext->SetBlockDim(coreNum);
    tilingData.set_totalLength(totalLength);
    tilingData.set_formerNum(formerNum);
    tilingData.set_tailNum(tailNum);
    tilingData.set_formerLength(formerLength);
    tilingData.set_tailLength(tailLength);
    tilingData.set_alignNum(alignNum);
    tilingData.set_totalLengthAligned(totalLengthAligned);
    tilingData.set_formerInputNum(formerInputNum);
    tilingData.set_tailInputNum(tailInputNum);
    tilingData.set_featsNum(featsNum);
    tilingData.set_outPointNum(outPointNum);
    tilingData.set_outPointNumAligned(outPointNumAligned);
    tilingData.set_featsAligned(featsAligned);
    tilingData.set_tileLength(tileLength);
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingDynamicScatter(gert::TilingContext* context)
{
    DynamicScatterTiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.RunKernelTiling();
}
} // optiling


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* featShape = context->GetInputShape(0);
    if (featShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outShape->SetDim(0, -1);
    outShape->SetDim(1, featShape->GetDim(1));
    return GRAPH_SUCCESS;
}
} // ge


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
        this->Input("coors_map")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("reduced_feats")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("reduce_type").AttrType(REQUIRED).String("max");
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingDynamicScatter);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(DynamicScatter);
}
