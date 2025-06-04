/*
* Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
*/

#include <iostream>
#include <map>
#include <vector>
#include <string>

#include "ge/utils.h"
#include "grid_sampler3d_grad_v1_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace std;

namespace optiling {

constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t FP32_BLOCK_NUM = 8;
constexpr size_t INTERPOLATION_MODE_INDEX = 0;
constexpr size_t PADDING_MODE_INDEX = 1;
constexpr size_t ALIGN_CORNERS_INDEX = 2;
constexpr int32_t GRAD_INPUT_INDEX = 0;
constexpr int32_t X_INPUT_INDEX = 1;
constexpr int32_t GRID_INPUT_INDEX = 2;
constexpr int32_t DTYPE_SIZE_32 = 4;
constexpr size_t CHECK_DIM_NUM = 5;
constexpr int BILINEAR = 0;
constexpr int ZEROS = 0;
constexpr int BILINEAR_DIVIDE_UB_NUM = 115;

constexpr uint32_t CONST_SEVENTEEN = 17;
constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;
constexpr uint32_t DIM_INDEX2 = 2;
constexpr uint32_t DIM_INDEX3 = 3;
constexpr uint32_t DIM_INDEX4 = 4;
constexpr uint32_t RESERVED_UB = static_cast<uint32_t>(10 * 1024);
constexpr uint32_t ALIGN_256_BYTES = 256;

template <typename TilingData, int32_t dataTypeLen>
class GridSampler3dGradV1Tiling {
public:
    explicit GridSampler3dGradV1Tiling(InputParamsInfo& param, const uint32_t inputCoreNum, const uint32_t inputUbSize)
    {
        this->batch = param.batch;
        this->coreNum = inputCoreNum;
        this->channel = param.channel;
        this->depth = param.depth;
        this->height = param.height;
        this->width = param.width;
        this->gridD = param.gridD;
        this->gridH = param.gridH;
        this->gridW = param.gridW;
        this->interpolation = param.interpolation;
        this->padding = param.padding;
        this->alignCorners = param.alignCorners;
        this->ubSize = FloorAlign(inputUbSize, BYTE_BLOCK);
        this->dataTypeSize = dataTypeLen;
        this->elementsPerBlock = BYTE_BLOCK / dataTypeSize;
        return;
    }

    void GetTiling(TilingData* tilingData);

private:
    void GetUsedCore();
    void SplitUb();
    void FillTilingData(TilingData* tilingData);
    template <typename T1, typename T2>
    inline T1 FloorAlign(T1 a, T2 b)
    {
        if (b == 0) {
            return 0;
        }
        return (a) / b * b;
    }

private:
    uint32_t batch = 0;
    uint32_t usedCoreNum = 0;
    uint32_t pNumPerCore = 0;
    uint32_t tailPNum = 0;
    uint32_t ubFactorElement = 0;
    uint32_t channel = 0;
    uint32_t depth = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t gridD;
    uint32_t gridH;
    uint32_t gridW;
    uint32_t ubSize = 0;
    uint32_t coreNum = 0;
    uint32_t interpolation = 0;
    uint32_t padding = 0;
    bool alignCorners = false;
    uint32_t group = 0;
    uint8_t dataTypeSize = 0;
    uint8_t elementsPerBlock = 0;
    uint32_t divideUbNum = 1;
    uint32_t extraUbSize = 0;
};

template <typename TilingData, int32_t dataTypeLen>
void GridSampler3dGradV1Tiling<TilingData, dataTypeLen>::GetUsedCore()
{
    uint64_t mulBDHW = batch * gridD * gridH * gridW;
    if (mulBDHW <= this->coreNum) {
        this->usedCoreNum = mulBDHW;
        this->pNumPerCore = 1;
        this->tailPNum = 0;
        return;
    }
    this->pNumPerCore = mulBDHW / this->coreNum;
    this->usedCoreNum = this->coreNum;
    this->tailPNum = mulBDHW % usedCoreNum;
}

template <typename TilingData, int32_t dataTypeLen>
void GridSampler3dGradV1Tiling<TilingData, dataTypeLen>::SplitUb()
{
    uint32_t alignChannel = 0;
    alignChannel = AlignUp(channel, FP32_BLOCK_NUM);

    if (static_cast<int32_t>(interpolation) == 0) {
        divideUbNum = BILINEAR_DIVIDE_UB_NUM;
        extraUbSize = CONST_SEVENTEEN * alignChannel * DTYPE_SIZE_32;
        group = static_cast<uint32_t>(1);
    }
    uint32_t tilingDataSize = AlignUp(sizeof(TilingData), BYTE_BLOCK);
    uint32_t canUseUbSize = FloorAlign(ubSize - tilingDataSize, BYTE_BLOCK);
    if (canUseUbSize <= extraUbSize) {
        ubFactorElement = static_cast<uint32_t>(0);
        return;
    }
    ubFactorElement = FloorAlign((canUseUbSize - extraUbSize) / divideUbNum, ALIGN_256_BYTES) / DTYPE_SIZE_32;
}

template <typename TilingData, int32_t dataTypeLen>
void GridSampler3dGradV1Tiling<TilingData, dataTypeLen>::FillTilingData(TilingData* tilingData)
{
    tilingData->set_batch(batch);
    tilingData->set_pNumPerCore(pNumPerCore);
    tilingData->set_tailPNum(tailPNum);
    tilingData->set_channel(channel);
    tilingData->set_depth(depth);
    tilingData->set_height(height);
    tilingData->set_width(width);
    tilingData->set_gridD(gridD);
    tilingData->set_gridH(gridH);
    tilingData->set_gridW(gridW);
    tilingData->set_blockNum(usedCoreNum);
    tilingData->set_ubFactorElement(ubFactorElement);
    tilingData->set_interpolation(interpolation);
    tilingData->set_padding(padding);
    tilingData->set_alignCorners(alignCorners);
    tilingData->set_group(group);
}

template <typename TilingData, int32_t dataTypeLen>
void GridSampler3dGradV1Tiling<TilingData, dataTypeLen>::GetTiling(TilingData* tilingData)
{
    GetUsedCore();
    SplitUb();
    FillTilingData(tilingData);
}

template <typename TilingData, int32_t dataTypeLen>
void GetGridSampler3dGradV1Tiling(TilingData* tilingData, InputParamsInfo& params, uint32_t coreNum, uint32_t ubSize)
{
    class GridSampler3dGradV1Tiling<TilingData, dataTypeLen> tilingObj(params, coreNum, ubSize);
    tilingObj.GetTiling(tilingData);
}

static ge::graphStatus GetInputInfo(gert::TilingContext* tilingContext, InputParamsInfo& params, ge::DataType dtype)
{
    const gert::StorageShape* gradShape = tilingContext->GetInputShape(GRAD_INPUT_INDEX);
    const gert::StorageShape* xShape = tilingContext->GetInputShape(X_INPUT_INDEX);
    const gert::StorageShape* gridShape = tilingContext->GetInputShape(GRID_INPUT_INDEX);

    if (xShape->GetStorageShape().GetDimNum() != CHECK_DIM_NUM) {
        return ge::GRAPH_FAILED;
    }
    uint32_t outD = gradShape->GetStorageShape().GetDim(DIM_INDEX1);
    uint32_t outH = gradShape->GetStorageShape().GetDim(DIM_INDEX2);
    uint32_t outW = gradShape->GetStorageShape().GetDim(DIM_INDEX3);
    params.batch = xShape->GetStorageShape().GetDim(DIM_INDEX0);
    params.channel = xShape->GetStorageShape().GetDim(DIM_INDEX4);
    params.depth = xShape->GetStorageShape().GetDim(DIM_INDEX1);
    params.height = xShape->GetStorageShape().GetDim(DIM_INDEX2);
    params.width = xShape->GetStorageShape().GetDim(DIM_INDEX3);
    params.gridD = gridShape->GetStorageShape().GetDim(DIM_INDEX1);
    params.gridH = gridShape->GetStorageShape().GetDim(DIM_INDEX2);
    params.gridW = gridShape->GetStorageShape().GetDim(DIM_INDEX3);

    if (outD != params.gridD || outH != params.gridH || outW != params.gridW) {
        return ge::GRAPH_FAILED;
    }

    if (tilingContext->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    params.interpolation = *tilingContext->GetAttrs()->GetAttrPointer<int>(INTERPOLATION_MODE_INDEX);
    params.padding = *tilingContext->GetAttrs()->GetAttrPointer<int>(PADDING_MODE_INDEX);
    params.alignCorners = *tilingContext->GetAttrs()->GetAttrPointer<bool>(ALIGN_CORNERS_INDEX);

    size_t xWorkspaceSize = params.batch * params.channel * params.depth * params.height * params.width * sizeof(float);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4GridSampler3dGradV1(gert::TilingContext* tilingContext)
{
    auto platformInfo = tilingContext->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    uint32_t ubSize = static_cast<uint32_t>(ubSizePlatForm);
    uint32_t availableUb = ubSize - RESERVED_UB;

    ge::DataType inputDatatype = tilingContext->GetInputDesc(0)->GetDataType();

    InputParamsInfo params;
    if (GetInputInfo(tilingContext, params, inputDatatype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    GridSampler3dGradV1TilingData tilingData;

    GetGridSampler3dGradV1Tiling<GridSampler3dGradV1TilingData, DTYPE_SIZE_32>(&tilingData, params, coreNum,
                                                                               availableUb);

    tilingContext->SetBlockDim(tilingData.get_blockNum());
    tilingContext->SetNeedAtomic(true);
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForGridSampler3dGradV1(gert::InferShapeContext* context)
{
    const gert::Shape* gradOut = context->GetInputShape(0);
    const gert::Shape* inputX = context->GetInputShape(1);
    const gert::Shape* inputGrid = context->GetInputShape(2);
    gert::Shape* dX = context->GetOutputShape(0);
    gert::Shape* dGrid = context->GetOutputShape(1);

    CHECK_NULLPTR(gradOut);
    CHECK_NULLPTR(inputX);
    CHECK_NULLPTR(inputGrid);
    CHECK_NULLPTR(dX);
    CHECK_NULLPTR(dGrid);

    int64_t B = inputX->GetDim(0);
    int64_t C = inputX->GetDim(4);
    int64_t D = inputX->GetDim(1);
    int64_t H = inputX->GetDim(2);
    int64_t W = inputX->GetDim(3);
    int64_t b = inputGrid->GetDim(0);
    int64_t d = inputGrid->GetDim(1);
    int64_t h = inputGrid->GetDim(2);
    int64_t w = inputGrid->GetDim(3);

    *dX = {B, D, H, W, C};
    *dGrid = {b, d, h, w, 3};

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForGridSampler3dGradV1(gert::InferDataTypeContext* context)
{
    CHECK_NULLPTR(context);
    const ge::DataType inputX_dtype = context->GetInputDataType(1);
    const ge::DataType inputGrid_dtype = context->GetInputDataType(2);

    context->SetOutputDataType(0, inputX_dtype);
    context->SetOutputDataType(1, inputGrid_dtype);

    return GRAPH_SUCCESS;
}
}

namespace ops {
class GridSampler3dGradV1 : public OpDef {
public:
    explicit GridSampler3dGradV1(const char* name) : OpDef(name)
    {
        this->Input("grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({{ge::FORMAT_ND}})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({{ge::FORMAT_ND}})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("grid")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({{ge::FORMAT_ND}})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({{ge::FORMAT_ND}})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dgrid")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({{ge::FORMAT_ND}})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("interpolation_mode").AttrType(REQUIRED).Int();
        this->Attr("padding_mode").AttrType(REQUIRED).Int();
        this->Attr("align_corners").AttrType(REQUIRED).Bool();

        this->SetInferShape(ge::InferShapeForGridSampler3dGradV1).SetInferDataType(ge::InferDataTypeForGridSampler3dGradV1);
        this->AICore().SetTiling(optiling::Tiling4GridSampler3dGradV1);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(GridSampler3dGradV1);
}  // namespace ops