/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include <cmath>
#include "points_in_box_all_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace ge;
using namespace std;
using namespace AscendC;
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static int32_t GetCeilInt(int32_t value1, int32_t value2)
{
    if (value2 == 0) {
        return value1;
    }
    return static_cast<int32_t>((value1 + value2 - 1) / value2);
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    PointsInBoxAllTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    auto coreNumber = ascendplatformInfo.GetCoreNumAiv();
    if (context->GetInputTensor(0) == nullptr || context->GetInputTensor(1) == nullptr || context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t totalResult = context->GetInputTensor(1)->GetShapeSize() / 3;
    auto boxShape = context->GetInputTensor(0)->GetStorageShape();
    auto pointShape = context->GetInputTensor(1)->GetStorageShape();
    uint32_t numBoxes = boxShape.GetDim(2);
    int32_t batchSize = boxShape.GetDim(0);

    int32_t coreData;
    int32_t usedCoreNum;
    int32_t coreLast;
    coreData = GetCeilInt(totalResult, coreNumber);
    usedCoreNum = GetCeilInt(totalResult, coreData);
    coreLast = coreData;
    if (coreData == 0) {
        return ge::GRAPH_FAILED;
    }
    if (totalResult % coreData != 0) {
        coreLast = totalResult % coreData;
    }
    uint64_t availableUbSize;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbSize);
    availableUbSize = (sqrt(248 * 248 + 4 * 36 * availableUbSize) - 248) / 2 / 36;
    availableUbSize = GetCeilInt(availableUbSize - 32, 32) * 32;
    context->SetBlockDim(usedCoreNum);
    tiling.set_coreData(coreData);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_copyLoop(coreData / availableUbSize);
    tiling.set_copyTail(coreData % availableUbSize);
    tiling.set_lastCopyLoop(coreLast / availableUbSize);
    tiling.set_lastCopyTail(coreLast % availableUbSize);
    tiling.set_npoints(pointShape.GetDim(1));
    tiling.set_boxNumber(boxShape.GetDim(2));
    tiling.set_availableUbSize(availableUbSize);
    tiling.set_batchSize(batchSize);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShapeForPointsInBoxAll(gert::InferShapeContext* context)
{
    const gert::Shape* pointShape = context->GetInputShape(1);
    if (pointShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* boxShape = context->GetInputShape(0);
    if (boxShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outputShape->SetDimNum(0);
    outputShape->AppendDim(pointShape->GetDim(0));
    outputShape->AppendDim(pointShape->GetDim(1));
    outputShape->AppendDim(boxShape->GetDim(2));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForPointsInBoxAll(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class PointsInBoxAll : public OpDef {
public:
    explicit PointsInBoxAll(const char* name) : OpDef(name)
    {
        this->Input("boxes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("pts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("boxes_idx_of_points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForPointsInBoxAll)
            .SetInferDataType(ge::InferDataTypeForPointsInBoxAll);
        
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(PointsInBoxAll);
}
