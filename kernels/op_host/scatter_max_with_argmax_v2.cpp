/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "scatter_max_with_argmax_v2.h"
#include "common.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "nms3d_normal_tiling.h"


using namespace std;

namespace optiling {
const uint64_t BLOCK_SIZE = 32;
const uint64_t MAX_OUT_LINE =  18000;
const uint64_t MAX_DEAL_NUM =  2048;

static uint64_t GetCeilInt(uint64_t value1, uint64_t value2)
{
    if (value2 == 0) {
        return value1;
    }
    return (value1 + value2 - 1) / value2;
}

static ge::graphStatus GetTaskTilingData(gert::TilingContext* context, ScatterMaxTilingData *tiling, uint64_t coreNum)
{
    uint64_t usedCoreNum = 1;
    uint64_t tilingMode = 0;
    uint64_t outTailNum = 1;

    auto varShape = context->GetInputShape(0)->GetStorageShape();
    auto updatesShape = context->GetInputShape(2)->GetStorageShape();
    auto indicesShape = context->GetInputShape(1)->GetStorageShape();

    auto updatesDim = updatesShape.GetDimNum();
    auto indicesDim = indicesShape.GetDimNum();
    auto outNum = varShape.GetShapeSize();
    auto indicesNum = indicesShape.GetShapeSize();
    auto updatesNum = updatesShape.GetShapeSize();
    if (updatesDim == 1) {
        tilingMode = 1;
    } else if ((updatesDim == indicesDim) && (indicesShape.GetDim(indicesDim - 1) != 1)) {
        tilingMode = 2;
        outTailNum = varShape.GetDim(outNum - 1);
    } else {
        outTailNum = outNum / varShape.GetDim(0);
    }

    if (outTailNum == 0 || indicesNum == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t allrepeatLine = varShape.GetDim(0);
    uint64_t outLineEachTask = MAX_OUT_LINE;
    uint64_t argmaxGap = 1; // a parameters to compute value in argmax
    for (uint64_t i = 1; i < indicesDim; i++) {
        argmaxGap *= indicesShape.GetDim(i);
    }

    uint64_t taskNumPerCore = 1;
    uint64_t taskNumLastCore = 1;
    uint64_t eachCoreLastTaskLine;
    uint64_t LastCoreLastTaskLine;
    uint64_t outlineEachCore;
    uint64_t outlinelastCore;
    uint64_t taskNum = allrepeatLine; // GetCeilInt(allrepeatLine, 4);
    if (taskNum > coreNum) {
        usedCoreNum = coreNum;
    } else {
        usedCoreNum = taskNum;
    }

    // out's dataNum each core deal with
    outlineEachCore = GetCeilInt(allrepeatLine, usedCoreNum);
    if (outlineEachCore == 0) {
        return ge::GRAPH_FAILED;
    }

    // if outlinelastCore = 0, data deal with in last core equal to outlineEachCore
    outlinelastCore = (allrepeatLine % outlineEachCore == 0) ? outlineEachCore : allrepeatLine % outlineEachCore;
    uint64_t actualUsedCoreNum = GetCeilInt(allrepeatLine, outlineEachCore);
    if (outlineEachCore > outLineEachTask) {
        taskNumPerCore = GetCeilInt(outlineEachCore, outLineEachTask); // taskNum each Core deal with
        eachCoreLastTaskLine = (outlineEachCore % outLineEachTask == 0) ? outLineEachTask : outlineEachCore % outLineEachTask; // dataLine last task deal with
        taskNumLastCore = GetCeilInt(outlinelastCore, outLineEachTask);
        LastCoreLastTaskLine = (outlinelastCore % outLineEachTask == 0) ? outLineEachTask : outlinelastCore % outLineEachTask; // dataLine last task deal with for last Core
    } else {
        taskNumPerCore = 1;
        outLineEachTask = outlineEachCore;
        eachCoreLastTaskLine = outlineEachCore;
        taskNumLastCore = 1;
        LastCoreLastTaskLine = outlineEachCore;
    }
    uint64_t inputIndicesNum = (indicesNum + 7) / 8 * 8;

    context->SetBlockDim(actualUsedCoreNum);
    tiling->set_usedCoreNum(actualUsedCoreNum);
    tiling->set_tilingMode(tilingMode);
    tiling->set_outTailNum(outTailNum);
    tiling->set_outEachCore(outlineEachCore * outTailNum);
    tiling->set_outLastCore(outlinelastCore * outTailNum);
    tiling->set_indicesNum(indicesNum);
    tiling->set_updatesNum(updatesNum);
    tiling->set_outNum(outNum);
    tiling->set_outLineEachTask(outLineEachTask);
    tiling->set_taskNumPerCore(taskNumPerCore);
    tiling->set_taskNumLastCore(taskNumLastCore);
    tiling->set_outeachCoreLastNum(eachCoreLastTaskLine * outTailNum);
    tiling->set_outLastCoreLastNum(LastCoreLastTaskLine * outTailNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ScatterMaxWithArgmaxV2TilingFunc(gert::TilingContext* context)
{
    ScatterMaxTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static uint64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t UB_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, UB_size);

    auto socVersion = ascendcPlatform.GetSocVersion();

    GetTaskTilingData(context, &tiling, coreNum);

    if (context->GetInputShape(1) == nullptr || context->GetInputShape(2) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto updatesShape = context->GetInputShape(2)->GetStorageShape();
    auto indicesShape = context->GetInputShape(1)->GetStorageShape();

    auto indicesDim = indicesShape.GetDimNum();
    auto updatesDim = updatesShape.GetDimNum();
    auto indicesNum = indicesShape.GetShapeSize();
    auto updatesNum = updatesShape.GetShapeSize();

    if (context->GetInputDesc(0) == nullptr || context->GetInputDesc(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dataDtype = context->GetInputDesc(0)->GetDataType();
    auto indicesDtype = context->GetInputDesc(1)->GetDataType();
    uint64_t bytesData = kDataSizeMap[dataDtype];       // now only support float32
    uint64_t bytesIndices = kDataSizeMap[indicesDtype]; // now only support int32

    uint64_t updatesTail = 1;
    if (updatesDim == 0) {
        return ge::GRAPH_FAILED;
    }
    for (uint64_t i = 1; i < updatesDim; i++) {
        updatesTail *= updatesShape.GetDim(i);
    }

    if (socVersion == platform_ascendc::SocVersion::ASCEND310P && updatesTail < BLOCK_SIZE / bytesData) {
        return ge::GRAPH_FAILED;
    }

    bool isOneDeal = (updatesTail / MAX_DEAL_NUM) == 0;
    uint64_t argmaxGap = 1; // a parameters to compute value in argmax
    for (uint64_t i = 1; i < indicesDim; i++) {
        argmaxGap *= indicesShape.GetDim(i);
    }

    auto outLineEachTask = tiling.get_outLineEachTask();
    uint64_t ubAvailableBytes = UB_size - std::min(updatesTail, MAX_DEAL_NUM) * 7 * 4 - outLineEachTask * 4 - 10*1024;

    uint64_t ubIndicesNum;
    uint64_t ubUpdatesNum;
    // Allocate UB space for indices and updates
    if (tiling.get_tilingMode() == 1) {
        uint64_t ubIndices = ubAvailableBytes / (updatesTail + 1) * 1 / BLOCK_SIZE * BLOCK_SIZE;
        uint64_t ubUpdates = ubIndices * updatesTail;
        ubIndicesNum = ubIndices / bytesIndices;
        ubUpdatesNum = ubUpdates / bytesData;
    } else {
        ubIndicesNum = ubAvailableBytes / BLOCK_SIZE * BLOCK_SIZE / bytesIndices;
        ubUpdatesNum = ubIndicesNum * updatesTail;
    }

    uint64_t indicesLoop = indicesNum / ubIndicesNum;
    uint64_t indicesLastNum = indicesNum % ubIndicesNum;
    uint64_t unpdatesLastNum = updatesNum % ubUpdatesNum;

    uint64_t initArgmax = updatesShape.GetDim(0);

    bool isAligned = false;
    if ((updatesTail % (BLOCK_SIZE / bytesData) == 0) && (updatesTail % (BLOCK_SIZE / bytesIndices) == 0)) {
        isAligned = true;
    }

    tiling.set_ubIndicesNum(ubIndicesNum);
    tiling.set_ubUpdatesNum(ubUpdatesNum);
    tiling.set_indicesLoop(indicesLoop);
    tiling.set_indicesLastNum(indicesLastNum);
    tiling.set_unpdatesLastNum(unpdatesLastNum);
    tiling.set_argmaxGap(argmaxGap);
    tiling.set_initArgmax(initArgmax);
    tiling.set_isAligned(isAligned);
    tiling.set_updatesTail(updatesTail);
    tiling.set_isOneDeal(isOneDeal);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus ScatterMaxWithArgmaxV2InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    gert::Shape* argmax_shape = context->GetOutputShape(1);
    if (x1_shape == nullptr || y_shape == nullptr || argmax_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *y_shape = *x1_shape;
    *argmax_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus ScatterMaxWithArgmaxV2InferDtype(gert::InferDataTypeContext *context)
{
    const ge::DataType var_dtype = context->GetInputDataType(0);
    const ge::DataType indices_dtype = context->GetInputDataType(1);
    context->SetOutputDataType(0, var_dtype);
    context->SetOutputDataType(1, indices_dtype);
    return GRAPH_SUCCESS;
}

}

namespace ops {
class ScatterMaxWithArgmaxV2 : public OpDef {
public:
    explicit ScatterMaxWithArgmaxV2(const char* name) : OpDef(name)
    {
        this->Input("var")
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
        this->Input("updates")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("argmax")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::ScatterMaxWithArgmaxV2InferShape)
             .SetInferDataType(ge::ScatterMaxWithArgmaxV2InferDtype);

        this->AICore()
            .SetTiling(optiling::ScatterMaxWithArgmaxV2TilingFunc);
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(ScatterMaxWithArgmaxV2);
}