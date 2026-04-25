/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#include "fake_tensor_quant_with_axis_tiling.h"

#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace fake_tensor_quant_with_axis {
constexpr uint32_t INPUTS_INPUT_INDEX = 0;
constexpr uint32_t AMAX_INPUT_INDEX = 1;
constexpr uint32_t AXIS_ATTR_INDEX = 0;
constexpr uint32_t NUM_BITS_ATTR_INDEX = 1;
constexpr uint32_t IS_UNSIGNED_ATTR_INDEX = 2;
constexpr uint32_t NARROW_RANGE_ATTR_INDEX = 3;
constexpr uint32_t kMaxValue = std::numeric_limits<uint32_t>::max();
constexpr uint64_t UB_RESERVE_BYTES = 16 * 1024;

const char* const opName = "FakeTensorQuantWithAxis";

constexpr int32_t TILING_FP32_BIT = 1;
constexpr int32_t TILING_FP16_BIT = 2;

enum class TilingMode {
    NormalMode = 0,
    SpecialMode
};

ge::graphStatus GetTilingKey(gert::TilingContext* context, const ge::DataType dtype)
{
    int32_t key = 0;
    switch (dtype) {
        case ge::DT_FLOAT:
            key = TILING_FP32_BIT;
            break;
        case ge::DT_FLOAT16:
            key = TILING_FP16_BIT;
            break;
        default:
            OP_LOGE(opName, "Unsupport dtype %d", dtype);
            return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(key);
    return ge::GRAPH_SUCCESS;
}

bool GetPlatformInfoAndCoreNum(gert::TilingContext* context, uint32_t& aivNum)
{
    auto platformInfo = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(opName, platformInfo, return false);

    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    aivNum = ascendcPlatform.GetCoreNumAiv();
    OPS_LOG_E_IF(opName, aivNum == 0, return false, "aivNum cannot be 0");

    context->SetBlockDim(aivNum);
    return true;
}

bool GetUBSize(const gert::TilingContext* context, uint64_t& ubSize)
{
    platform_ascendc::PlatformAscendC ascendcPlatform(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize -= UB_RESERVE_BYTES;
    OPS_LOG_E_IF(opName, ubSize == 0, return false, "ubSize cannot be 0");

    return true;
}

bool GetTaskNum(gert::TilingContext* context, const gert::Shape inputsShape, uint32_t& taskNum, uint32_t& axis,
    TilingMode& tilingMode)
{
    OPS_LOG_E_IF(
        opName, inputsShape.GetDimNum() <= 0, return false, "inputs dims is illegal %d", inputsShape.GetDimNum());

    auto attrPtr = context->GetAttrs();
    OPS_LOG_E_IF(opName, attrPtr == nullptr, return false, "attrPtr is nullptr");
    auto axisptr = attrPtr->GetInt(AXIS_ATTR_INDEX);
    OPS_LOG_E_IF(opName, axisptr == nullptr, return false, "axis is nullptr");
    axis = *axisptr;
    OPS_LOG_E_IF(opName, (axis < 0 || axis >= inputsShape.GetDimNum()), return false, "axis is illegal %d", axis);
    taskNum = inputsShape.GetDim(0);
    if (axis == inputsShape.GetDimNum() - 1) {
        // 如果取最后一维作为amax，普通分核方式会变为单点计算，效率很低，现改为特殊模式
        tilingMode = TilingMode::SpecialMode;
        for (int i = 1; i <= axis - 1; i++) {
            taskNum *= inputsShape.GetDim(i);
        }
    } else {
        for (int i = 1; i <= axis; i++) {
            taskNum *= inputsShape.GetDim(i);
        }
    }
    return true;
}

bool CheckParamInfo(gert::TilingContext* context, uint32_t& dataNumPerTask, uint32_t& taskNum, ge::DataType& dType,
    uint32_t& amaxTotalNum, TilingMode& tilingMode)
{
    auto tensorInputs = context->GetInputTensor(INPUTS_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(opName, tensorInputs, return ge::GRAPH_FAILED)
    auto tensorAmax = context->GetInputTensor(AMAX_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(opName, tensorAmax, return ge::GRAPH_FAILED)

    auto inputsDtype = tensorInputs->GetDataType();
    auto amaxDtype = tensorAmax->GetDataType();
    OPS_LOG_E_IF(opName, inputsDtype != amaxDtype, return false,
        "inputsDtype and amaxDtype must be same , but inputsDtype is %d, amaxDtype is %d", inputsDtype, amaxDtype);
    OPS_LOG_E_IF(opName, (inputsDtype != ge::DT_FLOAT && inputsDtype != ge::DT_FLOAT16), return false,
        "inputsDtype only support DT_FLOAT or DT_FLOAT16 , but inputsDtype is %d", inputsDtype);
    dType = inputsDtype;

    const gert::Shape amaxShape = tensorAmax->GetStorageShape();
    OPS_LOG_E_IF(opName, amaxShape.GetDimNum() != 1, return false, "amax dims is illegal %d", amaxShape.GetDimNum());
    const gert::Shape inputsShape = tensorInputs->GetStorageShape();
    uint32_t axis = 0;
    OPS_LOG_E_IF(
        opName, !GetTaskNum(context, inputsShape, taskNum, axis, tilingMode), return false, "GetTaskNum error");
    amaxTotalNum = amaxShape.GetShapeSize();
    auto inputsAxisDimNum = inputsShape.GetDim(axis);
    OPS_LOG_E_IF(opName, amaxTotalNum != inputsAxisDimNum, return false,
        "amaxTotalNum must be same as inputsAxisDimNum , but amaxTotalNum is %d, inputsAxisDimNum is %d, axis is %d",
        amaxTotalNum, inputsAxisDimNum, axis);
    auto inputsTotalNum = inputsShape.GetShapeSize();
    OPS_LOG_E_IF(opName,
        inputsTotalNum == gert::Shape::kInvalidDimValue || inputsTotalNum == 0 || inputsTotalNum > kMaxValue,
        return false, "inputs elem number is illegal %d", inputsTotalNum);
    OPS_LOG_E_IF(opName, taskNum == 0, return false, "taskNum is 0");
    dataNumPerTask = static_cast<uint32_t>(inputsTotalNum / taskNum);

    return true;
}

void SplitAivCoreNum(FakeTensorQuantWithAxisTilingData& tilingData, const int64_t taskNum, const uint32_t aivNum)
{
    if (aivNum == 0) {
        return;
    }
    uint32_t headCoreNum = taskNum % aivNum; // 头核个数，剩余的元素给头核处理
    tilingData.set_headCoreNum(headCoreNum);
    uint32_t eachTailCoreTaskNum = taskNum / aivNum;
    tilingData.set_eachTailCoreTaskNum(eachTailCoreTaskNum);
    tilingData.set_eachHeadCoreTaskNum(eachTailCoreTaskNum + 1);
}

ge::graphStatus CalculateElemSizeInUb(FakeTensorQuantWithAxisTilingData& tilingData, const ge::DataType dType,
    const uint32_t dataNumPerTask, const uint32_t amaxTotalNum, const uint64_t ubSize)
{
    // 每次计算为 inputs output2者占用内存，2者size一致，考虑并行搬运，就是2 * 2 * dtype
    // amax需要额外得临时内存用于计算，size与input一样，但是不用并行搬运
    constexpr uint32_t doubleBuffer = 2;
    constexpr uint32_t memUbCount = 3;
    // 每次计算元素占用的字节数
    uint64_t sizePerElem = (doubleBuffer * memUbCount) * ge::GetSizeByDataType(dType);
    // 每片ub中可以最多可以计算多少次，也就是搬运元素个数
    uint64_t maxCountInUbResult = ubSize / sizePerElem;
    if (maxCountInUbResult > kMaxValue) {
        OP_LOGE(opName, "maxCountInUb is too large. maxCountInUbResult is %llu", maxCountInUbResult);
        return ge::GRAPH_FAILED;
    }
    uint32_t maxCountInUb = static_cast<uint32_t>(maxCountInUbResult);
    tilingData.set_normalCopyElemNumInOneTask(maxCountInUb);

    uint32_t countInOneTask = (dataNumPerTask <= maxCountInUb) ? 1 : Ceil(dataNumPerTask, maxCountInUb);
    tilingData.set_countInOneTask(countInOneTask);
    uint32_t lastCopyElemNumInOneTask = dataNumPerTask - (countInOneTask - 1) * maxCountInUb;
    tilingData.set_lastCopyElemNumInOneTask(lastCopyElemNumInOneTask);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus CalculateBound(gert::TilingContext* context, FakeTensorQuantWithAxisTilingData& tilingData)
{
    auto attrPtr = context->GetAttrs();
    CHECK_NULLPTR(attrPtr);
    auto num_bits = attrPtr->GetInt(NUM_BITS_ATTR_INDEX);
    auto is_unsigned = attrPtr->GetBool(IS_UNSIGNED_ATTR_INDEX);
    auto narrow_range = attrPtr->GetBool(NARROW_RANGE_ATTR_INDEX);
    CHECK_NULLPTR(num_bits);
    OPS_LOG_E_IF(
        opName, (*num_bits <= 0 || *num_bits > 32), return ge::GRAPH_FAILED, "num_bits is illegal %d", *num_bits);
    CHECK_NULLPTR(is_unsigned);
    CHECK_NULLPTR(narrow_range);
    float bound = (1 << (*num_bits - 1 + int(*is_unsigned))) - 1;
    int32_t max_bound = bound;
    int32_t min_bound = -(bound + !(*narrow_range));
    tilingData.set_maxBound(max_bound);
    tilingData.set_minBound(min_bound);
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus DoOpTiling(gert::TilingContext* context, const uint32_t aivNum, const uint64_t ubSize)
{
    uint32_t dataNumPerTask = 0;
    uint32_t taskNum = 0;
    uint32_t amaxTotalNum = 0;
    ge::DataType dType = ge::DT_FLOAT;
    TilingMode tilingMode = TilingMode::NormalMode;
    if (!CheckParamInfo(context, dataNumPerTask, taskNum, dType, amaxTotalNum, tilingMode)) {
        OP_LOGE(opName, "CheckParamSize failed");
        return ge::GRAPH_FAILED;
    }
    FakeTensorQuantWithAxisTilingData tilingData;
    SplitAivCoreNum(tilingData, taskNum, aivNum);
    tilingData.set_totalElemNumPerTask(dataNumPerTask);
    tilingData.set_amaxTotalNum(amaxTotalNum);
    tilingData.set_tilingMode(static_cast<uint32_t>(tilingMode));

    CHECK_ON_SUCCESS(CalculateElemSizeInUb(tilingData, dType, dataNumPerTask, amaxTotalNum, ubSize));
    CHECK_ON_SUCCESS(CalculateBound(context, tilingData));
    CHECK_ON_SUCCESS(GetTilingKey(context, dType));

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CHECK_NULLPTR(context);
    uint32_t aivNum = 0;

    if (!GetPlatformInfoAndCoreNum(context, aivNum)) {
        OP_LOGE(opName, "GetPlatformInfoAndCoreNum failed");
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    if (!GetUBSize(context, ubSize)) {
        OP_LOGE(opName, "GetUBSize failed");
        return ge::GRAPH_FAILED;
    }

    CHECK_ON_SUCCESS(DoOpTiling(context, aivNum, ubSize));
    return ge::GRAPH_SUCCESS;
}
} // namespace fake_tensor_quant_with_axis
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (!x1_shape || !y_shape) {
        return ge::GRAPH_FAILED;
    }
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class FakeTensorQuantWithAxis : public OpDef {
public:
    explicit FakeTensorQuantWithAxis(const char* name) : OpDef(name)
    {
        this->Input("inputs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous()
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("amax")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous()
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").Int();
        this->Attr("num_bits").Int();
        this->Attr("is_unsigned").Bool();
        this->Attr("narrow_range").Bool();
        this->AICore().SetTiling(optiling::fake_tensor_quant_with_axis::TilingFunc);
        this->SetInferShape(ge::InferShape);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(FakeTensorQuantWithAxis);
} // namespace ops