/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#include "fake_tensor_quant_tiling.h"

#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace fake_tensor_quant {
constexpr uint32_t INPUTS_INPUT_INDEX = 0;
constexpr uint32_t AMAX_INPUT_INDEX = 1;
constexpr uint32_t NUM_BITS_ATTR_INDEX = 0;
constexpr uint32_t IS_UNSIGNED_ATTR_INDEX = 1;
constexpr uint32_t NARROW_RANGE_ATTR_INDEX = 2;
constexpr uint32_t kMaxValue = std::numeric_limits<uint32_t>::max();
constexpr uint64_t UB_RESERVE_BYTES = 10 * 1024;

const char* const opName = "FakeTensorQuant";

constexpr int32_t TILING_FP32_BIT = 1;
constexpr int32_t TILING_FP16_BIT = 2;

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

bool CheckParamInfo(gert::TilingContext* context, uint32_t& inputDataNum, ge::DataType& dType)
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
    const gert::Shape inputsShape = tensorInputs->GetStorageShape();
    auto inputsTotalNum = inputsShape.GetShapeSize();
    auto amaxTotalNum = amaxShape.GetShapeSize();
    OPS_LOG_E_IF(
        opName, amaxTotalNum != 1, return false, "amaxTotalNum must be 1 , but amaxTotalNum is %d", amaxTotalNum);
    OPS_LOG_E_IF(opName,
        inputsTotalNum == gert::Shape::kInvalidDimValue || inputsTotalNum == 0 || inputsTotalNum > kMaxValue,
        return false, "inputs elem number is illegal %d", inputsTotalNum);
    inputDataNum = static_cast<uint32_t>(inputsTotalNum);

    return true;
}

void SplitAivCoreNum(FakeTensorQuantTilingData& tilingData, const int64_t dataNum, const uint32_t aivNum)
{
    if (aivNum == 0) {
        return;
    }
    uint32_t headCoreNum = dataNum % aivNum; // 头核个数，剩余的元素给头核处理
    tilingData.set_headCoreNum(headCoreNum);
}

ge::graphStatus CalculateElemSizeInUb(FakeTensorQuantTilingData& tilingData, const ge::DataType dType,
    const uint32_t baseElemNum, const uint64_t ubSize)
{
    // 每次计算为 inputs output2者占用内存，2者size一致，考虑并行搬运，就是2 * 2 * dtype
    // amax_不进行搬运，只使用一份内存
    constexpr uint32_t doubleBuffer = 2;
    constexpr uint32_t memUbCount = 2;
    constexpr uint32_t UBAlignByte = 32;
    uint64_t UBAlign = UBAlignByte / ge::GetSizeByDataType(dType);
    // 每次计算元素占用的字节数
    uint64_t sizePerElem = (doubleBuffer * memUbCount + 1) * ge::GetSizeByDataType(dType);
    uint64_t maxCountInUbResult = ubSize / sizePerElem; // 每片ub中可以最多可以计算多少次，也就是搬运元素个数
    maxCountInUbResult = (maxCountInUbResult / UBAlign) * UBAlign;
    if (maxCountInUbResult > kMaxValue) {
        OP_LOGE(opName, "maxCountInUb is too large.");
        return ge::GRAPH_FAILED;
    }
    uint32_t maxCountInUb = static_cast<uint32_t>(maxCountInUbResult);
    // 头核/尾核每次可以拷贝的元素个数
    uint32_t normalCopyElemNum = (baseElemNum < maxCountInUb) ? baseElemNum : maxCountInUb;
    OPS_LOG_E_IF(opName, normalCopyElemNum == 0, return ge::GRAPH_FAILED, "normalCopyElemNum cannot be 0");
    tilingData.set_normalCopyElemNum(normalCopyElemNum);
    uint32_t eachHeadCoreElemNum = baseElemNum; // 每个头核处理的元素总个数
    tilingData.set_eachHeadCoreElemNum(eachHeadCoreElemNum);
    uint32_t eachTailCoreElemNum = baseElemNum - 1; // 每个尾核处理的元素总个数
    tilingData.set_eachTailCoreElemNum(eachTailCoreElemNum);
    // 头核总的搬运次数
    uint32_t headCoreCopyNum =
        (eachHeadCoreElemNum == normalCopyElemNum) ? 1 : Ceil(eachHeadCoreElemNum, normalCopyElemNum);
    tilingData.set_headCoreCopyNum(headCoreCopyNum);
    // 尾核总的搬运次数
    uint32_t tailCoreCopyNum = (eachTailCoreElemNum == 0) ? 0 : Ceil(eachTailCoreElemNum, normalCopyElemNum);
    tilingData.set_tailCoreCopyNum(tailCoreCopyNum);

    // 头核最后一次搬运的个数
    uint32_t headCoreLastCopyElemNum = eachHeadCoreElemNum - (headCoreCopyNum - 1) * normalCopyElemNum;
    tilingData.set_headCoreLastCopyElemNum(headCoreLastCopyElemNum);
    // 尾核最后一次搬运的个数
    uint32_t tailCoreLastCopyElemNum = (tailCoreCopyNum == 0) ? 0 : (eachTailCoreElemNum - (tailCoreCopyNum - 1) * normalCopyElemNum);
    tilingData.set_tailCoreLastCopyElemNum(tailCoreLastCopyElemNum);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus CalculateBound(gert::TilingContext* context, FakeTensorQuantTilingData& tilingData)
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
    uint32_t dataNum = 0;
    ge::DataType dType = ge::DT_FLOAT;
    if (!CheckParamInfo(context, dataNum, dType)) {
        OP_LOGE(opName, "CheckParamSize failed");
        return ge::GRAPH_FAILED;
    }
    if (aivNum == 0) {
        return ge::GRAPH_FAILED;
    }
    FakeTensorQuantTilingData tilingData;
    SplitAivCoreNum(tilingData, dataNum, aivNum);

    uint32_t baseElemNum = dataNum / aivNum + 1;
    CHECK_ON_SUCCESS(CalculateElemSizeInUb(tilingData, dType, baseElemNum, ubSize));
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
} // namespace fake_tensor_quant
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
class FakeTensorQuant : public OpDef {
public:
    explicit FakeTensorQuant(const char* name) : OpDef(name)
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
        this->Attr("num_bits").Int();
        this->Attr("is_unsigned").Bool();
        this->Attr("narrow_range").Bool();
        this->AICore().SetTiling(optiling::fake_tensor_quant::TilingFunc);
        this->SetInferShape(ge::InferShape);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(FakeTensorQuant);
} // namespace ops