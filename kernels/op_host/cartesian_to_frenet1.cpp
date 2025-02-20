#include "cartesian_to_frenet1.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
    constexpr float AVALIABLE_UB_RATIO = 0.3;
    constexpr uint32_t DATA_BLOCK_SIZE = 32;
    constexpr uint32_t ELEM_BYTE_SIZE = sizeof(float);
    constexpr uint32_t IDX_BYTE_SIZE = sizeof(int32_t);
    constexpr uint32_t ALIGN_NUM = DATA_BLOCK_SIZE/ELEM_BYTE_SIZE;
    constexpr uint32_t DIST_VEC_INPUT_IDX = 0;
    constexpr uint32_t MIN_IDX_OUTPUT_IDX = 0;
    constexpr uint32_t BACK_IDX_OUTPUT_IDX = 1;
    constexpr uint32_t BATCH_SIZE_IDX = 0;
    constexpr uint32_t NUM_POINTS_IDX = 1;
    constexpr uint32_t NUM_POLY_LINE_POINTS_IDX = 2;
    constexpr uint32_t POINT_DIM_IDX = 3;
}

namespace optiling {
const uint32_t BLOCK_DIM = 1;

static ge::graphStatus TilingForCartesianToFrenet1(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    /* Get system info: aivNum, ubSize */
    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto aivNum = ascendplatformInfo.GetCoreNumAiv();
    context->SetBlockDim(aivNum);
    uint64_t ubSize;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize *= AVALIABLE_UB_RATIO;
    if (aivNum == 0) {
        return ge::GRAPH_FAILED;
    }
    // /* Get shape info */
    auto dTypePtr = context->GetInputDesc(DIST_VEC_INPUT_IDX);
    auto distVecShapePtr = context->GetInputShape(DIST_VEC_INPUT_IDX);
    auto minIdxShapePtr = context->GetOutputShape(MIN_IDX_OUTPUT_IDX);
    auto backIdxShapePtr = context->GetOutputShape(BACK_IDX_OUTPUT_IDX);
    if (distVecShapePtr == nullptr || minIdxShapePtr == nullptr || backIdxShapePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto distVecShape = distVecShapePtr->GetStorageShape();
    uint32_t batchSize = distVecShape.GetDim(BATCH_SIZE_IDX);
    uint32_t numPoints = distVecShape.GetDim(NUM_POINTS_IDX);
    uint32_t numPolyLinePoints = distVecShape.GetDim(NUM_POLY_LINE_POINTS_IDX);
    uint32_t pointDim = distVecShape.GetDim(POINT_DIM_IDX);
    uint32_t taskLength = numPoints * pointDim;

    // /* Calculate tiling info */
    uint32_t bigCoreCount = (batchSize * numPoints) % aivNum;
    uint32_t taskSize = pointDim * numPolyLinePoints * ELEM_BYTE_SIZE;  // 一个task（指k2*2大小的数据）需要占多少内存，在对齐32字节后的block中前多少个字节为有效数据
    uint32_t taskSizeElem = pointDim * numPolyLinePoints;
    uint32_t taskSizeAlignedInit = (taskSize + DATA_BLOCK_SIZE - 1) / (DATA_BLOCK_SIZE) * DATA_BLOCK_SIZE; // 向上align后一个task的大小（对齐32字节的datablock）
    uint32_t taskSizeAligned = (taskSizeAlignedInit / 2) % 32 == 0 ? taskSizeAlignedInit : taskSizeAlignedInit + DATA_BLOCK_SIZE; // 对齐64字节后的长度
    uint32_t copyInAlignNum = (taskSize + DATA_BLOCK_SIZE * 2 - 2) / (DATA_BLOCK_SIZE * 2) * DATA_BLOCK_SIZE * 2;

    uint32_t rightPadding = (taskSizeAlignedInit - taskSize) / ELEM_BYTE_SIZE;
    uint32_t dstStride = (taskSizeAlignedInit / 2) % 32 == 0 ? 0 : 1;

    uint32_t totalTaskNum = batchSize * numPoints;  // 一共有多少个task(一个task的大小为k2*2)
    uint32_t usedCoreNum = std::min(totalTaskNum, aivNum);  // 一共有多少个core
    uint32_t avgTaskNum = totalTaskNum / aivNum;  // 平均每个核处理多少个task（大核处理avgTaskNum+1个task）

    uint32_t tileLength = (ubSize / taskSizeAligned) * taskSizeAligned; // 一个ub可以装下的align后的task的大小（每次搬运多少个字节）
    uint32_t tileTaskNum = ubSize / taskSizeAligned; // 一次能搬运多少个task进入ub

    uint32_t formerTileNum = (avgTaskNum + 1) / tileTaskNum; // 大核上需要搬运几次；如果一个核上的所有task可以fit in ub，只用搬一步，formerTileNum为0，remainder部分为需要搬入的长度。
    uint32_t formerTileRemainder = (avgTaskNum + 1) % tileTaskNum;

    uint32_t tailTileNum = avgTaskNum / tileTaskNum; // 小核上需要搬入几次
    uint32_t tailTileRemainder = (avgTaskNum) % tileTaskNum;

    uint32_t numTaskCurCore_b = avgTaskNum + 1;
    uint32_t TaskLengthCurCore_b = numTaskCurCore_b * numPolyLinePoints * pointDim;
    uint32_t TaskLengthCurCore_s = avgTaskNum * numPolyLinePoints * pointDim;

    uint32_t tileTaskNum_b = std::min(tileTaskNum, numTaskCurCore_b);
    uint32_t tileTaskNum_s = std::min(tileTaskNum, avgTaskNum);
    uint32_t taskResultSizeAligned_b = (tileTaskNum_b * IDX_BYTE_SIZE + DATA_BLOCK_SIZE - 1) / DATA_BLOCK_SIZE * DATA_BLOCK_SIZE;
    uint32_t axisSizeAligned_b = tileTaskNum_b * (numPolyLinePoints * ELEM_BYTE_SIZE + DATA_BLOCK_SIZE - 1) / DATA_BLOCK_SIZE * DATA_BLOCK_SIZE;
    uint32_t taskResultSizeAligned_s = (tileTaskNum_s * IDX_BYTE_SIZE + DATA_BLOCK_SIZE - 1) / DATA_BLOCK_SIZE * DATA_BLOCK_SIZE;
    uint32_t axisSizeAligned_s = tileTaskNum_s * (numPolyLinePoints * ELEM_BYTE_SIZE + DATA_BLOCK_SIZE - 1) / DATA_BLOCK_SIZE * DATA_BLOCK_SIZE;

    /* Set tilingData */
    CartesianToFrenet1TilingData tilingData;
    tilingData.set_numPolyLinePoints(numPolyLinePoints);
    tilingData.set_pointDim(pointDim);
    tilingData.set_taskSize(taskSize);
    tilingData.set_taskSizeElem(taskSizeElem);
    tilingData.set_taskSizeAligned(taskSizeAligned);
    tilingData.set_copyInAlignNum(copyInAlignNum);

    tilingData.set_dstStride(dstStride);
    tilingData.set_rightPadding(rightPadding);

    tilingData.set_tileLength(tileLength);
    tilingData.set_tileTaskNum(tileTaskNum);
    tilingData.set_formerTileNum(formerTileNum);
    tilingData.set_formerTileRemainder(formerTileRemainder);
    tilingData.set_tailTileNum(tailTileNum);
    tilingData.set_tailTileRemainder(tailTileRemainder);
    tilingData.set_bigCoreCount(bigCoreCount);
    tilingData.set_usedCoreNum(usedCoreNum);
    tilingData.set_avgTaskNum(avgTaskNum);

    tilingData.set_numTaskCurCore_b(numTaskCurCore_b);
    tilingData.set_TaskLengthCurCore_b(TaskLengthCurCore_b);
    tilingData.set_TaskLengthCurCore_s(TaskLengthCurCore_s);

    tilingData.set_tileTaskNum_b(tileTaskNum_b);
    tilingData.set_tileTaskNum_s(tileTaskNum_s);
    tilingData.set_taskResultSizeAligned_b(taskResultSizeAligned_b);
    tilingData.set_axisSizeAligned_b(axisSizeAligned_b);
    tilingData.set_taskResultSizeAligned_s(taskResultSizeAligned_s);
    tilingData.set_axisSizeAligned_s(axisSizeAligned_s);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t systemWorkspaceSize = ascendplatformInfo.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShapeForCartesianToFrenet1(gert::InferShapeContext* context)
{
    const gert::Shape* dist_vec_shape = context->GetInputShape(DIST_VEC_INPUT_IDX);
    gert::Shape* min_idx_shape = context->GetOutputShape(MIN_IDX_OUTPUT_IDX);
    gert::Shape* back_idx_shape = context->GetOutputShape(BACK_IDX_OUTPUT_IDX);
    if (dist_vec_shape == nullptr || min_idx_shape == nullptr || back_idx_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t batchSize = dist_vec_shape->GetDim(BATCH_SIZE_IDX);
    int64_t numTargetPoint = dist_vec_shape->GetDim(NUM_POINTS_IDX);
    int64_t numPolyLinePoint = dist_vec_shape->GetDim(NUM_POLY_LINE_POINTS_IDX);
    *min_idx_shape = {batchSize, numTargetPoint};
    *back_idx_shape = {batchSize, numTargetPoint};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForCartesianToFrenet1(gert::InferDataTypeContext* context)
{
    const ge::DataType pt_type = context->GetInputDataType(DIST_VEC_INPUT_IDX);
    const ge::DataType idx_dtype = DT_INT32;
    context->SetOutputDataType(MIN_IDX_OUTPUT_IDX, idx_dtype);
    context->SetOutputDataType(BACK_IDX_OUTPUT_IDX, idx_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class CartesianToFrenet1 : public OpDef {
public:
    explicit CartesianToFrenet1(const char* name) : OpDef(name)
    {
        this->Input("dist_vec")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("min_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("back_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForCartesianToFrenet1).SetInferDataType(ge::InferDataTypeForCartesianToFrenet1);
        this->AICore().SetTiling(optiling::TilingForCartesianToFrenet1);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(CartesianToFrenet1);
}
