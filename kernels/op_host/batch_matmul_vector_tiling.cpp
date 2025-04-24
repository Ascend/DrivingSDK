#include "ge/utils.h"
#include "batch_matmul_vector_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace std;
using namespace AscendC;

namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
const uint32_t DIM_THRESHOLD = 3;
const uint32_t ALIGN_NUM = 64;

static int32_t GetCeilInt(int32_t value1, int32_t value2)
{
    if (value2 == 0) {
        return value1;
    }
    return static_cast<int32_t>((value1 + value2 - 1) / value2);
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    BatchMatmulVectorTilingData tiling;
    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    const auto coreNumber = ascendplatformInfo.GetCoreNumAiv();

    CHECK_NULLPTR(context->GetInputTensor(0));
    const uint32_t totalResult = context->GetInputTensor(0)->GetShapeSize();
    const auto projectionMatShape = context->GetInputTensor(0)->GetStorageShape();
    
    const auto dimNum = projectionMatShape.GetDimNum();
    if (dimNum < DIM_THRESHOLD) {
        return ge::GRAPH_FAILED;
    }

    // 获取最后两个维度大小
    const auto dimSizeSecondLast = projectionMatShape.GetDim(dimNum - 2);
    const auto dimSizeLast = projectionMatShape.GetDim(dimNum - 1);
    
    const uint32_t ptsTotal = context->GetInputTensor(1)->GetShapeSize();
    if (dimSizeLast == 0) {
        return ge::GRAPH_FAILED;
    }

    const auto batchSize = totalResult / dimSizeLast;
    
    int32_t coreData = GetCeilInt(batchSize, coreNumber);
    coreData = GetCeilInt(coreData, ALIGN_NUM) * ALIGN_NUM;
    
    const int32_t coreUsed = GetCeilInt(batchSize, coreData);
    int32_t coreLast = coreData;
    if (coreData == 0) {
        return ge::GRAPH_FAILED;
    }
    if (batchSize % coreData != 0) {
        coreLast = batchSize % coreData;
    }

    uint64_t availableUbSize;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbSize);
    
    // UB空间分配考虑buffer复用因子
    constexpr int32_t bufferMultiplier = 24 * 4;  // 原代码中的24 * 4
    availableUbSize = availableUbSize / bufferMultiplier;
    availableUbSize = GetCeilInt(availableUbSize, ALIGN_NUM) * ALIGN_NUM;

    context->SetBlockDim(coreUsed);
    
    // 设置平铺参数
    tiling.set_coreData(coreData);
    tiling.set_coreUsed(coreUsed);
    tiling.set_copyLoop(coreData / availableUbSize);
    tiling.set_copyTail(coreData % availableUbSize);
    tiling.set_lastCopyLoop(coreLast / availableUbSize);
    tiling.set_lastCopyTail(coreLast % availableUbSize);
    tiling.set_availableUbSize(availableUbSize);
    tiling.set_totalResult(totalResult);
    tiling.set_ptsTotal(ptsTotal);
    tiling.set_dimSizeSecondLast(dimSizeSecondLast);
    tiling.set_dimSizeLast(dimSizeLast);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 1;
    
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    gert::Shape* yShape = context->GetOutputShape(0);
    gert::Shape* indicesOutShape = context->GetOutputShape(1);
    
    if (yShape == nullptr || indicesOutShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
   
    return GRAPH_SUCCESS;
}
}

namespace ops {
class BatchMatmulVector : public OpDef {
public:
    explicit BatchMatmulVector(const char* name) : OpDef(name)
    {
        this->Input("projection_mat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("pts_extend")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("point_2d")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(BatchMatmulVector);
}