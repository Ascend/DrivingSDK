/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */

#include "radius_tiling.h"
#include "common.h"
#include "ge/utils.h"

namespace {
    constexpr uint32_t X_INDEX = 0;
    constexpr uint32_t Y_INDEX = 1;
    constexpr uint32_t PTR_X_INDEX = 2;
    constexpr uint32_t PTR_Y_INDEX = 3;
    constexpr uint32_t OUT_TEMP_INDEX = 0;
    constexpr uint32_t OUT_FINAL_INDEX = 1;
    constexpr uint32_t NUM_NEIGHBORS_INDEX = 2;
    constexpr uint32_t COORDINATE_DIM = 2; // two-dimensional coordinates
    constexpr uint32_t BLOCK_BYTES = 32; // Single Block requires 32B alignment.
    constexpr uint32_t BUFFER_SIZE_32KB = 32768;
    constexpr uint32_t BUFFER_SIZE_16KB = 16384;
    constexpr uint32_t BUFFER_SIZE_8KB = 8192;
    constexpr uint32_t ALIGN_NUM = 8;
}

namespace optiling {
/****************class impl*****************/
static ge::graphStatus TilingForRadius(gert::TilingContext *context)
{
    uint32_t batchPerCore;
    uint32_t batchPerCoreTail;
    uint32_t headCoreNum;
    CHECK_NULLPTR(context);
    const gert::StorageShape *xShape = context->GetInputShape(0); // [2, num_points_x]
    const gert::StorageShape *yShape = context->GetInputShape(1); // [2, num_points_y]
    const gert::StorageShape *ptrXShape = context->GetInputShape(2); // [batch_size + 1]
    const gert::StorageShape *ptrYShape = context->GetInputShape(3); // [batch_size + 1]
    const gert::RuntimeAttrs *attr = context->GetAttrs();
    auto platformInfoPtr = context->GetPlatformInfo();

    CHECK_NULLPTR(xShape);
    CHECK_NULLPTR(yShape);
    CHECK_NULLPTR(ptrXShape);
    CHECK_NULLPTR(ptrYShape);
    CHECK_NULLPTR(platformInfoPtr);
    CHECK_NULLPTR(attr);

    auto platformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t batchSize = ptrXShape->GetStorageShape().GetDim(0) - 1;
    uint32_t numPointsX = xShape->GetStorageShape().GetDim(1);
    uint32_t numPointsY = yShape->GetStorageShape().GetDim(1);
    float r = *(attr->GetAttrPointer<float>(0));
    r = r * r; // Setting the radius to the square of the original radius can save one square root calculation when calculating the distance between two points.
    uint32_t maxNumNeighbors = *(attr->GetAttrPointer<int32_t>(1));
    
    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    uint32_t usedCoreNum = coreNum;
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    if (batchSize >= coreNum) {
        batchPerCore = (batchSize + coreNum) / coreNum;
        batchPerCoreTail = batchPerCore - 1;
        headCoreNum = batchSize - batchPerCoreTail * coreNum;
        usedCoreNum = coreNum;
    } else {
        batchPerCore = 1;
        batchPerCoreTail = 0;
        headCoreNum = batchSize;
        usedCoreNum = batchSize;
    }

    uint32_t bufferSizePtr = ((batchPerCore + 1) * sizeof(uint32_t) + BLOCK_BYTES) / BLOCK_BYTES * BLOCK_BYTES;
    uint32_t bufferSizePoints = BUFFER_SIZE_32KB;
    uint32_t numLocalPtr = bufferSizePtr / sizeof(uint32_t);
    uint32_t numLocalPoints = bufferSizePoints / 2 / sizeof(float);

    auto systemWorkspaceSize = platformInfo.GetLibApiWorkSpaceSize();
    auto usrWorkspaceSize = (coreNum + 1) * BLOCK_BYTES;
    auto currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] = systemWorkspaceSize + usrWorkspaceSize;
    
    RadiusTilingData TilingData;
    TilingData.set_batchSize(batchSize);
    TilingData.set_numPointsX(numPointsX);
    TilingData.set_numPointsY(numPointsY);
    TilingData.set_maxNumNeighbors(maxNumNeighbors);
    TilingData.set_usedCoreNum(usedCoreNum);
    TilingData.set_headCoreNum(headCoreNum);
    TilingData.set_batchPerCore(batchPerCore);
    TilingData.set_batchPerCoreTail(batchPerCoreTail);
    TilingData.set_bufferSizePtr(bufferSizePtr);
    TilingData.set_bufferSizePoints(bufferSizePoints);
    TilingData.set_numLocalPtr(numLocalPtr);
    TilingData.set_numLocalPoints(numLocalPoints);
    
    TilingData.set_r(r);
    
    context->SetBlockDim(usedCoreNum);
    CHECK_NULLPTR(context->GetRawTilingData());
    TilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InfershapeForRadius(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(X_INDEX); // [2, num_points_x]
    const gert::Shape *yShape = context->GetInputShape(Y_INDEX); // [2, num_points_x]
    const gert::Shape *ptrXShape = context->GetInputShape(PTR_X_INDEX); // [batch_size + 1]
    
    gert::Shape *outTempShape = context->GetOutputShape(OUT_TEMP_INDEX); // [2, num_points_y * max_num_neighbors]
    gert::Shape *outFinalShape = context->GetOutputShape(OUT_FINAL_INDEX); // [2, num_points_y * max_num_neighbors]
    gert::Shape *numNeighborsShape = context->GetOutputShape(NUM_NEIGHBORS_INDEX); // [8]
    const gert::RuntimeAttrs *attr = context->GetAttrs();

    CHECK_NULLPTR(xShape);
    CHECK_NULLPTR(yShape);
    CHECK_NULLPTR(ptrXShape);
    CHECK_NULLPTR(outTempShape);
    CHECK_NULLPTR(outFinalShape);
    CHECK_NULLPTR(numNeighborsShape);
    CHECK_NULLPTR(attr);

    uint32_t numPointsX = xShape->GetDim(1);
    uint32_t numPointsY = yShape->GetDim(1);
    uint32_t outtmpdim0 = outTempShape->GetDim(0);
    uint32_t outfinaldim0 = outFinalShape->GetDim(0);
    uint32_t batchSize = ptrXShape->GetDim(0) - 1;
    const int32_t maxNumNeighbors = *attr->GetAttrPointer<int32_t>(1);

    outTempShape->SetDimNum(outtmpdim0);
    outTempShape->SetDim(0, COORDINATE_DIM);
    outTempShape->SetDim(1, numPointsY * maxNumNeighbors);

    outFinalShape->SetDimNum(outfinaldim0);
    outFinalShape->SetDim(0, COORDINATE_DIM);
    outFinalShape->SetDim(1, numPointsY * maxNumNeighbors);
    
    numNeighborsShape->SetDimNum(1);
    numNeighborsShape->SetDim(0, ALIGN_NUM);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForRadius(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(OUT_TEMP_INDEX, ge::DT_INT32);
    context->SetOutputDataType(OUT_FINAL_INDEX, ge::DT_INT32);
    context->SetOutputDataType(NUM_NEIGHBORS_INDEX, ge::DT_INT32);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Radius : public OpDef {
public:
    explicit Radius(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ptr_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ptr_y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out_temp")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out_final")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("actual_num_neighbors")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("r")
            .AttrType(REQUIRED)
            .Float();
        this->Attr("max_num_neighbors")
            .AttrType(REQUIRED)
            .Int();
        this->SetInferShape(ge::InfershapeForRadius)
            .SetInferDataType(ge::InferDataTypeForRadius);
        this->AICore().SetTiling(optiling::TilingForRadius);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(Radius);
}